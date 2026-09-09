# -*- coding: utf-8 -*-
"""WP3 — Unified benchmark engine for evaluation-protocol experiments.

Design notes:
  1. STRICT CALENDAR LAG: the entity-year grid is filled so that shift(1)
     always selects the previous calendar year; persistence uses the
     previous calendar year's observed value.
  2. UNIT-INVARIANT GP/MR-GP: kernel inputs and target are standardised.
  3. PREPROCESSING SCOPE: "global" fits on all rows (leaky shortcut);
     "fold" fits on training rows only.

Design:
  Datasets : A = 21-city China CDW panel (2015-2024)
             B = 33-country Eurostat MSW panel (2004-2023)
             C0 = Beijing+Shenzhen CDW case study (n=20, has CDW substreams)
  Models   : persistence, entity_mean, OLS, Lasso, RF, XGBoost, GP, MR-GP
  Protocol factors (full factorial):
    validation : rolling_origin | loocv | random_5fold
    timing     : contemporaneous | lagged (all covariates at t-1)
    subcomp    : off | on  (target-derived substream features; only where they exist)
    preproc    : fold (scaler/imputer refit inside training data) | global (fit on all rows)
  Seeds     : 0, 1, 2 (matter for RF/XGB and fold shuffling only)

Honesty constraints:
  - evaluation rows are identical across models and timing variants
    (rows whose entity has an observed previous CALENDAR year);
  - LOOCV / random CV appear ONLY as "audited protocols" inside the matrix;
  - persistence is evaluated on exactly the same rows as every other model;
  - no hyperparameter tuning anywhere (documented defaults).

Outputs: output/benchmark_cells.csv, output/benchmark_predictions.parquet,
output/engine_log.txt
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

D1 = Path(__file__).resolve().parents[1]
OUT = D1 / "output"
OUT.mkdir(exist_ok=True)

MODELS = ["persistence", "entity_mean", "ols", "lasso", "rf", "xgb", "gp", "mrgp"]
VALIDATIONS = ["rolling_origin", "loocv", "random_5fold"]
TIMINGS = ["contemporaneous", "lagged"]
PREPROCS = ["fold", "global"]
SEEDS = [0, 1, 2]
N_THREADS = 10


# ----------------------------------------------------------------------------
# fast exact GP with median-heuristic kernel, standardised inputs/target
# ----------------------------------------------------------------------------
class FastGP:
    """Exact GP, RBF kernel, scalar median-heuristic lengthscale.
    REV A: X and y are standardised to unit variance inside fit(), making the
    model invariant to units (tonnes vs kilotonnes)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FastGP":
        self.x_mu = X.mean(0)
        self.x_sd = np.where(X.std(0) > 1e-12, X.std(0), 1.0)
        self.y_mu = float(y.mean())
        self.y_sd = float(y.std()) if y.std() > 1e-12 else 1.0

        Xs = (X - self.x_mu) / self.x_sd
        ys = (y - self.y_mu) / self.y_sd

        r = ys  # zero-mean after standardisation
        self.X = Xs
        sub = Xs if len(Xs) <= 300 else Xs[np.random.RandomState(0).choice(len(Xs), 300, replace=False)]
        D2 = _sqdist(sub, sub)
        off = D2[np.triu_indices(len(sub), 1)]
        med = float(np.median(off[off > 1e-12])) if (off > 1e-12).any() else 1.0
        self.ls2 = max(med, 1e-6)
        noise = 0.05  # 5% of standardised variance
        K = np.exp(-_sqdist(Xs, Xs) / (2 * self.ls2))
        K[np.diag_indices_from(K)] += noise
        self.alpha = np.linalg.solve(K, r)
        return self

    def predict(self, T: np.ndarray) -> np.ndarray:
        Ts = (T - self.x_mu) / self.x_sd
        pred_s = np.exp(-_sqdist(Ts, self.X) / (2 * self.ls2)) @ self.alpha
        return pred_s * self.y_sd + self.y_mu


class MechResidGP:
    """MR-GP: mechanistic OLS mean on a small physically-motivated subset
    plus a FastGP on the residuals over the full feature set."""

    def __init__(self, mech_idx: list[int]):
        self.mech_idx = mech_idx

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MechResidGP":
        Z = np.column_stack([np.ones(len(X)), X[:, self.mech_idx]])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        self.beta = beta
        resid = y - Z @ beta
        self.gp = FastGP().fit(X, resid)
        return self

    def predict(self, T: np.ndarray) -> np.ndarray:
        Zt = np.column_stack([np.ones(len(T)), T[:, self.mech_idx]])
        return Zt @ self.beta + self.gp.predict(T)


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    d2 = a2 + b2 - 2.0 * A @ B.T
    return np.maximum(d2, 0.0)


# ----------------------------------------------------------------------------
# dataset specs
# ----------------------------------------------------------------------------
@dataclass
class Spec:
    name: str
    path: Path
    entity: str
    target: str
    base_feats: list[str]
    leak_feats: list[str]
    mech_feats: list[str]
    rolling_years: list[int]
    loader: str = "parquet"


SPECS = {
    "A": Spec("A", D1 / "data" / "dataset_A_21city.parquet", "city", "cdw_total_kt",
              ["gdp_yi_cny", "population_wan", "bldg_under_construction_km2",
               "bldg_completed_km2", "year"],
              [],
              ["bldg_completed_km2"],
              list(range(2020, 2025))),
    "A17": Spec("A17", D1 / "data" / "dataset_A17_18city.parquet", "city", "cdw_total_kt",
                ["gdp_yi_cny", "population_wan", "bldg_under_construction_km2",
                 "bldg_completed_km2", "year"],
                [],
                ["bldg_completed_km2"],
                list(range(2020, 2025))),
    "B": Spec("B", D1 / "data" / "dataset_B_eurostat.csv", "country", "msw_gen_t",
              ["log_gdp", "log_pop", "year"],
              ["msw_trt_t"],
              ["log_gdp", "log_pop"],
              list(range(2019, 2024)), loader="csv"),
    "C0": Spec("C0", D1 / "data" / "dataset_C0_bjsz.csv", "city", "cdw_10kt",
               ["floor_started_10k_m2", "floor_completed_10k_m2", "tertiary_share_pct",
                "gdp_yi_cny", "population_wan", "year", "is_BJ"],
               ["soil_10kt", "demolition_10kt", "constr_waste_10kt"],
               ["floor_completed_10k_m2"],
               list(range(2020, 2025)), loader="csv"),
}


def load_frame(spec: Spec) -> pd.DataFrame:
    if spec.loader == "parquet":
        df = pd.read_parquet(spec.path)
    else:
        df = pd.read_csv(spec.path)
    df = df.sort_values([spec.entity, "year"]).reset_index(drop=True)

    # --- REV A: fill the complete entity-year grid so shift(1) = calendar t-1 ---
    years_full = np.arange(df["year"].min(), df["year"].max() + 1)
    grids = []
    for ent, g in df.groupby(spec.entity):
        gy = g.set_index("year").reindex(years_full)
        gy[spec.entity] = ent
        gy.index.name = "year"
        grids.append(gy.reset_index())
    df = pd.concat(grids, ignore_index=True)
    # ---------------------------------------------------------------------------

    # entity one-hot dummies (uniform for all models; C0 uses its is_BJ column)
    if spec.name != "C0":
        dummies = pd.get_dummies(df[spec.entity], prefix="ent", dtype=float)
        df = pd.concat([df, dummies], axis=1)
        spec = Spec(**{**spec.__dict__, "base_feats": spec.base_feats})
    return df


def build_matrices(spec: Spec, df: pd.DataFrame):
    """Return feature matrices for both timings plus eval/train index masks."""
    ents = df[spec.entity].values
    dummy_cols = ([c for c in df.columns if c.startswith("ent_")]
                  if spec.name != "C0" else (["is_BJ"] if "is_BJ" in df.columns else []))
    feats_now = spec.base_feats + dummy_cols            # honest feature block
    feats_lag = [c for c in spec.base_feats if c != "year"] + dummy_cols  # lagged block

    Xc = df[feats_now].astype(float).copy()
    Xl = df[feats_now].astype(float).copy()
    lag_src = df[[c for c in feats_lag]].astype(float).copy()
    shift_cols = [c for c in feats_lag if not c.startswith("ent_") and c != "is_BJ"]
    for c in shift_cols:
        Xl[c] = lag_src[c].groupby(ents).shift(1)
    Xl["year"] = df["year"].astype(float)               # trend not shifted

    Lc = df[spec.leak_feats].astype(float).copy() if spec.leak_feats else None
    Ll = None
    if Lc is not None:
        Ll = Lc.copy()
        for c in spec.leak_feats:
            Ll[c] = Lc[c].groupby(ents).shift(1)

    # causal LOCF imputation of remaining NaNs (uses past only)
    for X in (Xc, Xl):
        for c in X.columns:
            X[c] = X[c].groupby(ents).ffill()
    for L in (Lc, Ll):
        if L is not None:
            for c in L.columns:
                L[c] = L[c].groupby(ents).ffill()

    # --- REV A: evaluation requires an OBSERVED previous calendar year for persistence ---
    y = df[spec.target].astype(float).values
    y_prev = pd.Series(y, index=np.arange(len(df))).groupby(ents).shift(1)
    lag_ok = y_prev.notna().values   # previous calendar year is observed
    d = dict(Xc=Xc, Xl=Xl, Lc=Lc, Ll=Ll, ents=ents, years=df["year"].values,
             y=y, lag_ok=lag_ok, feat_cols=feats_now, dummy_cols=dummy_cols)
    return d


def make_splits(d, spec: Spec, validation: str, timing: str, seed: int):
    n = len(d["y"])
    eval_mask = d["lag_ok"]
    eval_idx = np.where(eval_mask)[0]
    train_elig = eval_mask if timing == "lagged" else np.ones(n, bool)
    if validation == "rolling_origin":
        for yy in spec.rolling_years:
            te = np.where((d["years"] == yy) & eval_mask & np.isfinite(d["y"]))[0]
            tr = np.where((d["years"] < yy) & train_elig & np.isfinite(d["y"]))[0]
            if len(te):
                yield tr, te
    elif validation == "loocv":
        for i in eval_idx:
            if not np.isfinite(d["y"][i]):
                continue
            tr = np.setdiff1d(np.where(train_elig & np.isfinite(d["y"]))[0], [i])
            yield tr, np.array([i])
    elif validation == "random_5fold":
        rng = np.random.RandomState(seed)
        perm = rng.permutation(eval_idx)
        perm = perm[np.isfinite(d["y"][perm])]
        for fold in np.array_split(perm, 5):
            tr = np.setdiff1d(np.where(train_elig & np.isfinite(d["y"]))[0], fold)
            yield tr, fold
    else:
        raise ValueError(validation)


def fit_one_model(model: str, seed: int, Xtr, ytr, Xte, d, tr_idx, te_idx):
    if model == "persistence":
        # y_hat(e,t) = y(e,t-1); guaranteed available on eval rows (grid filled)
        key = pd.Series(d["y"]).groupby(d["ents"]).shift(1).values
        return key[te_idx]
    if model == "entity_mean":
        df_tr = pd.DataFrame({"e": d["ents"][tr_idx], "y": ytr})
        means = df_tr.groupby("e")["y"].mean()
        glob = ytr.mean()
        return np.array([means.get(e, glob) for e in d["ents"][te_idx]])
    if model == "ols":
        m = LinearRegression().fit(Xtr, ytr)
        return m.predict(Xte)
    if model == "lasso":
        alpha = max(0.01 * float(np.std(ytr)), 1e-8)
        m = Lasso(alpha=alpha, max_iter=200000).fit(Xtr, ytr)
        return m.predict(Xte)
    if model == "rf":
        m = RandomForestRegressor(n_estimators=300, random_state=seed,
                                  n_jobs=1, min_samples_leaf=1).fit(Xtr, ytr)
        return m.predict(Xte)
    if model == "xgb":
        m = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=seed,
                         n_jobs=1, verbosity=0).fit(Xtr, ytr)
        return m.predict(Xte)
    if model == "gp":
        m = FastGP().fit(Xtr, ytr)
        return m.predict(Xte)
    if model == "mrgp":
        feat_cols = d["feat_cols_used"]
        mech_idx = [feat_cols.index(c) for c in d["mech_feats_used"]]
        m = MechResidGP(mech_idx).fit(Xtr, ytr)
        return m.predict(Xte)
    raise ValueError(model)


def run_cell(ds: str, spec: Spec, d, validation: str, timing: str, subcomp, preproc: str, seed: int):
    model_key = f"{validation}/{timing}/{subcomp}/{preproc}/s{seed}"
    X = d["Xc"] if timing == "contemporaneous" else d["Xl"]
    feat_cols = list(X.columns)
    L = None
    if subcomp == "on" and d["Lc"] is not None:
        L = d["Lc"] if timing == "contemporaneous" else d["Ll"]
        feat_cols += list(L.columns)
        X = pd.concat([X, L], axis=1)
    mech_avail = [c for c in spec.mech_feats if c in feat_cols]
    dd = dict(d)
    dd["feat_cols_used"] = feat_cols
    dd["mech_feats_used"] = mech_avail
    Xv = X.values.astype(float)

    splits = list(make_splits(d, spec, validation, timing, seed))
    rows = []
    log = []

    def one_split(args):
        k, (tr, te) = args
        ytr = d["y"][tr]
        if preproc == "global":
            # REV A: fit scaler/imputer on ALL rows (the leaky shortcut)
            med = np.nanmedian(Xv, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            Af = np.where(np.isfinite(Xv), Xv, med)
            mu, sd = Af.mean(0), Af.std(0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            Xtr, Xte = (Af[tr] - mu) / sd, (Af[te] - mu) / sd
        else:
            med = np.nanmedian(Xv[tr], axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            Af = np.where(np.isfinite(Xv), Xv, med)
            mu, sd = Af[tr].mean(0), Af[tr].std(0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            Xtr, Xte = (Af[tr] - mu) / sd, (Af[te] - mu) / sd
        out = {}
        for model in MODELS:
            try:
                out[model] = fit_one_model(model, seed, Xtr, ytr, Xte, dd, tr, te)
            except Exception as exc:  # context logged; coverage assert fails loudly
                log.append((model, f"FAILED {validation}/{timing}/{subcomp}/"
                                   f"{preproc}/seed{seed} {type(exc).__name__}: {exc}"))
                out[model] = None
        return te, out

    n_eval = sum(len(te) for _, te in splits)
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        for te, out in ex.map(one_split, list(enumerate(splits))):
            for model, pred in out.items():
                if pred is None:
                    log.append((model, "FAILED"))
                    continue
                for j, i in enumerate(te):
                    rows.append(dict(
                        dataset=ds, model=model, validation=validation, timing=timing,
                        subcomp=("na" if d["Lc"] is None else subcomp), preproc=preproc,
                        seed=seed, entity=d["ents"][i], year=int(d["years"][i]),
                        y_true=float(d["y"][i]), y_pred=float(pred[j])))
    # integrity: every model must cover every evaluation row exactly once
    got = {m: sum(1 for r in rows if r["model"] == m) for m in MODELS}
    for m, n in got.items():
        assert n == n_eval, f"{ds}/{model_key}: {m} produced {n} rows, expected {n_eval}"
    return rows, log


def metrics(preds: np.ndarray, y: np.ndarray) -> dict:
    resid = y - preds
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(
        r2_pooled=1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        rmse=float(np.sqrt(np.mean(resid ** 2))),
        mae=float(np.mean(np.abs(resid))),
        n_test=int(len(y)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="C0,A,B")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    datasets = [s for s in args.datasets.split(",") if s]

    seeds = [0] if args.quick else SEEDS
    validations = ["rolling_origin", "loocv"] if args.quick else VALIDATIONS

    cells_path = OUT / "benchmark_cells.csv"
    pred_path = OUT / "benchmark_predictions.parquet"
    # REV A: append mode — read existing predictions, drop rows for datasets
    # being re-run, then append new results at the end.
    if pred_path.exists():
        existing = pd.read_parquet(pred_path)
        existing = existing[~existing.dataset.isin(datasets)]
        all_preds = existing.to_dict("records")
    else:
        all_preds = []
    cell_rows = []
    t0 = time.time()
    log_lines = []
    for ds in datasets:
        spec = SPECS[ds]
        df = load_frame(spec)
        d = build_matrices(spec, df)
        subcomps = ["off", "on"] if d["Lc"] is not None else ["off"]
        n_cells = 0
        for validation in validations:
            for timing in TIMINGS:
                for subcomp in subcomps:
                    for preproc in PREPROCS:
                        for seed in seeds:
                            rows, log = run_cell(ds, spec, d, validation, timing,
                                                 subcomp, preproc, seed)
                            all_preds.extend(rows)
                            pr = pd.DataFrame(rows)
                            for model in MODELS:
                                pm = pr[pr.model == model]
                                if len(pm) == 0:
                                    continue
                                m = metrics(pm.y_pred.values, pm.y_true.values)
                                cell_rows.append(dict(
                                    dataset=ds, model=model, validation=validation,
                                    timing=timing, subcomp=pm.subcomp.iloc[0],
                                    preproc=preproc, seed=seed, **m))
                            n_cells += 1
                            msg = (f"[{ds}] {validation:14s} {timing:15s} subcomp={subcomp:4s} "
                                   f"preproc={preproc:6s} seed={seed}  ({time.time()-t0:7.1f}s)")
                            print(msg, flush=True)
                            log_lines.append(msg)
        print(f"[{ds}] done: {n_cells} protocol cells", flush=True)
        pd.DataFrame(cell_rows).to_csv(cells_path, index=False)   # flush per dataset

    pdf = pd.DataFrame(all_preds)
    dup = pdf.duplicated(subset=["dataset", "model", "validation", "timing",
                                 "subcomp", "preproc", "seed", "entity", "year"])
    assert not dup.any(), f"duplicate prediction keys: {int(dup.sum())} rows"
    pd.DataFrame(cell_rows).to_csv(cells_path, index=False)
    pdf.to_parquet(pred_path, index=False)
    with open(OUT / "engine_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + f"\ntotal {time.time()-t0:.1f}s\n")
    print(f"cells -> {cells_path} ({len(cell_rows)} rows)")
    print(f"predictions -> {pred_path} ({len(all_preds)} rows)")


if __name__ == "__main__":
    main()
