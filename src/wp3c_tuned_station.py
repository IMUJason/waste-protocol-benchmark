# -*- coding: utf-8 -*-
"""WP3c — Tuned-model station: fold-internal hyperparameter tuning for
RF and XGBoost, addressing the "untuned strawman" objection.

Design (pre-specified):
  models : rf_tuned (grid: max_depth {None,6} x min_samples_leaf {1,5}),
           xgb_tuned (grid: max_depth {3,6} x learning_rate {0.05,0.15})
  selection: forward validation inside each training fold;
             best config refit on full training fold
  stations: honest = (rolling_origin, lagged, off, fold)
            common = (loocv, contemporaneous, on-if-exists, global)
  datasets: C0, A, A17, B; seeds 0-2; eval rows identical to main engine.

Outputs: appends model rows to benchmark_predictions.parquet (dataset-level
flush-safe rewrite) and writes output/analysis_tuned.csv with pooled metrics
and persistence comparison (persistence values reused from existing rows).
"""
import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
D1 = HERE.parent
OUT = D1 / "output"

spec_ = importlib.util.spec_from_file_location("eng", HERE / "wp3_benchmark_engine.py")
eng = importlib.util.module_from_spec(spec_)
sys.modules["eng"] = eng
spec_.loader.exec_module(eng)

RF_GRID = [{"max_depth": md, "min_samples_leaf": msl}
           for md in (None, 6) for msl in (1, 5)]
XGB_GRID = [{"max_depth": md, "learning_rate": lr}
            for md in (3, 6) for lr in (0.05, 0.15)]


def make(model_name, cfg, seed):
    if model_name == "rf_tuned":
        return RandomForestRegressor(n_estimators=300, random_state=seed,
                                     n_jobs=1, **cfg)
    return XGBRegressor(n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, n_jobs=1, verbosity=0, **cfg)


def fit_tuned(model_name, seed, Xtr, ytr, Xte, d, tr_idx):
    """Fold-internal tuning with FORWARD validation inside the training fold.
    For each hyperparameter config, the training fold is split by year:
    train on years < cutoff, validate on the last year. This mirrors the
    outer rolling-origin design and avoids shuffled CV inside a temporal task.
    If the training fold has < 2 distinct years, no tuning is performed."""
    grid = RF_GRID if model_name == "rf_tuned" else XGB_GRID
    years_tr = d["years"][tr_idx]
    uniq_years = np.unique(years_tr)
    if len(uniq_years) < 2:
        best = grid[0]
    else:
        cutoff = uniq_years[-1]  # last year = validation
        itr = np.where(years_tr < cutoff)[0]
        ite = np.where(years_tr == cutoff)[0]
        best, best_rmse = None, np.inf
        for cfg in grid:
            m = make(model_name, cfg, seed).fit(Xtr[itr], ytr[itr])
            pred = m.predict(Xtr[ite])
            r = float(np.sqrt(np.mean((ytr[ite] - pred) ** 2)))
            if r < best_rmse:
                best_rmse, best = r, cfg
    m = make(model_name, best, seed).fit(Xtr, ytr)
    return m.predict(Xte), best


def station_for(ds):
    """(validation, timing, subcomp, preproc) for honest and common."""
    has_leak = ds in ("C0", "B")
    return dict(
        honest=("rolling_origin", "lagged", "off", "fold"),
        common=("loocv", "contemporaneous", "on" if has_leak else "off", "global"),
    )


def run(ds, d, spec, station_name, seed):
    v, t, s, p = station_for(ds)[station_name]
    X = d["Xc"] if t == "contemporaneous" else d["Xl"]
    if s == "on" and d["Lc"] is not None:
        L = d["Lc"] if t == "contemporaneous" else d["Ll"]
        X = pd.concat([X, L], axis=1)
    Xv = X.values.astype(float)
    splits = list(eng.make_splits(d, spec, v, t, seed))
    rows = []

    def one(args):
        k, (tr, te) = args
        ytr = d["y"][tr]
        med = np.nanmedian(Xv[tr], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Af = np.where(np.isfinite(Xv), Xv, med)
        mu, sd = Af[tr].mean(0), Af[tr].std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Xtr, Xte = (Af[tr] - mu) / sd, (Af[te] - mu) / sd
        out = {}
        for mdl in ("rf_tuned", "xgb_tuned"):
            try:
                pred, cfg = fit_tuned(mdl, seed, Xtr, ytr, Xte, d, tr)
                out[mdl] = (pred, cfg)
            except Exception:
                out[mdl] = (None, None)
        return te, out

    with ThreadPoolExecutor(max_workers=10) as ex:
        for te, out in ex.map(one, list(enumerate(splits))):
            for mdl, (pred, cfg) in out.items():
                if pred is None:
                    continue
                for j, i in enumerate(te):
                    rows.append(dict(dataset=ds, model=mdl, validation=v, timing=t,
                                     subcomp=s, preproc=p, seed=seed,
                                     entity=d["ents"][i], year=int(d["years"][i]),
                                     y_true=float(d["y"][i]), y_pred=float(pred[j])))
    return rows


def main() -> None:
    cells_path = OUT / "benchmark_cells.csv"
    pred_path = OUT / "benchmark_predictions.parquet"
    preds = pd.read_parquet(pred_path)
    preds = preds[~preds.model.isin(["rf_tuned", "xgb_tuned"])]

    t0 = time.time()
    for ds in ["C0", "A", "A17", "B"]:
        if ds == "A17":
            if not (D1 / "data" / "dataset_A17_18city.parquet").exists():
                a = pd.read_parquet(D1 / "data" / "dataset_A_21city.parquet")
                drop = {"广州", "重庆", "西安"}  # Guangzhou, Chongqing, Xi'an
                a = a[~a["city"].isin(drop)].reset_index(drop=True)
                a.to_parquet(D1 / "data" / "dataset_A17_18city.parquet", index=False)
            eng.SPECS["A17"] = eng.Spec("A17", D1 / "data" / "dataset_A17_18city.parquet",
                                        "city", "cdw_total_kt",
                                        ["gdp_yi_cny", "population_wan",
                                         "bldg_under_construction_km2",
                                         "bldg_completed_km2", "year"],
                                        [], ["bldg_completed_km2"], list(range(2020, 2025)))
        spec = eng.SPECS[ds]
        df = eng.load_frame(spec)
        d = eng.build_matrices(spec, df)
        for seed in [0, 1, 2]:
            for st in ("honest", "common"):
                rows = run(ds, d, spec, st, seed)
                preds = pd.concat([preds, pd.DataFrame(rows)], ignore_index=True)
                print(f"[{ds}] {st} seed={seed}: {len(rows)} preds ({time.time()-t0:.0f}s)",
                      flush=True)

    preds.to_parquet(pred_path, index=False)

    # analysis table
    out_rows = []
    for ds in ["C0", "A", "A17", "B"]:
        st_map = station_for(ds)
        # persistence per station
        for st, (v, t, s, p) in st_map.items():
            pm = preds[(preds.dataset == ds) & (preds.model == "persistence")
                       & (preds.validation == v) & (preds.timing == t)
                       & (preds.preproc == p) & (preds.seed == 0)]
            pm = pm[pm.subcomp == "na"] if s != "on" else pm[pm.subcomp == "on"]
            if len(pm) == 0:
                pm = preds[(preds.dataset == ds) & (preds.model == "persistence")
                           & (preds.validation == v) & (preds.timing == t)
                           & (preds.seed == 0)]
                pm = pm[pm.subcomp != "on"]
            rmse_p = float(np.sqrt(((pm.y_true - pm.y_pred) ** 2).mean()))
            ybar = pm.y_true.mean()
            r2_p = 1 - ((pm.y_true - pm.y_pred) ** 2).sum() / ((pm.y_true - ybar) ** 2).sum()
            out_rows.append(dict(dataset=ds, model="persistence", station=st,
                                 r2=r2_p, rmse=rmse_p, n=len(pm),
                                 beats_persistence=np.nan, selected_cfg=""))
            for mdl in ("rf_tuned", "xgb_tuned"):
                m = preds[(preds.dataset == ds) & (preds.model == mdl)
                          & (preds.validation == v) & (preds.timing == t)
                          & (preds.subcomp == s) & (preds.preproc == p) & (preds.seed == 0)]
                if len(m) == 0:
                    continue
                rmse = float(np.sqrt(((m.y_true - m.y_pred) ** 2).mean()))
                r2 = 1 - ((m.y_true - m.y_pred) ** 2).sum() / ((m.y_true - m.y_true.mean()) ** 2).sum()
                out_rows.append(dict(dataset=ds, model=mdl, station=st, r2=r2,
                                     rmse=rmse, n=len(m), beats_persistence=int(rmse < rmse_p),
                                     selected_cfg=""))
    pd.DataFrame(out_rows).to_csv(OUT / "analysis_tuned.csv", index=False)
    print(pd.DataFrame(out_rows).round(3).to_string(index=False))
    print("analysis_tuned.csv written")


if __name__ == "__main__":
    main()
