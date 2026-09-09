# -*- coding: utf-8 -*-
"""WP4 — Analysis of the benchmark matrix.

Analysis design:
  1. COMMON EVALUATION SET: all protocol comparisons (inflation, ranking,
     persistence wins) are restricted to the entity-years of the rolling-origin
     station (the honest test window). LOOCV/random-CV rows outside that window
     are excluded from the comparison, so the difference is purely protocol,
     not test-set composition.
  2. PAIRED BOOTSTRAP: a single entity draw (with replacement) is applied to
     all models and both stations within one replicate; differences and ratios
     are computed inside the replicate before taking quantiles. Duplicate
     entities are kept (multiplicity preserved).
  3. CORRECT KENDALL TAU: the permutation test is removed; we report the
     bootstrap CI of tau and test the hypothesis tau = 1 (perfect agreement)
     by checking whether the CI excludes 1.
  4. PERSISTENCE COMPARISON: each model is compared against persistence
     evaluated on the SAME station and SAME evaluation rows, not against the
     honest-station persistence.

Outputs (under output/): analysis_ladder.csv, analysis_inflation.csv,
analysis_ranking.csv, analysis_persistence.csv, analysis_persistence_winrate.csv,
analysis_factor_decomp.csv, analysis_headline.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

D1 = Path(__file__).resolve().parents[1]
OUT = D1 / "output"
B_BOOT = 2000
RNG = np.random.default_rng(20260902)

FITTED = ["ols", "lasso", "rf", "xgb", "gp", "mrgp"]

# protocol stations: name -> (validation, timing, subcomp, preproc); subcomp "off" means "not-on"
STATIONS = {
    "H": ("rolling_origin", "lagged", "off", "fold"),
    "H_leak": ("rolling_origin", "lagged", "on", "fold"),
    "H_contemp": ("rolling_origin", "contemporaneous", "off", "fold"),
    "H_contemp_leak": ("rolling_origin", "contemporaneous", "on", "fold"),
    "LOOCV_exante": ("loocv", "lagged", "off", "fold"),
    "LOOCV": ("loocv", "contemporaneous", "off", "fold"),
    "LOOCV_global": ("loocv", "contemporaneous", "off", "global"),
    "LOOCV_leak_global": ("loocv", "contemporaneous", "on", "global"),
    "RAND5_leak_global": ("random_5fold", "contemporaneous", "on", "global"),
    "RAND5": ("random_5fold", "contemporaneous", "off", "global"),
    "RAND5_exante": ("random_5fold", "lagged", "off", "fold"),
    "H_global": ("rolling_origin", "lagged", "off", "global"),
}
HONEST = "H"


def build_index() -> dict:
    preds = pd.read_parquet(OUT / "benchmark_predictions.parquet")
    idx = {}
    for key, g in preds.groupby(["dataset", "model", "validation", "timing",
                                 "subcomp", "preproc", "seed"]):
        idx[tuple(key)] = g
    return idx


def frame_for(idx, ds, model, station, seed=0):
    v, t, s, p = STATIONS[station]
    if s == "on":
        sub = "on"
        key = (ds, model, v, t, sub, p, seed)
        return idx.get(key, pd.DataFrame())
    # "off" matches subcomp off OR na (dataset without leak features)
    for sub in ("off", "na"):
        key = (ds, model, v, t, sub, p, seed)
        if key in idx:
            return idx[key]
    return pd.DataFrame()


def ent_stats(m: pd.DataFrame) -> pd.DataFrame:
    if len(m) == 0:
        return pd.DataFrame(columns=["n", "ssres", "sum_y", "sum_y2"])
    r2 = (m.y_true - m.y_pred) ** 2
    g = m.assign(_r2=r2).groupby("entity").agg(
        n=("y_true", "size"), ssres=("_r2", "sum"),
        sum_y=("y_true", "sum"), sum_y2=("y_true", lambda z: (z ** 2).sum()))
    return g


def pooled_from_stats(st: pd.DataFrame) -> dict:
    n = st.n.sum()
    ssres = st.ssres.sum()
    ss_tot = st.sum_y2.sum() - st.sum_y.sum() ** 2 / n
    return dict(n=int(n), rmse=float(np.sqrt(ssres / n)),
                r2=float(1 - ssres / ss_tot) if ss_tot > 0 else np.nan)


def common_eval_years(idx, ds):
    """Entity-years of the rolling-origin honest station (the common test window)."""
    m = frame_for(idx, ds, "persistence", HONEST)
    if len(m) == 0:
        return set()
    return set(zip(m.entity, m.year))


def restrict_to_common(m: pd.DataFrame, common_set: set) -> pd.DataFrame:
    if len(m) == 0 or len(common_set) == 0:
        return m.iloc[0:0].copy()
    keys = list(zip(m.entity, m.year))
    mask = [k in common_set for k in keys]
    return m[mask].copy()


def boot_paired_r2(st_h: pd.DataFrame, st_c: pd.DataFrame, B=B_BOOT) -> np.ndarray:
    """Paired bootstrap: same entity draw applied to both stations.
    Returns distribution of (R2_common - R2_honest)."""
    ents = st_h.index.values
    st_h_n = st_h.n.values; st_h_res = st_h.ssres.values
    st_h_sy = st_h.sum_y.values; st_h_sy2 = st_h.sum_y2.values
    st_c_n = st_c.n.values; st_c_res = st_c.ssres.values
    st_c_sy = st_c.sum_y.values; st_c_sy2 = st_c.sum_y2.values
    out = np.empty(B)
    for b in range(B):
        pick = RNG.choice(np.arange(len(ents)), size=len(ents), replace=True)
        nh = st_h_n[pick].sum(); ssh = st_h_res[pick].sum()
        ssth = st_h_sy2[pick].sum() - st_h_sy[pick].sum() ** 2 / nh
        r2h = 1 - ssh / ssth if ssth > 0 else np.nan
        nc = st_c_n[pick].sum(); ssc = st_c_res[pick].sum()
        sstc = st_c_sy2[pick].sum() - st_c_sy[pick].sum() ** 2 / nc
        r2c = 1 - ssc / sstc if sstc > 0 else np.nan
        out[b] = r2c - r2h
    return out


def boot_rmse_ratio(st_m: pd.DataFrame, st_p: pd.DataFrame, B=B_BOOT) -> np.ndarray:
    """Paired bootstrap of RMSE(model)/RMSE(persistence) on the same entity draw."""
    ents = st_m.index.values
    m_res = st_m.ssres.values; m_n = st_m.n.values
    p_res = st_p.ssres.values; p_n = st_p.n.values
    out = np.empty(B)
    for b in range(B):
        pick = RNG.choice(np.arange(len(ents)), size=len(ents), replace=True)
        rm = np.sqrt(m_res[pick].sum() / m_n[pick].sum())
        rp = np.sqrt(p_res[pick].sum() / p_n[pick].sum())
        out[b] = rm / rp
    return out


def boot_tau(stats_dict: dict, B=B_BOOT) -> np.ndarray:
    """Paired bootstrap of Kendall tau between two rankings.
    stats_dict: {model: (stationA_ent_stats, stationB_ent_stats)} with aligned entities."""
    models = list(stats_dict.keys())
    ents = stats_dict[models[0]][0].index.values
    out = np.empty(B)
    for b in range(B):
        pick = RNG.choice(np.arange(len(ents)), size=len(ents), replace=True)
        ra, rb = {}, {}
        for mdl in models:
            A, Bx = stats_dict[mdl]
            ra[mdl] = np.sqrt(A.ssres.values[pick].sum() / A.n.values[pick].sum())
            rb[mdl] = np.sqrt(Bx.ssres.values[pick].sum() / Bx.n.values[pick].sum())
        cm = sorted(set(ra) & set(rb))
        if len(cm) >= 4:
            t, _ = stats.kendalltau([ra[c] for c in cm], [rb[c] for c in cm])
            out[b] = t
        else:
            out[b] = np.nan
    return out


def main() -> None:
    idx = build_index()
    cells = pd.read_csv(OUT / "benchmark_cells.csv")
    datasets = sorted({k[0] for k in idx})

    def has_leak(ds):
        return any(k[0] == ds and k[4] == "on" for k in idx)

    def common(ds):
        return "LOOCV_leak_global" if has_leak(ds) else "LOOCV_global"

    # ---------- ladder (raw, all rows) ----------
    ladder_rows = []
    for ds in datasets:
        for mdl in ["persistence"] + FITTED:
            for st_name in STATIONS:
                m = frame_for(idx, ds, mdl, st_name)
                if len(m) == 0:
                    continue
                p = pooled_from_stats(ent_stats(m))
                ladder_rows.append(dict(dataset=ds, model=mdl, station=st_name, **p))
    ladder = pd.DataFrame(ladder_rows)
    ladder.to_csv(OUT / "analysis_ladder.csv", index=False)

    # ---------- inflation honest vs common (COMMON EVAL SET) ----------
    infl_rows = []
    for ds in datasets:
        C = common(ds)
        common_set = common_eval_years(idx, ds)
        for mdl in FITTED + ["persistence"]:
            m_h = restrict_to_common(frame_for(idx, ds, mdl, HONEST), common_set)
            m_c = restrict_to_common(frame_for(idx, ds, mdl, C), common_set)
            if len(m_h) == 0 or len(m_c) == 0:
                continue
            sh, sc = ent_stats(m_h), ent_stats(m_c)
            ph, pc = pooled_from_stats(sh), pooled_from_stats(sc)
            d = boot_paired_r2(sh, sc)
            row = dict(dataset=ds, model=mdl, r2_honest=ph["r2"], r2_common=pc["r2"],
                       delta=pc["r2"] - ph["r2"],
                       delta_lo=float(np.nanpercentile(d, 2.5)),
                       delta_hi=float(np.nanpercentile(d, 97.5)))
            infl_rows.append(row)

    # skill vs persistence (honest station, common eval set) with paired bootstrap
    for ds in datasets:
        common_set = common_eval_years(idx, ds)
        m_p = restrict_to_common(frame_for(idx, ds, "persistence", HONEST), common_set)
        if len(m_p) == 0:
            continue
        sp = ent_stats(m_p)
        for mdl in FITTED:
            m_m = restrict_to_common(frame_for(idx, ds, mdl, HONEST), common_set)
            if len(m_m) == 0:
                continue
            sm = ent_stats(m_m)
            sk = 1 - boot_rmse_ratio(sm, sp)
            for row in infl_rows:
                if row["dataset"] == ds and row["model"] == mdl:
                    row["skill_vs_persistence_honest"] = float(np.nanmedian(sk))
                    row["skill_lo"] = float(np.nanpercentile(sk, 2.5))
                    row["skill_hi"] = float(np.nanpercentile(sk, 97.5))
    infl = pd.DataFrame(infl_rows)
    infl.to_csv(OUT / "analysis_inflation.csv", index=False)

    # ---------- ranking flips (COMMON EVAL SET) ----------
    def rank_pair(ds, sa, sb):
        common_set = common_eval_years(idx, ds)
        stats_dict = {}
        for mdl in FITTED:
            A = restrict_to_common(frame_for(idx, ds, mdl, sa), common_set)
            Bx = restrict_to_common(frame_for(idx, ds, mdl, sb), common_set)
            if len(A) == 0 or len(Bx) == 0:
                continue
            stats_dict[mdl] = (ent_stats(A), ent_stats(Bx))
        if len(stats_dict) < 4:
            return None
        # point estimate
        ra = {m: pooled_from_stats(v[0])["rmse"] for m, v in stats_dict.items()}
        rb = {m: pooled_from_stats(v[1])["rmse"] for m, v in stats_dict.items()}
        tau, _ = stats.kendalltau([ra[c] for c in sorted(ra)], [rb[c] for c in sorted(rb)])
        taus = boot_tau(stats_dict)
        lo, hi = np.nanpercentile(taus, 2.5), np.nanpercentile(taus, 97.5)
        return dict(tau=float(tau), n_models=len(stats_dict),
                    tau_lo=float(lo), tau_hi=float(hi),
                    ci_excludes_1=bool(hi < 1))

    rank_rows = []
    for ds in datasets:
        for sa, sb, nm in [(HONEST, common(ds), "honest_vs_common"),
                           (HONEST, "LOOCV", "honest_vs_LOOCV_noleak"),
                           (HONEST, "RAND5", "honest_vs_random5"),
                           (HONEST, "LOOCV_exante", "honest_vs_LOOCV_exante")]:
            r = rank_pair(ds, sa, sb)
            if r:
                rank_rows.append(dict(dataset=ds, comparison=nm, **r))
    ranking = pd.DataFrame(rank_rows)
    ranking.to_csv(OUT / "analysis_ranking.csv", index=False)

    # ---------- persistence accounting (SAME STATION, SAME EVAL SET) ----------
    pers_rows = []
    for ds in datasets:
        C = common(ds)
        common_set = common_eval_years(idx, ds)
        m_p_h = restrict_to_common(frame_for(idx, ds, "persistence", HONEST), common_set)
        m_p_c = restrict_to_common(frame_for(idx, ds, "persistence", C), common_set)
        if len(m_p_h) == 0 or len(m_p_c) == 0:
            continue
        rmse_p_h = pooled_from_stats(ent_stats(m_p_h))["rmse"]
        rmse_p_c = pooled_from_stats(ent_stats(m_p_c))["rmse"]
        row = dict(dataset=ds, rmse_persistence_honest=rmse_p_h,
                   rmse_persistence_common=rmse_p_c)
        for mdl in FITTED:
            for st_name, tag in [(HONEST, "honest"), (C, "common")]:
                m = restrict_to_common(frame_for(idx, ds, mdl, st_name), common_set)
                if len(m):
                    row[f"{mdl}_beats_persistence_{tag}"] = int(
                        pooled_from_stats(ent_stats(m))["rmse"] < (rmse_p_h if tag == "honest" else rmse_p_c))
        pers_rows.append(row)
    pd.DataFrame(pers_rows).to_csv(OUT / "analysis_persistence.csv", index=False)

    # per entity-year win rate (honest station, common eval set)
    win_rows = []
    for ds in datasets:
        common_set = common_eval_years(idx, ds)
        m = restrict_to_common(frame_for(idx, ds, "persistence", HONEST), common_set)
        if len(m) == 0:
            continue
        pmap = {(r.entity, r.year): abs(r.y_true - r.y_pred) for r in m.itertuples()}
        frames = {mdl: restrict_to_common(frame_for(idx, ds, mdl, HONEST), common_set) for mdl in FITTED}
        for (ent, yr), perr in pmap.items():
            wins = tot = 0
            for mdl, fr in frames.items():
                if len(fr) == 0:
                    continue
                hit = fr[(fr.entity == ent) & (fr.year == yr)]
                if len(hit):
                    tot += 1
                    wins += int(abs(hit.y_true.iloc[0] - hit.y_pred.iloc[0]) < perr)
            if tot:
                win_rows.append(dict(dataset=ds, entity=ent, year=yr,
                                     win_rate=wins / tot, n_models=tot))
    pd.DataFrame(win_rows).to_csv(OUT / "analysis_persistence_winrate.csv", index=False)

    # ---------- factor decomposition (COMMON EVAL SET) ----------
    FACTORS = {
        "validation: rolling->LOOCV": ("validation", "loocv"),
        "validation: rolling->random5": ("validation", "random_5fold"),
        "timing: lagged->contemporaneous": ("timing", "contemporaneous"),
        "preproc: fold->global": ("preproc", "global"),
        "subcomp: off->on": ("subcomp", "on"),
    }
    dec_rows = []
    for ds in datasets:
        common_set = common_eval_years(idx, ds)
        for fname, (fkey, fval) in FACTORS.items():
            if fkey == "subcomp" and not has_leak(ds):
                continue
            deltas = []
            for mdl in FITTED:
                variant = dict(zip(["validation", "timing", "subcomp", "preproc"], STATIONS[HONEST]))
                variant[fkey] = fval
                st_name = None
                for nm, cfg in STATIONS.items():
                    cv, ct, cs, cp = cfg
                    if (cv == variant["validation"] and ct == variant["timing"]
                            and cp == variant["preproc"]
                            and (cs == "on") == (variant["subcomp"] == "on")):
                        st_name = nm
                        break
                if st_name is None:
                    continue
                A = restrict_to_common(frame_for(idx, ds, mdl, st_name), common_set)
                B0 = restrict_to_common(frame_for(idx, ds, mdl, HONEST), common_set)
                if len(A) == 0 or len(B0) == 0:
                    continue
                deltas.append(pooled_from_stats(ent_stats(A))["r2"] - pooled_from_stats(ent_stats(B0))["r2"])
            if deltas:
                dec_rows.append(dict(dataset=ds, factor=fname,
                                     mean_delta_r2=float(np.mean(deltas)),
                                     median_delta_r2=float(np.median(deltas)),
                                     min=float(np.min(deltas)), max=float(np.max(deltas)),
                                     n_models=len(deltas)))
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(OUT / "analysis_factor_decomp.csv", index=False)

    # ---------- headline ----------
    headline = {}
    for ds in datasets:
        sub = infl[(infl.dataset == ds) & (infl.model != "persistence")]
        if len(sub):
            headline[f"{ds}_mean_inflation_pp"] = round(float(sub.delta.mean() * 100), 1)
            headline[f"{ds}_median_inflation_pp"] = round(float(sub.delta.median() * 100), 1)
            headline[f"{ds}_mean_inflation_ci_pp"] = [
                round(float(sub.delta_lo.mean() * 100), 1),
                round(float(sub.delta_hi.mean() * 100), 1)]
            headline[f"{ds}_max_inflation_pp"] = round(float(sub.delta.max() * 100), 1)
            best = sub.loc[sub.delta.idxmax()]
            headline[f"{ds}_max_inflation_model"] = best.model
        rk = ranking[(ranking.dataset == ds) & (ranking.comparison == "honest_vs_common")]
        if len(rk):
            headline[f"{ds}_tau_honest_vs_common"] = round(float(rk.tau.iloc[0]), 3)
            headline[f"{ds}_tau_ci"] = [round(float(rk.tau_lo.iloc[0]), 3),
                                        round(float(rk.tau_hi.iloc[0]), 3)]
            headline[f"{ds}_tau_ci_excludes_1"] = bool(rk.ci_excludes_1.iloc[0])
        pe_df = pd.DataFrame(pers_rows)
        if len(pe_df):
            row = pe_df[pe_df.dataset == ds].iloc[0]
            hs = [c for c in pe_df.columns if c.endswith("_beats_persistence_honest")]
            cs = [c for c in pe_df.columns if c.endswith("_beats_persistence_common")]
            def _sum(col_list):
                vals = [int(row[c]) for c in col_list if pd.notna(row[c])]
                return f"{sum(vals)}/{len(vals)}"
            headline[f"{ds}_models_beating_persistence_honest"] = _sum(hs)
            headline[f"{ds}_models_beating_persistence_common"] = _sum(cs)
    (OUT / "analysis_headline.json").write_text(
        json.dumps(headline, indent=2, ensure_ascii=False), encoding="utf-8")

    print("== inflation =="); print(infl.round(3).to_string(index=False))
    print("\n== ranking =="); print(ranking.round(3).to_string(index=False))
    print("\n== factor decomposition =="); print(dec.round(3).to_string(index=False))
    print("\n== headline =="); print(json.dumps(headline, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
