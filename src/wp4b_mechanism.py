# -*- coding: utf-8 -*-
"""WP4b — Mechanism analysis: per-entity protocol gain vs series structure.

  - Only panels A (21 Chinese cities) and B (33 European countries) are
    used; the case study and the screened subset are nested/overlapping, so
    excluding them leaves every entity statistically independent.
  - Gain is computed on the common (rolling-origin) evaluation window,
    consistent with wp4_analysis.py.
  - gain = log(common RMSE / honest RMSE), averaged over the six covariate
    models; positive = common protocol makes the model look better.
  - Trend strength = R^2 of a linear fit of the entity's observed target
    series on year.

Outputs: output/analysis_mechanism.csv
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D1 = Path(__file__).resolve().parents[1]
OUT = D1 / "output"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "pdf.fonttype": 42, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})

FITTED = ["ols", "lasso", "rf", "xgb", "gp", "mrgp"]
HONEST = dict(validation="rolling_origin", timing="lagged", preproc="fold")


def station_frame(preds, ds, model, honest: bool):
    if honest:
        m = preds[(preds.dataset == ds) & (preds.model == model)
                  & (preds.validation == "rolling_origin")
                  & (preds.timing == "lagged")
                  & (preds.preproc == "fold") & (preds.seed == 0)]
        m = m[m.subcomp != "on"]
    else:
        # common station: LOOCV, contemporaneous, leak-on where available, global
        m = preds[(preds.dataset == ds) & (preds.model == model)
                  & (preds.validation == "loocv")
                  & (preds.timing == "contemporaneous")
                  & (preds.preproc == "global") & (preds.seed == 0)]
        has_leak = (preds[(preds.dataset == ds)].subcomp == "on").any()
        m = m[m.subcomp == "on"] if has_leak else m[m.subcomp != "on"]
    return m


def main():
    preds = pd.read_parquet(OUT / "benchmark_predictions.parquet")
    # REV B: series structure from SOURCE panels (not the prediction frame,
    # which only covers the 5-year evaluation window)
    src = {
        "A": pd.read_parquet(D1 / "data" / "dataset_A_21city.parquet")
             .rename(columns={"city": "entity", "cdw_total_kt": "y"})[["entity", "year", "y"]],
        "B": pd.read_csv(D1 / "data" / "dataset_B_eurostat.csv")
             .rename(columns={"country": "entity", "msw_gen_t": "y"})[["entity", "year", "y"]],
    }
    rows = []
    for ds in ("A", "B"):  # REV A: independent entities only
        ph = station_frame(preds, ds, "persistence", True)
        if len(ph) == 0:
            continue
        common_keys = set(zip(ph.entity, ph.year))
        panel = src[ds].dropna(subset=["y"])
        for ent, g in panel.groupby("entity"):
            y = g.sort_values("year")[["year", "y"]]
            n_years = len(y)
            if n_years < 4:
                continue
            # trend strength: R^2 of y ~ year
            x = y.year.values.astype(float)
            yy = y["y"].values
            b, a = np.polyfit(x, yy, 1)
            ss_res = ((yy - (a + b * x)) ** 2).sum()
            ss_tot = ((yy - yy.mean()) ** 2).sum()
            trend_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            # protocol gain on common eval set, averaged over covariate models
            gains = []
            for mdl in FITTED:
                mh = station_frame(preds, ds, mdl, True)
                mc = station_frame(preds, ds, mdl, False)
                mh = mh[(mh.entity == ent) & mh.apply(lambda r: (r.entity, r.year) in common_keys, axis=1)]
                mc = mc[(mc.entity == ent) & mc.apply(lambda r: (r.entity, r.year) in common_keys, axis=1)]
                if len(mh) < 2 or len(mc) < 2:
                    continue
                rmse_h = float(np.sqrt(((mh.y_true - mh.y_pred) ** 2).mean()))
                rmse_c = float(np.sqrt(((mc.y_true - mc.y_pred) ** 2).mean()))
                if rmse_h > 0 and rmse_c > 0:
                    gains.append(np.log(rmse_c / rmse_h))
            if gains:
                rows.append(dict(dataset=ds, entity=ent, n_years=n_years,
                                 trend_r2=trend_r2, log_gain=float(np.mean(gains)),
                                 n_models=len(gains)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "analysis_mechanism.csv", index=False)
    print(f"mechanism: {len(d)} entities ({d.dataset.value_counts().to_dict()}), "
          f"mean models/entity {d.n_models.mean():.1f}")

    print("analysis_mechanism.csv written")


if __name__ == "__main__":
    main()
