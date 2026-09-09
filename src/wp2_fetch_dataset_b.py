# -*- coding: utf-8 -*-
"""WP2 — Dataset B freeze: Eurostat municipal solid waste panel.

Downloads via the public Eurostat JSON-API:
  - env_wasmun  GEN (waste generated), unit THS_T (converted to tonnes) -> target
  - env_wasmun  TRT (total treatment), unit THS_T -> target-adjacent "leaky" feature
  - nama_10_gdp CLV10_MNAC (real GDP, M EUR)   -> macro predictor
  - tps00001    population on 1 Jan (persons)  -> macro predictor

Selection rule (frozen): 2-letter country codes only (no aggregates),
countries with >= 15 non-null GEN years; years 2004-2023.
Output: data/dataset_B_eurostat.csv (long), data/dataset_B_diagnostics.csv,
appends a Dataset B section to FREEZE.md.
"""
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd

D1 = Path(__file__).resolve().parents[1]
OUT_DIR = D1 / "data"
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def fetch_json(dataset: str, params: dict) -> pd.DataFrame:
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{API}/{dataset}?format=JSON&lang=EN&{qs}"
    with urllib.request.urlopen(url, timeout=120) as r:
        js = json.load(r)
    ids, size = js["id"], js["size"]
    dims = []
    for dim in ids:
        cats = js["dimension"][dim]["category"]
        idx = cats["index"]
        labels = cats["label"]
        # index maps code->position; invert
        pos = {v: k for k, v in idx.items()}
        dims.append([pos[i] for i in range(len(idx))])
    cols = [d for d in dims]
    mi = pd.MultiIndex.from_product(cols, names=ids)
    s = pd.Series(js["value"].values(), index=mi[list(js["value"].keys())]
                  if False else None) if False else None
    # js["value"] is a dict position->value; build in row-major order
    arr = np.full(len(mi), np.nan)
    vmap = {int(k): float(v) for k, v in js["value"].items()}
    for pos, v in vmap.items():
        arr[pos] = v
    df = s if s is not None else pd.DataFrame(arr, index=mi, columns=["value"])
    return df.reset_index()


def main() -> None:
    print("downloading env_wasmun GEN ...")
    gen = fetch_json("env_wasmun", {"wst_oper": "GEN", "unit": "THS_T"})
    gen["value"] = gen["value"] * 1e3  # thousand tonnes -> tonnes
    print("downloading env_wasmun TRT ...")
    trt = fetch_json("env_wasmun", {"wst_oper": "TRT", "unit": "THS_T"})
    trt["value"] = trt["value"] * 1e3
    print("downloading nama_10_gdp ...")
    gdp = fetch_json("nama_10_gdp", {"unit": "CLV10_MNAC", "na_item": "B1GQ"})
    print("downloading tps00001 ...")
    pop = fetch_json("tps00001", {})

    gen = gen.rename(columns={"value": "msw_gen_t"})[["geo", "time", "msw_gen_t"]]
    trt = trt.rename(columns={"value": "msw_trt_t"})[["geo", "time", "msw_trt_t"]]
    gdp = gdp.rename(columns={"value": "gdp_meur"})[["geo", "time", "gdp_meur"]]
    pop = pop.rename(columns={"value": "population"})[["geo", "time", "population"]]

    df = (gen.merge(trt, on=["geo", "time"], how="outer")
             .merge(gdp, on=["geo", "time"], how="left")
             .merge(pop, on=["geo", "time"], how="left"))
    df["time"] = df["time"].astype(int)
    df = df[(df["time"] >= 2004) & (df["time"] <= 2023)]

    # keep 2-letter country codes only (drops EU27_2020, EA20, DEA etc.)
    df = df[df["geo"].str.fullmatch(r"[A-Z]{2}")].copy()

    # frozen selection: >=15 non-null GEN years
    n_years = (df.dropna(subset=["msw_gen_t"])
                 .groupby("geo")["time"].agg(["count", "min", "max"])
                 .rename(columns={"count": "n_gen_years"}))
    keep = n_years[n_years["n_gen_years"] >= 15].index.tolist()
    df = df[df["geo"].isin(keep)].sort_values(["geo", "time"]).reset_index(drop=True)

    df = df.rename(columns={"geo": "country", "time": "year"})
    # data-quality flag: Eurostat reported series; mark gaps
    df["msw_data_flag"] = np.where(df["msw_gen_t"].notna(), "reported", "missing")
    df["log_gdp"] = np.log(df["gdp_meur"].where(df["gdp_meur"] > 0))
    df["log_pop"] = np.log(df["population"].where(df["population"] > 0))
    df["log_msw"] = np.log(df["msw_gen_t"].where(df["msw_gen_t"] > 0))

    out = OUT_DIR / "dataset_B_eurostat.csv"
    df.to_csv(out, index=False)
    diag = df.groupby("country").agg(
        n_gen=("msw_gen_t", "count"), n_trt=("msw_trt_t", "count"),
        year_min=("year", "min"), year_max=("year", "max")).reset_index()
    diag.to_csv(OUT_DIR / "dataset_B_diagnostics.csv", index=False)
    sha = hashlib.sha256(pd.read_csv(out).to_csv(index=False).encode()).hexdigest()[:16]

    with open(D1 / "FREEZE.md", "a", encoding="utf-8") as f:
        f.write(f"""

# Dataset B freeze (WP2)

- frozen file: `data/dataset_B_eurostat.csv` ({len(df)} rows, {df['country'].nunique()} countries)
- sha256(csv-serialization) prefix: `{sha}`
- source: Eurostat env_wasmun (GEN/TRT, unit T), nama_10_gdp (CLV10_MNAC), tps00001
- selection rule (frozen): 2-letter geo codes, >=15 non-null GEN years, 2004-2023
- target: msw_gen_t; leaky feature: msw_trt_t (same-year total treatment, mass-balance
  adjacent to the target -- the analog of CDW subcomponent leakage in Dataset A)
- predictors: log_gdp, log_pop, year (+ country effects inside models)
- flags: all non-missing observations 'reported' (Eurostat harmonised reporting);
  missing GEN years flagged 'missing' and excluded from fitting/evaluation
""")
    print(diag.to_string(index=False))
    print(f"\nfrozen -> {out}  sha256[:16]={sha}")


if __name__ == "__main__":
    main()
