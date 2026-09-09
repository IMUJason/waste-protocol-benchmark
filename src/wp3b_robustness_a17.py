# -*- coding: utf-8 -*-
"""WP3b — Build dataset A17 (robustness subset): dataset A without the
flagged cities (drop suspected_estimated: 广州 Guangzhou;
contradicted: 重庆 Chongqing, 西安 Xi'an -> 18 cities). City keys match
the dataset's Chinese names. Data step only; the benchmark engine runs A17 as part of
its main pass (A17 is registered in wp3_benchmark_engine.SPECS).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
D1 = HERE.parent


def main() -> None:
    a = pd.read_parquet(D1 / "data" / "dataset_A_21city.parquet")
    drop = {"广州", "重庆", "西安"}
    a17 = a[~a["city"].isin(drop)].copy().reset_index(drop=True)
    a17.to_parquet(D1 / "data" / "dataset_A17_18city.parquet", index=False)
    print(f"A17 written: {a17.shape}, cities={a17.city.nunique()}")


if __name__ == "__main__":
    main()
