# -*- coding: utf-8 -*-
"""One-command reproduction for the waste-forecasting protocol benchmark.

Runs, in order: Eurostat fetch (network; skippable), A17 subset build,
benchmark engine (C0, A, A17, B), tuned-tree station, analysis, mechanism
analysis, and number generation.

Usage:  python src/run_all.py [--skip-download]
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("fetch dataset B (Eurostat)", ["wp2_fetch_dataset_b.py"]),
    ("build dataset A17 (robustness subset)", ["wp3b_robustness_a17.py"]),
    ("benchmark engine (C0, A, A17, B)", ["wp3_benchmark_engine.py", "--datasets", "C0,A,A17,B"]),
    ("tuned-tree station", ["wp3c_tuned_station.py"]),
    ("analysis", ["wp4_analysis.py"]),
    ("mechanism analysis", ["wp4b_mechanism.py"]),
    ("number generation", ["wp5_make_numbers.py"]),
]


def main() -> None:
    skip_download = "--skip-download" in sys.argv
    for name, args in STEPS:
        if skip_download and "fetch" in args[0]:
            print(f"== SKIP {name} (network; uses frozen data/dataset_B_eurostat.csv) ==")
            continue
        print(f"\n===== {name} =====", flush=True)
        r = subprocess.run([sys.executable, str(HERE / args[0]), *args[1:]])
        if r.returncode != 0:
            sys.exit(f"FAILED at {name}")
    print("\nAll steps completed. Results in output/ (analysis CSVs, numbers.tex).")


if __name__ == "__main__":
    main()
