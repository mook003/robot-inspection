from __future__ import annotations

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from .io import (
    load_points_robot,
    load_points_tracker,
    load_program_positions,
    load_measurements_csv,
    validate_correspondence,
    DataFormatError,
)
from .pipeline import transform_and_compute
from .metrics import compute_ap, compute_rp
from .report import build_tables, save_report
from .report import build_tables


def main(argv: list[str] | None = None) -> None:
    #Set patameters
    p = argparse.ArgumentParser(description="AP/RP расчёт с преобразованием трекер→робот")
    p.add_argument("--ref-robot", type=Path, required=True)
    p.add_argument("--ref-tracker", type=Path, required=True)
    p.add_argument("--prog", type=Path, required=True)
    p.add_argument("--meas", type=Path, required=True, help="CSV измерений в системе трекера или робота")
    p.add_argument("--out", type=Path, required=False)
    p.add_argument("--cycles", type=int, default=30)
    p.add_argument("--max-resid", type=float, default=0.1)
    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args(argv)

    #Read all data
    try:
        P_R, names_r = load_points_robot(args.ref_robot)
        P_T, names_t = load_points_tracker(args.ref_tracker)
        validate_correspondence(names_r, names_t, P_R.shape[0], P_T.shape[0])
        prog = load_program_positions(args.prog)
        meas = load_measurements_csv(args.meas, required_positions=prog.keys(), expected_cycles=args.cycles)
    except (FileNotFoundError, DataFormatError) as e:
        print(f"Input error: {e}", file=sys.stderr)
        sys.exit(2)

    #Status of data
    print(f"refs: robot={P_R.shape[0]} tracker={P_T.shape[0]}")
    print(f"program positions: {sorted(prog.keys())}")
    print("measurements:  " + ", ".join(f"{k}:{v.shape[0]}" for k, v in meas.items()))

    #Data processing
    rows = []
    T, residuals, max_r, rms_r, meas_robot, rows = transform_and_compute(
        P_R, P_T, prog, meas, max_resid=args.max_resid
    )
    R, t = T.R, T.t
    print(f"T_RT: max residual = {max_r:.5f} mm, rms = {rms_r:.5f} mm")




    df_rt, df_metrics = build_tables(R, t, residuals, rows)

    if args.dry_run and not args.out:
        with pd.option_context('display.max_columns', None, 'display.width', 160):
            print(df_rt)
            print(df_metrics)
        return

    if args.out:
        save_report(args.out, df_rt, df_metrics)
        print(f"Saved: {args.out}")
    else:
        with pd.option_context('display.max_columns', None, 'display.width', 160):
            print(df_rt)
            print(df_metrics)


if __name__ == "__main__":
    main()