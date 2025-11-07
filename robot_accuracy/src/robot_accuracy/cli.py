from __future__ import annotations


import sys
from pathlib import Path
import argparse
import pandas as pd

from .io import (
load_points_robot,
load_points_tracker,
load_program_positions,
load_measurements_csv,
validate_correspondence,
DataFormatError,
)
# Функции ниже будут реализованы в последующих шагах
from .transform import estimate_rt_svd # type: ignore
from .metrics import compute_ap, compute_rp # type: ignore
from .report import build_tables, save_report # type: ignore

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="AP/RP расчёт точности робота")
    p.add_argument("--ref-robot", type=Path, required=True)
    p.add_argument("--ref-tracker", type=Path, required=True)
    p.add_argument("--prog", type=Path, required=True)
    p.add_argument("--meas", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path('results.csv'), help="Путь к results.csv|xlsx")
    p.add_argument("--cycles", type=int, default=30)
    p.add_argument("--max-resid", type=float, default=0.1, help="мм, допуск остатка при оценке T_RT")
    p.add_argument("--dry-run", action="store_true", help="Только проверки и сводка без расчётов")

    args = p.parse_args(argv)

    try:
        rr, names_r = load_points_robot(args.ref_robot)
        rt, names_t = load_points_tracker(args.ref_tracker)
        validate_correspondence(names_r, names_t, rr.shape[0], rt.shape[0])
        prog = load_program_positions(args.prog)
        meas = load_measurements_csv(args.meas, required_positions=prog.keys(), expected_cycles=args.cycles)
    except DataFormatError as e:
        print(f"Input error: Wrong format '{e}'", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as e:
        print(f"Input error: No such file '{e}'", file=sys.stderr)
        sys.exit(2)

    print(f"refs: robot={rr.shape[0]} tracker={rt.shape[0]}")
    print(f"program positions: {sorted(prog.keys())}")
    print("measurements: " + ", ".join(f"{k}:{v.shape[0]}" for k, v in meas.items()))


    if args.dry_run:
        return


    # Заглушка до реализации ядра расчётов
    try:
        # R, t, resid, max_resid = estimate_rt_svd(rr, rt)
        # if max_resid > args.max_resid:
        # raise RuntimeError(f"Max residual {max_resid:.3f} mm exceeds {args.max_resid} mm")
        # Здесь же должны вычисляться AP/RP и формироваться таблицы
        df_rt = pd.DataFrame({
        "note": ["Заглушка: реализация расчётов будет добавлена"],
        })
        df_metrics = pd.DataFrame({
        "position": sorted(meas.keys()),
        "AP": [None]*len(meas),
        "RP": [None]*len(meas),
        })
        build_tables # noqa: F401 (сигнатура зарезервирована)
        save_report(args.out, df_rt, df_metrics)
        print(f"Saved: {args.out}")
    except Exception as e:
        print(f"Processing error: {e}", file=sys.stderr)
        sys.exit(1)