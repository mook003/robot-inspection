#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import math
import random
from typing import Iterable

import numpy as np


def rot_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = [math.radians(d) for d in (rx_deg, ry_deg, rz_deg)]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], float)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], float)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], float)
    return Rz @ Ry @ Rx


def write_points_csv(path: Path, names: Iterable[str], pts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "x", "y", "z"])
        for name, p in zip(names, pts):
            w.writerow([name, f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}"])


def write_program_csv(path: Path, labels: list[str], prog: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["position", "x", "y", "z"])
        for label, p in zip(labels, prog):
            w.writerow([label, f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}"])


def write_measurements_csv(path: Path, labels: list[str], centers: np.ndarray, cycles: int, noise: float, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "position", "x", "y", "z"])
        for label, c in zip(labels, centers):
            for cycle in range(1, cycles + 1):
                dx = rng.gauss(0.0, noise)
                dy = rng.gauss(0.0, noise)
                dz = rng.gauss(0.0, noise)
                w.writerow([cycle, label, f"{c[0] + dx:.3f}", f"{c[1] + dy:.3f}", f"{c[2] + dz:.3f}"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate demo data for robot-accuracy")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--noise", type=float, default=0.10, help="mm, Gaussian sigma for measurements")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=100.0, help="mm spacing for reference points")
    ap.add_argument("--rot", type=float, nargs=3, default=[1.0, 1.5, 2.0], metavar=("RX","RY","RZ"), help="deg rotation from tracker->robot")
    ap.add_argument("--trans", type=float, nargs=3, default=[5.0, -3.0, 2.0], metavar=("TX","TY","TZ"), help="mm translation from tracker->robot")
    ap.add_argument("--refs", type=int, choices=[4,5,6], default=6, help="number of reference points to emit")
    args = ap.parse_args()

    out_dir: Path = args.out

    # Reference points in tracker frame
    s = args.scale
    all_names = ["A", "B", "C", "D", "E", "F"]
    all_P_T = np.array([
        [0.0, 0.0, 0.0],   # A
        [s, 0.0, 0.0],     # B
        [0.0, s, 0.0],     # C
        [0.0, 0.0, s],     # D
        [s, s, 0.0],       # E
        [s, 0.0, s],       # F
    ], float)
    # Select first N references; for 4 this is a non-coplanar tetra (A,B,C,D)
    N = args.refs
    names = all_names[:N]
    P_T = all_P_T[:N]


    # Apply known rigid transform to get robot-frame refs: P_R = R * P_T + t
    R = rot_xyz(*args.rot)
    t = np.array(args.trans, float)
    P_R = (R @ P_T.T).T + t.reshape(1, 3)

    write_points_csv(out_dir / "ref_tracker.csv", names, P_T)
    write_points_csv(out_dir / "ref_robot.csv", names, P_R)

    # Program positions in robot base frame
    prog_labels = ["P1", "P2", "P3", "P4", "P5"]
    P_prog = np.array([
        [10.0, 10.0, 10.0],
        [50.0, 10.0, 10.0],
        [10.0, 50.0, 10.0],
        [10.0, 10.0, 50.0],
        [50.0, 50.0, 50.0],
    ], float)
    write_program_csv(out_dir / "program.csv", prog_labels, P_prog)

    # Measurements already in robot base frame, around program positions
    write_measurements_csv(out_dir / "measurements.csv", prog_labels, P_prog, args.cycles, args.noise, args.seed)

    print(f"Wrote: {out_dir}/ref_tracker.csv, ref_robot.csv, program.csv, measurements.csv")
    print(f"Transform tracker->robot: R=rot_xyz({args.rot[0]:.2f},{args.rot[1]:.2f},{args.rot[2]:.2f}) deg, t={tuple(args.trans)} mm")


if __name__ == "__main__":
    main()
