from __future__ import annotations

"""
I/O utilities for robot accuracy project.

Supported inputs
- Reference points (robot base frame): CSV or JSON with columns: name(optional), x, y, z
- Reference points (tracker frame): CSV or JSON with columns: name(optional), x, y, z
- Program positions (P1..P5): CSV or JSON with columns: position, x, y, z
- Measurements: CSV with columns: cycle, position, x, y, z

Notes
- Column names are case-insensitive; leading/trailing spaces are ignored.
- Units: millimeters. No unit conversion is performed here.
- Validation raises DataFormatError on structural issues.

Example CSV headers
- points: name,x,y,z
- program positions: position,x,y,z
- measurements: cycle,position,x,y,z
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import math
import pandas as pd
import numpy as np

__all__ = [
    "DataFormatError",
    "load_points_robot",
    "load_points_tracker",
    "load_program_positions",
    "load_measurements_csv",
    "validate_correspondence",
]


class DataFormatError(ValueError):
    pass


# ---------- helpers ----------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
        return _normalize_columns(df)

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Accept either {"points": [...]} or a raw list
        if isinstance(data, dict) and "points" in data:
            data = data["points"]
        if not isinstance(data, list):
            raise DataFormatError(f"JSON must contain a list of objects at root or under 'points' in {path}")
        df = pd.DataFrame(data)
        return _normalize_columns(df)

    raise DataFormatError(f"Unsupported file type: {path.suffix}")


def _require_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataFormatError(
            f"Missing columns {missing} for {context}. Present columns: {list(df.columns)}"
        )


def _extract_xyz(df: pd.DataFrame, context: str) -> np.ndarray:
    # Accept x,y,z or X,Y,Z with varying whitespace; columns already normalized
    cols = [c for c in df.columns]
    needed = ["x", "y", "z"]
    _require_columns(df, needed, context)
    arr = df[needed].astype(float).to_numpy()
    if not np.all(np.isfinite(arr)):
        raise DataFormatError(f"Non-finite coordinates in {context}")
    return arr


# ---------- public loaders ----------

def load_points_robot(path: Path | str) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Load reference points in the robot base frame.

    Returns
    -------
    points : (N,3) float64 array
    names  : list[str] or None if no 'name' column was provided
    """
    df = _read_table(Path(path))
    context = "robot reference points"
    pts = _extract_xyz(df, context)
    names = None
    if "name" in df.columns:
        names = [str(n).strip() for n in df["name"].tolist()]
    return pts, names


def load_points_tracker(path: Path | str) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Load reference points in the tracker frame. Same contract as load_points_robot."""
    df = _read_table(Path(path))
    context = "tracker reference points"
    pts = _extract_xyz(df, context)
    names = None
    if "name" in df.columns:
        names = [str(n).strip() for n in df["name"].tolist()]
    return pts, names


def validate_correspondence(names_robot: Optional[Sequence[str]],
                            names_tracker: Optional[Sequence[str]],
                            n_robot: int,
                            n_tracker: int) -> None:
    """
    Ensure robot ↔ tracker reference sets are comparable.
    - Same count
    - If names provided for both, same multiset and same order
    - Require N >= 4
    """
    if n_robot != n_tracker:
        raise DataFormatError(f"Mismatched count of reference points: robot={n_robot}, tracker={n_tracker}")
    if n_robot < 4:
        raise DataFormatError("Need at least 4 non-coplanar reference points for a stable SVD estimate")

    if names_robot is not None and names_tracker is not None:
        if list(names_robot) != list(names_tracker):
            # Try set equality to guide the user
            if set(names_robot) == set(names_tracker):
                raise DataFormatError(
                    "Name sets match but order differs. Sort both files identically by 'name'."
                )
            else:
                raise DataFormatError(
                    "Name sets differ between robot and tracker references. Make them consistent."
                )


def load_program_positions(path: Path | str) -> Dict[str, np.ndarray]:
    """
    Load program positions P1..P5.

    Returns dict mapping position label -> (3,) array.
    Accepted 'position' labels are arbitrary strings (e.g., 'P1').
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
        df = _normalize_columns(df)
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "positions" in data:
            data = data["positions"]
        if not isinstance(data, list):
            raise DataFormatError("JSON must contain a list under 'positions' or at root for program positions")
        df = pd.DataFrame(data)
        df = _normalize_columns(df)
    else:
        raise DataFormatError(f"Unsupported file type for program positions: {path.suffix}")

    _require_columns(df, ["position", "x", "y", "z"], "program positions")
    pos = {}
    for _, row in df.iterrows():
        label = str(row["position"]).strip()
        vec = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float)
        if label in pos:
            raise DataFormatError(f"Duplicate program position label: {label}")
        pos[label] = vec
    if not pos:
        raise DataFormatError("No program positions found")
    return pos


def load_measurements_csv(path: Path | str,
                          required_positions: Optional[Iterable[str]] = None,
                          expected_cycles: Optional[int] = 30) -> Dict[str, np.ndarray]:
    """
    Load measurement points already expressed in the robot base frame after applying T_RT.

    Parameters
    ----------
    path : CSV path with columns: cycle, position, x, y, z
    required_positions : optional iterable of expected labels (e.g., {"P1",...,"P5"})
    expected_cycles : expected number of cycles per position; if None, skip this check

    Returns
    -------
    dict: position -> (N,3) float64 array (ordered by cycle ascending if 'cycle' is present)
    """
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    _require_columns(df, ["position", "x", "y", "z"], "measurements")

    # Optional cycle column for ordering
    if "cycle" in df.columns:
        # Coerce to int where possible
        try:
            df["cycle"] = pd.to_numeric(df["cycle"], errors="raise").astype(int)
        except Exception:
            raise DataFormatError("Column 'cycle' must be integer if present")
        df = df.sort_values(["position", "cycle"]).reset_index(drop=True)

    # Group and build arrays
    groups = df.groupby("position", sort=False)
    out: Dict[str, np.ndarray] = {}
    for label, g in groups:
        arr = _extract_xyz(g, f"measurements for position {label}")
        out[str(label).strip()] = arr

    if not out:
        raise DataFormatError("No measurements found")

    if required_positions is not None:
        req = {str(p).strip() for p in required_positions}
        have = set(out.keys())
        missing = sorted(req - have)
        extra = sorted(have - req)
        if missing:
            raise DataFormatError(f"Missing positions in measurements: {missing}")
        if extra:
            raise DataFormatError(f"Unexpected positions in measurements: {extra}")

    if expected_cycles is not None:
        bad = {k: v.shape[0] for k, v in out.items() if v.shape[0] != expected_cycles}
        if bad:
            raise DataFormatError(
                "Wrong number of cycles per position: "
                + ", ".join(f"{k}:{n}" for k, n in bad.items())
            )

    return out


# ---------- simple CLI probe ----------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Probe loaders and print dataset summary")
    p.add_argument("--ref_robot", type=Path, required=True)
    p.add_argument("--ref_tracker", type=Path, required=True)
    p.add_argument("--prog", type=Path, required=True)
    p.add_argument("--meas", type=Path, required=True)
    args = p.parse_args()

    rr, nn_r = load_points_robot(args.ref_robot)
    rt, nn_t = load_points_tracker(args.ref_tracker)
    validate_correspondence(nn_r, nn_t, rr.shape[0], rt.shape[0])

    prog = load_program_positions(args.prog)
    meas = load_measurements_csv(args.meas, required_positions=prog.keys(), expected_cycles=30)

    print(f"robot refs:    N={rr.shape[0]}")
    print(f"tracker refs:  N={rt.shape[0]}")
    print(f"program pos:   {sorted(prog.keys())}")
    print("measurements:  " + ", ".join(f"{k}:{v.shape}" for k, v in meas.items()))
