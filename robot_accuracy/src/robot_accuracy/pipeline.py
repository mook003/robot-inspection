from __future__ import annotations


from typing import Dict, List, Tuple
import numpy as np


from .transform import estimate_rt_svd, TransformRT
from .metrics import compute_ap, compute_rp




def transform_and_compute(
    P_R: np.ndarray,
    P_T: np.ndarray,
    prog: dict[str, np.ndarray],
    meas_tracker: Dict[str, np.ndarray],
    max_resid: float | None = None,
):
    """
    Этап 2: оценить T_RT (трекер->робот), преобразовать измерения в базу робота,
    посчитать метрики AP/RP по каждой позиции.


    Returns
    -------
    T : TransformRT
    residuals : (N,) остатки калибровки, мм
    max_r : float
    rms_r : float
    meas_robot : dict position -> (M,3) измерения в базе робота
    rows : list[dict] метрики по позициям для отчёта
    """
    T, residuals, max_r, rms_r = estimate_rt_svd(P_R, P_T)
    if max_resid is not None and max_r > max_resid:
        raise ValueError(f"Max residual {max_r:.3f} mm exceeds limit {max_resid:.3f} mm")


    meas_robot: Dict[str, np.ndarray] = {k: T.apply(v) for k, v in meas_tracker.items()}


    rows: List[dict] = []
    for pos in sorted(prog.keys()):
        if pos not in meas_robot:
            raise KeyError(f"Missing measurements for position {pos}")
        arr = meas_robot[pos]
        c = arr.mean(axis=0)
        L_bar, sigma, RP = compute_rp(arr)
        AP = compute_ap(c, prog[pos])
        d = c - prog[pos]
        rows.append({
            "position": pos,
            "prog_x": float(prog[pos][0]),
            "prog_y": float(prog[pos][1]),
            "prog_z": float(prog[pos][2]),
            "mean_x": float(c[0]), 
            "mean_y": float(c[1]), 
            "mean_z": float(c[2]),
            "dX": float(d[0]), 
            "dY": float(d[1]), 
            "dZ": float(d[2]),
            "AP": float(AP),
            "L_bar": float(L_bar), 
            "sigma": float(sigma), 
            "RP": float(RP),
            "n": int(arr.shape[0]),
        })


    return T, residuals, max_r, rms_r, meas_robot, rows