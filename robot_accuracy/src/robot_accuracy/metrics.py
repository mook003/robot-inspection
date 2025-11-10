from __future__ import annotations
import numpy as np

def compute_ap(mean_meas: np.ndarray, prog: np.ndarray) -> float:
    mean_meas = np.asarray(mean_meas, float).reshape(3)
    prog = np.asarray(prog, float).reshape(3)
    return float(np.linalg.norm(mean_meas - prog))

def compute_rp(meas: np.ndarray, ) -> tuple[float, float, float]:
    """
    meas: (M,3). Возвращает (L_bar, sigma, RP=L_bar+3*sigma), мм
    """
    meas = np.asarray(meas, float)
    c = meas.mean(axis=0)
    d = np.linalg.norm(meas - c, axis=1)
    L = float(d.mean())
    sigma = float(d.std(ddof=1)) if d.size > 1 else 0.0
    return L, sigma, L + 3.0 * sigma