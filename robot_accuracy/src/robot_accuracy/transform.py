from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class TransformRT:
    """
    Жёсткое преобразование: трекер -> база робота
        p_R = R @ p_T + t
    """
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        return (self.R @ pts.T).T + self.t.reshape(1, 3)

    def invert(self) -> "TransformRT":
        R_T = self.R.T
        return TransformRT(R=R_T, t=-R_T @ self.t)

    def as_homogeneous(self) -> np.ndarray:
        H = np.eye(4, dtype=float)
        H[:3, :3] = self.R
        H[:3, 3] = self.t
        return H


def estimate_rt_svd(P_R: np.ndarray, P_T: np.ndarray) -> tuple[TransformRT, np.ndarray, float, float]:
    """
    Оценка R,t методом Kabsch (SVD).

    Вход
    ----
    P_R : (N,3) точки в базе робота (целевые)
    P_T : (N,3) точки в системе трекера (исходные)

    Выход
    -----
    T     : TransformRT с det(R)=+1, такое что p_R = R p_T + t
    resid : (N,) евклидовы остатки (в тех же единицах, что вход)
    max_r : максимальный остаток
    rms_r : RMS-остаток
    """
    P_T = np.asarray(P_T, dtype=float)
    P_R = np.asarray(P_R, dtype=float)

    if P_T.shape != P_R.shape or P_T.ndim != 2 or P_T.shape[1] != 3:
        raise ValueError("P_R и P_T должны иметь одинаковую форму (N,3)")
    N = P_T.shape[0]
    if N < 3:
        raise ValueError("Нужно минимум 3 соответствующие точки")

    if not np.all(np.isfinite(P_T)) or not np.all(np.isfinite(P_R)):
        raise ValueError("Обнаружены NaN/Inf во входных данных")

    # Правильное центрирование
    P_T_mean = P_T.mean(axis=0)
    P_R_mean = P_R.mean(axis=0)
    X = P_T - P_T_mean
    Y = P_R - P_R_mean

    # Минимальное требование: не коллинеарность (ранг >= 2)
    if np.linalg.matrix_rank(X) < 2 or np.linalg.matrix_rank(Y) < 2:
        raise ValueError("Опорные точки вырождены (коллинеарны или почти)")

    # Кросс-ковариация и SVD
    H = X.T @ Y  # (3x3)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Дет-фикс против отражения
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = P_R_mean - R @ P_T_mean

    T = TransformRT(R=R, t=t)

    mapped = T.apply(P_T)
    resid = np.linalg.norm(mapped - P_R, axis=1)
    max_r = float(resid.max())
    rms_r = float(np.sqrt((resid ** 2).mean()))

    return T, resid, max_r, rms_r


# Утилиты без датакласса
def apply_rt(R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float)
    return (np.asarray(R, float) @ pts.T).T + np.asarray(t, float).reshape(1, 3)


def invert_rt(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_T = np.asarray(R, float).T
    return R_T, -R_T @ np.asarray(t, float)
