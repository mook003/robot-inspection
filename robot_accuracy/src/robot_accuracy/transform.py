from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class TransformRT:
    """
    Жёсткое преобразование трекер -> база робота:
        p_R = R @ p_T + t
    """
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float)
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
    Вычислить T_RT по парам соответствующих точек методом Kabsch (SVD).

    Вход
    ----
    P_R : (N,3) точки в базе робота (целевые)
    P_T : (N,3) точки в системе трекера (исходные)

    Выход
    -----
    T     : TransformRT с det(R)=+1
    resid : (N,) евклидовы остатки, мм
    max_r : float, максимальный остаток, мм
    rms_r : float, rms-остаток, мм
    """
    A = np.asarray(P_T, dtype=float)
    B = np.asarray(P_R, dtype=float)

    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("P_R и P_T должны иметь одинаковую форму (N,3)")
    N = A.shape[0]
    if N < 4:
        raise ValueError("Нужно минимум 4 точки")

    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
        raise ValueError("Обнаружены нечисловые значения в входных данных")

    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    Ac = A - a_mean
    Bc = B - b_mean

    # Проверка вырожденности (хотя бы ранга 3 для устойчивости)
    if np.linalg.matrix_rank(Ac) < 3 or np.linalg.matrix_rank(Bc) < 3:
        raise ValueError("Опорные точки вырождены (почти коллинеарны/копланарны)")

    H = Ac.T @ Bc  # (3x3)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Коррекция отражения
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = b_mean - R @ a_mean

    mapped = (R @ A.T).T + t.reshape(1, 3)
    resid = np.linalg.norm(mapped - B, axis=1)
    max_r = float(resid.max())
    rms_r = float(np.sqrt((resid ** 2).mean()))

    return TransformRT(R=R, t=t), resid, max_r, rms_r


# Утилиты без датакласса (на случай прямого использования)
def apply_rt(R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float)
    return (np.asarray(R, float) @ pts.T).T + np.asarray(t, float).reshape(1, 3)


def invert_rt(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_T = np.asarray(R, float).T
    return R_T, -R_T @ np.asarray(t, float)
