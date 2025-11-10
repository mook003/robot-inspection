from __future__ import annotations


from pathlib import Path
import numpy as np
import pandas as pd




def build_tables(R, t, residuals, rows):
    """
    rows: список словарей по позициям (из CLI)
    Возвращает (df_rt, df_metrics)
    """
    residuals = np.asarray(residuals, float)
    df_rt = pd.DataFrame({
        "R11": [R[0, 0]], "R12": [R[0, 1]], "R13": [R[0, 2]],
        "R21": [R[1, 0]], "R22": [R[1, 1]], "R23": [R[1, 2]],
        "R31": [R[2, 0]], "R32": [R[2, 1]], "R33": [R[2, 2]],
        "tX": [t[0]], "tY": [t[1]], "tZ": [t[2]],
        "max_residual": [float(residuals.max())],
        "rms_residual": [float(np.sqrt((residuals**2).mean()))],
        "N_pairs": [int(residuals.size)],
    })


    # Упорядочим столбцы метрик
    df_metrics = pd.DataFrame(rows)
    order = [
    "position",
    "prog_x", "prog_y", "prog_z",
    "mean_x", "mean_y", "mean_z",
    "dX", "dY", "dZ",
    "AP", "L_bar", "sigma", "RP", "n"
    ]
    df_metrics = df_metrics[[c for c in order if c in df_metrics.columns]]
    return df_rt, df_metrics




def save_report(path: Path, df_rt: pd.DataFrame, df_metrics: pd.DataFrame) -> None:
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(path) as xls:
            df_rt.to_excel(xls, sheet_name="T_RT", index=False)
            df_metrics.to_excel(xls, sheet_name="Metrics", index=False)
    else:
        df_rt.to_csv(path.with_suffix(".rt.csv"), index=False)
        df_metrics.to_csv(path.with_suffix(".metrics.csv"), index=False)