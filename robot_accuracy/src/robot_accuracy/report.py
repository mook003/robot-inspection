from __future__ import annotations

from pathlib import Path
import pandas as pd

def build_tables(*_, **__):
    """Заглушка: финальные таблицы соберём после реализации ядра."""
    return None

def save_report(path: Path, df_rt: pd.DataFrame, df_metrics: pd.DataFrame) -> None:
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(path) as xls:
            df_rt.to_excel(xls, sheet_name="T_RT", index=False)
            df_metrics.to_excel(xls, sheet_name="Metrics", index=False)
    else:
        # По умолчанию пишем две CSV рядом
        df_rt.to_csv(path.with_suffix(".rt.csv"), index=False)
        df_metrics.to_csv(path.with_suffix(".metrics.csv"), index=False)