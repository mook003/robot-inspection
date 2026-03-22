# `robot_accuracy.report`

Functions for assembling and saving the output report.

---

## `build_tables`

```python
def build_tables(
    R: np.ndarray,
    t: np.ndarray,
    residuals: np.ndarray,
    rows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]
```

Assemble two DataFrames from the pipeline results.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `R` | `(3, 3)` ndarray | Rotation matrix from `T_RT` |
| `t` | `(3,)` ndarray | Translation vector from `T_RT` |
| `residuals` | `(N,)` ndarray | Per-point calibration residuals (mm) |
| `rows` | list of dicts | Per-position metric rows from `transform_and_compute` |

**Returns:** `(df_rt, df_metrics)`

### `df_rt` — Transform Sheet

| Column | Description |
|---|---|
| `R11` … `R33` | All 9 elements of the rotation matrix |
| `tX`, `tY`, `tZ` | Translation vector (mm) |
| `max_residual` | Max calibration residual (mm) |
| `rms_residual` | RMS calibration residual (mm) |
| `N_pairs` | Number of reference point pairs used |

### `df_metrics` — Metrics Sheet

| Column | Description |
|---|---|
| `position` | Position label |
| `prog_x/y/z` | Nominal coordinates (mm) |
| `mean_x/y/z` | Centroid of measurements in robot frame (mm) |
| `dX/dY/dZ` | Offset `mean − prog` (mm) |
| `AP` | Accuracy (mm) |
| `L_bar` | Mean distance to centroid (mm) |
| `sigma` | Std dev of distances, ddof=1 (mm) |
| `RP` | Repeatability = `L̄ + 3σ` (mm) |
| `n` | Cycle count |

---

## `save_report`

```python
def save_report(
    path: Path,
    df_rt: pd.DataFrame,
    df_metrics: pd.DataFrame,
) -> None
```

Save the two DataFrames to disk.

**Behaviour by extension:**

| Extension | Output |
|---|---|
| `.xlsx` | Single Excel file with two sheets: `T_RT` and `Metrics`. Requires `openpyxl`. |
| Any other | Two CSV files: `<stem>.rt.csv` and `<stem>.metrics.csv` |

**Example:**

```python
from pathlib import Path
from robot_accuracy.report import build_tables, save_report

df_rt, df_metrics = build_tables(T.R, T.t, residuals, rows)

# Excel
save_report(Path("results.xlsx"), df_rt, df_metrics)

# CSV pair: results.rt.csv + results.metrics.csv
save_report(Path("results.csv"), df_rt, df_metrics)
```
