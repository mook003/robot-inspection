# Quick Start

This guide walks through a full run using the demo data included in the repository.

## Step 1 — Generate Demo Data (optional)

The repository ships with ready-made data in `data/`. To regenerate it (or create a variant with different noise/rotation):

```bash
python scripts/gen_demo_data.py \
  --out data \
  --cycles 30 \
  --noise 0.10 \
  --rot 1.0 1.5 2.0 \
  --trans 5.0 -3.0 2.0
```

This writes four CSV files: `ref_tracker.csv`, `ref_robot.csv`, `program.csv`, `measurements.csv`.

## Step 2 — Run the Pipeline

=== "Installed (entry-point)"
    ```bash
    robot-accuracy \
      --ref-robot   data/ref_robot.csv \
      --ref-tracker data/ref_tracker.csv \
      --prog        data/program.csv \
      --meas        data/measurements.csv \
      --cycles      30 \
      --max-resid   0.1 \
      --out         results.xlsx
    ```

=== "Without install (PYTHONPATH)"
    ```bash
    PYTHONPATH=robot_accuracy/src python -m robot_accuracy \
      --ref-robot   data/ref_robot.csv \
      --ref-tracker data/ref_tracker.csv \
      --prog        data/program.csv \
      --meas        data/measurements.csv \
      --cycles      30 \
      --max-resid   0.1 \
      --out         results.xlsx
    ```

## Step 3 — Check the Output

The terminal prints a summary:

```
refs: robot=4 tracker=4
program positions: ['P1', 'P2', 'P3', 'P4']
measurements:  P1:30, P2:30, P3:30, P4:30
T_RT: max residual = 0.00000 mm, rms = 0.00000 mm
Saved: results.xlsx
```

The output file `results.xlsx` contains two sheets:

| Sheet | Contents |
|---|---|
| `T_RT` | 3×3 rotation matrix `R`, translation vector `t`, residual stats |
| `Metrics` | Per-position: program coords, mean measurement, `dX/dY/dZ`, `AP`, `L̄`, `σ`, `RP` |

## Dry Run (no file saved)

Add `--dry-run` to print the tables to stdout without saving:

```bash
robot-accuracy ... --dry-run
```

## Python API Usage

```python
from robot_accuracy.io import (
    load_points_robot, load_points_tracker,
    load_program_positions, load_measurements_csv,
    validate_correspondence,
)
from robot_accuracy.pipeline import transform_and_compute
from robot_accuracy.report import build_tables, save_report
from pathlib import Path

P_R, names_r = load_points_robot("data/ref_robot.csv")
P_T, names_t = load_points_tracker("data/ref_tracker.csv")
validate_correspondence(names_r, names_t, P_R.shape[0], P_T.shape[0])

prog = load_program_positions("data/program.csv")
meas = load_measurements_csv("data/measurements.csv",
                             required_positions=prog.keys(),
                             expected_cycles=30)

T, residuals, max_r, rms_r, meas_robot, rows = transform_and_compute(
    P_R, P_T, prog, meas, max_resid=0.1
)

df_rt, df_metrics = build_tables(T.R, T.t, residuals, rows)
save_report(Path("results.xlsx"), df_rt, df_metrics)
```
