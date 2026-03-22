# CLI Reference

The command-line interface is available as `robot-accuracy` (after install) or via `python -m robot_accuracy`.

## Synopsis

```
robot-accuracy [OPTIONS]
```

## Options

| Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `--ref-robot` | path | ✅ | — | CSV/JSON with reference points in the robot base frame |
| `--ref-tracker` | path | ✅ | — | CSV/JSON with reference points in the tracker frame |
| `--prog` | path | ✅ | — | CSV/JSON with nominal program positions |
| `--meas` | path | ✅ | — | CSV with measurements in the tracker frame |
| `--cycles` | int | | `30` | Expected number of measurement cycles per position |
| `--max-resid` | float | | `0.1` | Maximum allowed calibration residual (mm). Aborts if exceeded. |
| `--out` | path | | — | Output file path. `.xlsx` → Excel; any other extension → two CSV files |
| `--dry-run` | flag | | `false` | Print tables to stdout without saving any file |

## Behaviour

1. All four input files are loaded and validated.
2. `T_RT` is estimated from the reference points.
3. If `--max-resid` is given and the max calibration residual exceeds it, the program exits with code `2`.
4. Measurements are transformed and AP/RP are computed per position.
5. If `--out` is provided, the report is saved. Otherwise (or with `--dry-run`) the tables are printed to stdout.

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `2` | Input error (bad file, wrong format, residual exceeded) |

## Examples

**Basic run, save to Excel:**
```bash
robot-accuracy \
  --ref-robot data/ref_robot.csv \
  --ref-tracker data/ref_tracker.csv \
  --prog data/program.csv \
  --meas data/measurements.csv \
  --out results.xlsx
```

**Preview without saving:**
```bash
robot-accuracy \
  --ref-robot data/ref_robot.csv \
  --ref-tracker data/ref_tracker.csv \
  --prog data/program.csv \
  --meas data/measurements.csv \
  --dry-run
```

**Save to CSV pair (`results.rt.csv` + `results.metrics.csv`):**
```bash
robot-accuracy ... --out results.csv
```

**Strict residual threshold:**
```bash
robot-accuracy ... --max-resid 0.05
```
