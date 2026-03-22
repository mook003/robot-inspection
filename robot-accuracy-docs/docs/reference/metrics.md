# Metrics: AP and RP

## Accuracy (AP)

**AP** measures how close the robot's mean measured position is to the nominal (programmed) position.

$$AP = \| \bar{x} - p_{nom} \|_2$$

where:

- $\bar{x} = \frac{1}{M} \sum_{j=1}^{M} x_j$ — centroid of the `M` repeated measurements in the robot frame
- $p_{nom}$ — nominal (ground-truth) position from `program.csv`

A lower AP value indicates better **systematic accuracy** — the robot consistently reaches close to the commanded point.

---

## Repeatability (RP)

**RP** measures the spread of repeated measurements around their own centroid, regardless of offset from the nominal point.

$$RP = \bar{L} + 3\sigma$$

where:

$$d_j = \| x_j - \bar{x} \|_2, \quad \bar{L} = \frac{1}{M}\sum d_j, \quad \sigma = \sqrt{\frac{1}{M-1}\sum (d_j - \bar{L})^2}$$

The `3σ` factor assumes normally distributed distances and captures ~99.7 % of measurements.

A lower RP value indicates better **mechanical repeatability** — the robot reliably returns to the same pose.

---

## Interpreting Results

| Scenario | AP | RP | Meaning |
|---|---|---|---|
| Ideal | low | low | Accurate and repeatable |
| Good calibration, worn mechanics | low | high | On target on average, but scattered |
| Bad calibration, good mechanics | high | low | Consistent offset — can be corrected |
| Both bad | high | high | Systematic and random errors present |

---

## Output Columns

The `Metrics` sheet in the report contains:

| Column | Description |
|---|---|
| `position` | Position label (e.g. `P1`) |
| `prog_x/y/z` | Nominal coordinates (mm) |
| `mean_x/y/z` | Centroid of measurements in robot frame (mm) |
| `dX/dY/dZ` | Component-wise offset `mean − prog` (mm) |
| `AP` | Accuracy (mm) |
| `L_bar` | Mean distance to centroid (mm) |
| `sigma` | Std dev of distances (mm, ddof=1) |
| `RP` | Repeatability = `L̄ + 3σ` (mm) |
| `n` | Number of measurement cycles used |
