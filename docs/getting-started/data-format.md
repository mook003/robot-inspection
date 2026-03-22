# Data Format

All input files are **CSV** (or JSON). Column names are case-insensitive; leading/trailing spaces are stripped automatically.

!!! note "Units"
    All coordinates are in **millimetres**. No unit conversion is performed.

---

## Reference Points — Robot Frame (`--ref-robot`)

Points measured or provided in the **robot base frame**. Used as the *target* side of the Kabsch estimation.

```csv
name,x,y,z
A,5.000,-3.000,2.000
B,104.905,0.489,-0.618
C,1.556,96.925,3.745
D,7.677,-4.653,101.951
```

- `name` column is optional but recommended for correspondence validation.
- Minimum **4 non-coplanar** points required for a stable SVD estimate.

---

## Reference Points — Tracker Frame (`--ref-tracker`)

The same physical points measured by the tracker. Used as the *source* side.

```csv
name,x,y,z
A,0.000,0.000,0.000
B,100.000,0.000,0.000
C,0.000,100.000,0.000
D,0.000,0.000,100.000
```

!!! warning "Order matters"
    Row order must match `--ref-robot`. If both files have a `name` column, the names must appear in the same order — otherwise the loader raises a `DataFormatError`.

---

## Program Positions (`--prog`)

Ground-truth (nominal) robot positions in the **robot base frame**.

```csv
position,x,y,z
P1,10.000,10.000,10.000
P2,50.000,10.000,10.000
P3,10.000,50.000,10.000
P4,10.000,10.000,50.000
P5,50.000,50.000,50.000
```

- `position` labels are arbitrary strings (e.g. `P1`, `pos_01`).
- No duplicate labels allowed.

---

## Measurements (`--meas`)

Tracker measurements for each robot position, expressed in the **tracker frame**. The pipeline applies `T_RT` internally.

```csv
cycle,position,x,y,z
1,P1,9.986,9.983,9.989
2,P1,10.070,9.987,9.850
3,P1,10.033,9.973,9.978
...
1,P2,49.991,10.012,9.997
2,P2,50.023,9.968,10.031
...
```

- `cycle` must be an integer and is used for row ordering.
- Every position listed in `--prog` must appear in this file.
- The number of cycles per position must equal `--cycles` (default: 30).

---

## JSON Input

Both point files and the program file also accept JSON:

```json
[
  {"name": "A", "x": 0, "y": 0, "z": 0},
  {"name": "B", "x": 100, "y": 0, "z": 0}
]
```

Or wrapped:

```json
{"points": [...]}
```

For program positions:

```json
{"positions": [{"position": "P1", "x": 10, "y": 10, "z": 10}]}
```
