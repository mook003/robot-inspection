# `robot_accuracy.metrics`

Pure-NumPy functions for computing AP and RP metrics.

---

## `compute_ap`

```python
def compute_ap(mean_meas: np.ndarray, prog: np.ndarray) -> float
```

Compute **Accuracy (AP)** for a single position.

$$AP = \| \bar{x} - p_{nom} \|_2$$

**Parameters**

| Name | Type | Description |
|---|---|---|
| `mean_meas` | array-like, shape `(3,)` | Centroid of measurements in the robot frame |
| `prog` | array-like, shape `(3,)` | Nominal position from the program |

**Returns:** AP value in mm (float).

---

## `compute_rp`

```python
def compute_rp(meas: np.ndarray) -> tuple[float, float, float]
```

Compute **Repeatability (RP)** for a single position.

$$RP = \bar{L} + 3\sigma$$

**Parameters**

| Name | Type | Description |
|---|---|---|
| `meas` | `(M, 3)` ndarray | All measurement cycles in the robot frame |

**Returns:** `(L_bar, sigma, RP)` — all in mm.

| Value | Description |
|---|---|
| `L_bar` | Mean distance from measurements to their centroid |
| `sigma` | Standard deviation of distances (ddof=1) |
| `RP` | `L_bar + 3 * sigma` |

!!! note "Single-point edge case"
    If `M == 1`, `sigma` is set to `0.0` and `RP = L_bar`.
