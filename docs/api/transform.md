# `robot_accuracy.transform`

Rigid transformation between the tracker frame and the robot base frame.

---

## `TransformRT`

```python
@dataclass(frozen=True)
class TransformRT:
    R: np.ndarray  # (3, 3) rotation matrix
    t: np.ndarray  # (3,)  translation vector
```

Represents the rigid transform `p_R = R @ p_T + t`.

### Methods

#### `apply(pts)`

```python
def apply(self, pts: np.ndarray) -> np.ndarray
```

Apply the transform to an array of points.

| Parameter | Type | Description |
|---|---|---|
| `pts` | `(N, 3)` ndarray | Points in the tracker frame |

Returns `(N, 3)` ndarray — points in the robot base frame.

---

#### `invert()`

```python
def invert(self) -> TransformRT
```

Returns the inverse transform (robot → tracker frame).

$$R^{-1} = R^\top, \quad t^{-1} = -R^\top t$$

---

#### `as_homogeneous()`

```python
def as_homogeneous(self) -> np.ndarray
```

Returns the `(4, 4)` homogeneous transformation matrix.

---

## `estimate_rt_svd`

```python
def estimate_rt_svd(
    P_R: np.ndarray,
    P_T: np.ndarray,
) -> tuple[TransformRT, np.ndarray, float, float]
```

Estimate the rigid transform from corresponding point pairs using the **Kabsch algorithm**.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `P_R` | `(N, 3)` ndarray | Target points in the robot base frame |
| `P_T` | `(N, 3)` ndarray | Source points in the tracker frame |

**Returns**

| Name | Type | Description |
|---|---|---|
| `T` | `TransformRT` | Estimated transform with `det(R) = +1` |
| `resid` | `(N,)` ndarray | Per-point Euclidean residuals (mm) |
| `max_r` | float | Maximum residual (mm) |
| `rms_r` | float | RMS residual (mm) |

**Raises**

- `ValueError` — if `N < 3`, shapes mismatch, NaN/Inf values, or points are collinear.

---

## `apply_rt` / `invert_rt`

Functional alternatives that work directly with `(R, t)` arrays without the dataclass:

```python
def apply_rt(R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray
def invert_rt(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]
```
