# `robot_accuracy.io`

Data loading and validation utilities. All loaders accept CSV or JSON input.

---

## `DataFormatError`

```python
class DataFormatError(ValueError)
```

Raised by any loader or validator when the input file has structural issues (missing columns, wrong type, duplicate labels, etc.).

---

## `load_points_robot`

```python
def load_points_robot(path: Path | str) -> tuple[np.ndarray, list[str] | None]
```

Load reference points in the **robot base frame**.

**Returns:** `(points, names)` where `points` is `(N, 3)` float64 and `names` is a list of strings or `None` if no `name` column is present.

---

## `load_points_tracker`

```python
def load_points_tracker(path: Path | str) -> tuple[np.ndarray, list[str] | None]
```

Identical contract to `load_points_robot`, for tracker-frame reference points.

---

## `validate_correspondence`

```python
def validate_correspondence(
    names_robot: Sequence[str] | None,
    names_tracker: Sequence[str] | None,
    n_robot: int,
    n_tracker: int,
) -> None
```

Checks that both reference sets are compatible:

- Same point count
- `N ≥ 4` (minimum for stable SVD)
- If both have names: same order (not just same set)

Raises `DataFormatError` with a descriptive message on failure.

---

## `load_program_positions`

```python
def load_program_positions(path: Path | str) -> dict[str, np.ndarray]
```

Load nominal robot positions (ground truth).

**Returns:** dict mapping position label → `(3,)` float64 array.

**Raises:** `DataFormatError` on duplicate labels or missing columns.

---

## `load_measurements_csv`

```python
def load_measurements_csv(
    path: Path | str,
    required_positions: Iterable[str] | None = None,
    expected_cycles: int | None = 30,
) -> dict[str, np.ndarray]
```

Load tracker measurements and group by position.

**Parameters**

| Name | Description |
|---|---|
| `path` | CSV with columns `cycle, position, x, y, z` |
| `required_positions` | Set of expected position labels; raises if any are missing or extra |
| `expected_cycles` | Expected cycle count per position; `None` skips this check |

**Returns:** dict mapping position label → `(M, 3)` float64 array, rows ordered by `cycle`.
