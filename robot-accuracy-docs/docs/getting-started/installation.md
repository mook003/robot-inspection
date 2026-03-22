# Installation

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.24
- Pandas ≥ 2.0
- `openpyxl` ≥ 3.1 (optional, for XLSX export)

## Editable Install (recommended)

Clone the repository and install in editable mode with the optional `excel` extra:

```bash
git clone https://github.com/mook003/robot-inspection.git
cd robot-inspection

pip install -e robot_accuracy[excel]
```

This installs the `robot-accuracy` entry-point so you can call it directly:

```bash
robot-accuracy --help
```

## Without Installation

If your environment does not support PEP 660 editable installs (old `pip` / `setuptools`), run directly via `PYTHONPATH`:

```bash
PYTHONPATH=robot_accuracy/src python -m robot_accuracy --help
```

!!! tip "Upgrading pip/setuptools"
    To enable editable installs via `pyproject.toml`, upgrade your tools first:
    ```bash
    pip install --upgrade pip setuptools
    ```

## Verifying the Install

```bash
python -c "import robot_accuracy; print(robot_accuracy.__version__)"
# 0.0.1
```
