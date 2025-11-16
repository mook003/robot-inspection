# robot-accuracy

Мини-библиотека и CLI для оценки трансформации `T_RT` (трекер → база робота) по опорным точкам и расчёта метрик точности позиционирования (AP) и повторяемости (RP) по сериям измерений.

## Возможности

- Оценка жёсткого преобразования `p_R = R · p_T + t` методом Kabsch (SVD) с коррекцией `det(R)=+1`.
- Преобразование измерений трекера в систему робота.
- Расчёт:
  - **AP** (Accuracy): расстояние между средним измерением и референтной (реальной) координатой точки.
  - **RP** (Repeatability): `L̄ + 3σ`, где `L̄` — среднее расстояние до центроида серии, `σ` — СКО расстояний.
- Экспорт отчёта в `XLSX` либо в пару CSV-файлов.

## Структура репозитория

```
├── data                       #test data
│   ├── measurements.csv 
│   ├── program.csv
│   ├── ref_robot.csv
│   └── ref_tracker.csv
├── robot_accuracy             # main python lib
│   ├── LICENSE
│   ├── pyproject.toml
│   ├── README.md
│   └── src
│       ├── robot_accuracy
│       │   ├── cli.py         #command line interface
│       │   ├── __init__.py
│       │   ├── io.py          #data parser 
│       │   ├── __main__.py
│       │   ├── metrics.py     #metics computation
│       │   ├── pipeline.py    #pipeline of transform and metrics computation 
│       │   ├── report.py      #report building
│       │   ├── transform.py   #transform computation
│       │   └── version.py     #version of lib
│       └── tests              #tests (TBA)
└── scripts                    #additional scripts
    └── gen_demo_data.py       #script for test data generation
```

## Требования

- Python ≥ 3.10
- NumPy, Pandas
- Для экспорта в XLSX: `openpyxl`

## Установка

В режиме разработки (editable):

```bash
pip install -e robot_accuracy[excel]
````

Если среда не поддерживает PEP 660 (editable через `pyproject.toml`), либо обновите `pip`/`setuptools`, либо используйте запуск без установки (см. ниже).

## Быстрый старт

### 1) Подготовьте входные файлы

Форматы CSV:

**Опорные точки (база робота и трекер)**

```csv
name,x,y,z
A,0,0,0
B,100,0,0
C,0,100,0
D,0,0,100
```

**Реальные координаты позиций (ground truth)**

```csv
position,x,y,z
P1,10,10,10
P2,50,10,10
...
```

**Измерения (в системе трекера)**

```csv
cycle,position,x,y,z
1,P1,10.02,9.99,10.01
2,P1,9.98,10.05,9.97
...
```

>`note:` Ожидается одинаковое количество циклов (например, 30) на каждую позицию.

### 2) Запуск через установленный пакет

```bash
robot-accuracy \
  --ref-robot data/ref_robot.csv \
  --ref-tracker data/ref_tracker.csv \
  --real data/real_positions.csv \
  --prog data/program.csv \
  --meas data/measurements.csv \
  --cycles 30 \
  --max-resid 0.1 \
  --out results.xlsx
```

### 3) Запуск без установки (через `PYTHONPATH`)

```bash
PYTHONPATH=robot_accuracy/src python -m robot_accuracy \
  --ref-robot data/ref_robot.csv \
  --ref-tracker data/ref_tracker.csv \
  --real data/real_positions.csv \
  --prog data/program.csv \
  --meas data/measurements.csv \
  --cycles 30 \
  --max-resid 0.1 \
  --out results.xlsx
```

`--dry-run` выведет сводку без сохранения файлов.

## Что считается

* **T_RT**: по наборам соответствующих опорных точек `P_T` (трекер) и `P_R` (робот).
* **AP**: `‖ mean(measurements_R) − real_point ‖₂`.
* **RP**: для каждой позиции:

  * `c = mean(measurements_R)`,
  * `d_i = ‖ x_i − c ‖₂`,
  * `L̄ = mean(d_i)`, `σ = std(d_i, ddof=1)`,
  * `RP = L̄ + 3σ`.
