# ML Failure Investigator

Diagnosing **why** ML models fail: data quality issues, distribution drift, and performance drops — and measuring how accurately those diagnoses can be made through controlled failure experiments.

> 🚧 Work in progress. Current stage: **data loading & cleaning (Week 1).**

## Core idea

Most drift tools answer *"did the data change?"*. This project also asks *"how often is the diagnosis right?"*:
known failures (missing values, feature shift, label noise, leakage) are injected on purpose, and the detector is evaluated on detection rate and false-alarm rate.

## Project structure

```
ml-failure-investigator/
├── data/raw/          # raw Adult files (not tracked by git)
├── notebooks/         # exploration
├── src/mlfi/          # the package
│   ├── config.py      # paths, column names, target, random seed
│   └── data.py        # load_raw() + clean()
├── tests/             # pytest tests
└── pyproject.toml
```

## Setup (Windows / PowerShell)

```powershell
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data

[UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult) — download and place `adult.data`, `adult.test`, `adult.names` into `data/raw/`.

## Run tests

```powershell
pytest
```

## Roadmap

- [x] Project skeleton
- [ ] Data cleaning (`clean()`)
- [ ] Data quality report
- [ ] Baseline model
- [ ] Drift detection (KS / chi-square / PSI)
- [ ] Controlled failure injection + evaluation
