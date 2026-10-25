# Turbofan Remaining Useful Life Prediction

**Chua Shanwei · 35393343 · MMA3001 Numerical Methods and Machine Learning**

Estimate the remaining operating cycles of NASA C-MAPSS FD001 engines using
age-only linear regression, multi-sensor linear regression, random forest and
histogram gradient boosting. The code reproduces the student's current
Assignment results with engine-level validation and quantitative performance analysis.

**Opening the downloaded ZIP? Start with `START_HERE.html`.** It links directly
to the guide, HTML API documentation, licensing and readable test report.

## Required repository contents

| Requirement | Location | What to open |
|---|---|---|
| README and complete workflow instructions | This file | README.md or START_HERE.html |
| Generated HTML source-code documentation | [docs/api/](docs/api/index.html) | docs/api/index.html in a browser |
| Licensing agreement | [LICENSE](LICENSE) | MIT for original code; [NASA attribution](data/README.md) is separate |
| Functional source with comments and docstrings | [src/rul_project/](src/rul_project/) | Purpose-named modules listed below |
| Test functions and test report | [tests/](tests/README.md) | [Readable HTML report](tests/reports/pytest_report.html) and [pytest XML](tests/reports/pytest_results.xml) |
| Final accessible repository and zipped clone | [GitHub main](https://github.com/Shanwei0111/MMA3001Project) | Repository link for review; zipped clone for Moodle |

## Folder layout

```text
MMA3001Project/
├── START_HERE.html                Clickable entry point for the local clone
├── README.md                      Purpose, setup and reproduction commands
├── LICENSE                        Original source-code license
├── pyproject.toml                 Package metadata and test configuration
├── requirements.txt               Main dependencies
├── requirements-lock.txt          Exact verified environment, including transitive packages
├── src/rul_project/               Reusable, documented Python source
├── tests/
│   ├── test_data_and_metrics.py   RUL labels, grouping and hand-calculated errors
│   ├── test_repository_contracts.py  Data integrity, fixed settings and safe imports
│   ├── test_workflow_functions.py   Endpoint pairing, uncertainty and path handling
│   └── reports/                  Latest recorded HTML and XML test reports
├── docs/
│   ├── project_guide.html         Browser version of this README
│   ├── api/                      pdoc pages generated from current source
│   └── DEVELOPMENT.md            Actual branch history, refactor map and AI disclosure
├── data/
│   ├── README.md                 NASA source, access and dataset attribution
│   ├── manifest.json             Original FD001 SHA-256 fingerprints
│   └── raw/                      Verified source files; ignored by public Git history
├── results/
│   ├── reference/                Frozen report evidence, grouped by purpose
│   ├── verification/             Reproduction checks and source fingerprints
│   └── generated/                Ignored output from your new experiment runs
├── reports/                      Written project PDF and editable report text
├── tools/                        Documentation, verification and ZIP utilities
└── .github/workflows/            Automated tests and documentation synchronization
```

The hidden `.git` folder in a cloned repository stores version history. It is
not project code. The submission ZIP retains it so the clone includes its history.
Generated caches, environments and rerun outputs are excluded from the ZIP.

## Source files: where to read each part

| Python file in src/rul_project | Responsibility |
|---|---|
| `model_configuration.py` | Four estimator factories, random seed and ordered input features |
| `prepare_dataset.py` | Source parsing, uncapped training RUL, grouped splitting and test endpoint alignment |
| `train_models.py` | Baseline/ensemble training, five-fold grouped CV and forest depth study |
| `evaluate_test_set.py` | Final refitting, raw/clipped test metrics and retained failure cases |
| `rul_metrics.py` | MAE, RMSE, bias, overprediction percentage and asymmetric 13/10 score |
| `analyse_uncertainty.py` | Paired engine bootstrap for RF, HGB and multi-sensor linear regression |
| `benchmark_performance.py` | Timings, profiles, selected-array memory and operation estimates |
| `plot_model_results.py` | Validation baseline, official diagnostics and comparison figures |
| `data_access.py` | Verified local copying or official NASA archive acquisition |
| `run_project.py` | Explicit stage order and command-line path handling |
| `__main__.py`, `__init__.py` | Standard Python package entry points |

Functions receive explicit inputs and return tables or write clearly documented
files. Models and feature order are defined once in `model_configuration.py`.
Importing a module does not start an experiment. Google-style docstrings specify
parameters, returns, exceptions and usage; inline comments explain design choices.

## Environment and installation

The verified environment is **CPython 3.14.5 on Windows 11** with NumPy 2.5.4,
pandas 3.1.0, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2,
threadpoolctl 3.7.0, pytest 9.1.1 and pdoc 16.0.0. All 30 dependency/build-tool
versions are pinned in `requirements-lock.txt`. Python 3.14 is required.

Run these commands from the repository root:

```powershell
git clone https://github.com/Shanwei0111/MMA3001Project.git
cd MMA3001Project
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.venv\Scripts\Activate.ps1
```

If activation is restricted, use `.venv\Scripts\python.exe` in place of `python`.
On Linux/macOS, the equivalent interpreter is `.venv/bin/python`; the recorded
benchmark and exact environment verification were performed on Windows.

## Data preparation

The three required raw files are `train_FD001.txt`, `test_FD001.txt` and
`RUL_FD001.txt`. Use one of the following acquisition methods:

```powershell
# Existing NASA folder supplied by the student
python -m rul_project data --data-source "../CMAPSSData"

# Or download the archive linked by NASA's official catalog
python -m rul_project data --download-data
```

The Moodle ZIP already includes the verified FD001 files. Run
`python -m rul_project data` to check them. Public Git history excludes the raw
files; [data/manifest.json](data/manifest.json) verifies exact bytes. Corrupt or
changed files are rejected without overwrite. [data/README.md](data/README.md)
explains the data source, original units and licensing boundary.

## Reproduce the project

```powershell
python -m pytest --junitxml=results/generated/pytest_results.xml
python -m rul_project all
python tools/verify_reproducibility.py
python tools/build_api_documentation.py --check
```

Allow several minutes for the complete workflow. The all command prepares
labels, splits engines, trains all candidates, repeats grouped CV, evaluates
official endpoints, bootstraps uncertainty, plots results and profiles cost.
All fits use the same one-thread budget and seed 42. No windows, smoothing,
feature scaling, engine weights or capped RUL targets are introduced.

Individual stages are available in this dependency order:

```powershell
python -m rul_project prepare-data
python -m rul_project split-engines
python -m rul_project train-linear
python -m rul_project train-forest
python -m rul_project train-hgb
python -m rul_project evaluate-test
python -m rul_project analyse-uncertainty
python -m rul_project plot-results
python -m rul_project benchmark
```

`results/generated/` holds the run's CSVs, figures and profiles. Use
`--output-dir results/generated/trial` for a separate run; `--data-dir` overrides
the raw source directory. The protected reference results cannot be chosen as
an output directory. Earlier command spellings remain accepted for old notes.

## Results and engineering comparison

| Model | Test MAE (cycles) | Test RMSE (cycles) | NASA-style score |
|---|---:|---:|---:|
| Age-only linear | 33.46 | 39.31 | 17239.68 |
| Multi-sensor linear | 26.40 | 31.83 | 9272.61 |
| Random forest | 19.38 | 25.82 | 5804.70 |
| Histogram gradient boosting | 18.93 | 26.18 | 7212.36 |

All models use the same 100 official test endpoints. RF was retained for RMSE
and asymmetric-score performance; HGB is faster and smaller. Paired HGB-minus-RF
error intervals include zero. The [written report](reports/MMA3001_Chua_Shanwei_35393343_Written_Project_Documentation.pdf)
contains the equations, physical input units, complete validation, hardware
benchmarks and interpretation of optimistic errors.

[results/reference/](results/reference/README.md) groups frozen metrics,
predictions, uncertainty, performance and figures into separate subfolders.
The original numbers are preserved. `file_index.json` connects original CSV
names to their new locations. Verification compares both validation and test
predictions, both CV tables and both bootstrap tables; timings are machine-specific.

## Tests, HTML documentation and maintenance

See [tests/README.md](tests/README.md) for coverage and the recorded report.
The organized suite currently contains **61 passing tests**. The refactor was
also checked against all four validation-prediction tables and the official
test/CV/bootstrap results, with maximum numeric difference **0.0**.
Read [results/verification/README.md](results/verification/README.md) for the
fresh-clone check and reproduction tolerance. The report's historical 36-test
evidence is retained separately from the current suite.

API documentation is generated directly from the source:

```powershell
python tools/build_api_documentation.py
python tools/build_api_documentation.py --check
python -m http.server 8000
```

Open `http://localhost:8000/START_HERE.html`, or double-click `START_HERE.html`
in the local clone. API pages include View Source links and parameter/return
contracts. GitHub displays HTML file contents as source; local browser pages
provide the rendered version. The source manifest and CI reject stale docs;
CI also regenerates pages and checks that no differences appear.

## History, licensing and submission

Meaningful changes are recorded on `refactor/organized-submission` before
integration into `main`. Earlier versions remain in Git history and on
`archive/reference-implementation`. [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
maps the learning scripts to the organized modules and discloses the actual
development chronology and AI-assisted repository work.

[LICENSE](LICENSE) covers original code and documentation. NASA data and
third-party libraries retain separate terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [data/README.md](data/README.md).

Final review link: **https://github.com/Shanwei0111/MMA3001Project**

After committing and pushing, create the required zipped clone:

```powershell
python tools/create_submission_zip.py --destination ../MMA3001_Chua_Shanwei_35393343_Repository_Clone.zip --data-source data/raw
```

The utility clones published remote main into an empty temporary directory,
adds verified local FD001 data and zips it with Git history. Submit this ZIP
through Moodle; submit the written report separately. Use `--overwrite` only
when intentionally replacing an older ZIP at the same path.
