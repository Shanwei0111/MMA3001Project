# Turbofan Remaining Useful Life - NASA C-MAPSS FD001

**Chua Shanwei · 35393343 · MMA3001 Numerical Methods and Machine Learning**

Estimate cycles to simulated engine failure from operating age and six
gas-path sensors. The current submission compares **age-only linear regression,
multi-sensor linear regression, random forest (RF), and histogram gradient
boosting (HGB)** using the student's current `Assignment` experiments.

## Results and report

| Model | Test MAE (cycles) | Test RMSE (cycles) | NASA-style score |
|---|---:|---:|---:|
| Age-only linear | 33.46 | 39.31 | 17239.68 |
| Multi-sensor linear | 26.40 | 31.83 | 9272.61 |
| Random forest | 19.38 | 25.82 | 5804.70 |
| Histogram gradient boosting | 18.93 | 26.18 | 7212.36 |

These values use one final observed endpoint per official test engine (100
engines). RF was retained for RMSE and asymmetric-score performance. HGB is
faster and smaller; paired HGB-minus-RF error intervals include zero. The
report explains test reuse, positive bias and the uncertainty limitations.

- [Written project documentation (PDF)](reports/MMA3001_Chua_Shanwei_35393343_Written_Project_Documentation.pdf)
- [Editable report text](reports/MMA3001_Chua_Shanwei_35393343_Written_Project_Documentation.md)
- [HTML API documentation](docs/html/index.html): open after cloning, or serve locally below.
- [Original report evidence](evidence/reported/) and [source provenance](evidence/source_provenance.json).
- [Testing guide and reports](docs/TESTING.md), [method and I/O](docs/METHOD.md),
  [development history](docs/DEVELOPMENT.md), [AI assistance record](docs/AI_USE.md).

## Environment and installation

The recorded experiment uses **CPython 3.14.5**, NumPy 2.5.4, pandas 3.1.0,
SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2 and threadpoolctl 3.7.0.
Python 3.14 is required by this pinned environment. README commands run from
the repository root. Use an empty output directory for an independent rerun.

```powershell
git clone https://github.com/Shanwei0111/MMA3001Project.git
cd MMA3001Project
python --version
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
```

In the commands below, `python` means this virtual environment's interpreter;
activate it with `.venv\Scripts\Activate.ps1`, or use its full path. On
Linux/macOS use `.venv/bin/python` and `source .venv/bin/activate`.

## Obtain the exact data

Raw NASA files are excluded from public Git history and are not licensed as
project code. Follow [data attribution and access](data/README.md). Three
files are required: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`.
Acquire them using **one** of these options:

```powershell
# Copy the student's existing NASA files, validating their SHA-256 hashes.
python -m rul_project data --data-source "../CMAPSSData"

# Or explicitly download the NASA archive linked by its official catalog.
python -m rul_project data --download-data
```

Alternatively, put unmodified files in `data/raw/` and run `python -m
rul_project data`. [data/manifest.json](data/manifest.json) records source
fingerprints and byte counts. Mismatched files are rejected without overwrite.

## Reproduce the workflow

```powershell
python -m pytest --junitxml=outputs/pytest_results.xml
python -m rul_project all
python scripts/verify_reproduction.py --output-dir outputs
python scripts/build_docs.py
python scripts/build_docs.py --check
```

The full run includes development fitting, the RF depth study, five grouped
folds for each ensemble, official testing, timing/profiling, plots and paired
bootstrap. Allow several minutes on a laptop. Models are fitted on one
computation thread, with random seed 42. Predictions/metrics should reproduce
within the verification tolerance; timings and platform-dependent storage
inspection may change. No feature scaling, smoothing, windows, RUL cap or
engine weighting is introduced during packaging.

`outputs/` contains rerun results and is ignored by Git. `evidence/reported/`
is the frozen evidence behind the written report. Never replace old benchmark
times with a new run without recording the changed hardware and procedure.
The CLI rejects using the frozen evidence directory as its output destination.

Individual learning stages can also be run in this dependency order:

```powershell
python -m rul_project preparedata
python -m rul_project splitdata
python -m rul_project linear_models
python -m rul_project randomforest
python -m rul_project histgradientboosting
python -m rul_project testevaluation
python -m rul_project plotbaseline
python -m rul_project performance
python -m rul_project uncertainty
```

The eight original learning scripts map to the first eight stage modules in
`src/rul_project/steps/`; `uncertainty` reproduces the later HGB report analysis.
All stages have main guards. Importing them for pdoc or tests performs no I/O,
training or plotting. `python -m rul_project --help` lists path overrides:

```powershell
python -m rul_project all --data-dir data/raw --output-dir outputs/independent
```

## Repository layout

```text
src/rul_project/          Shared API, CLI, validated data access and learning stages
tests/                   Production-function tests and workflow contract checks
data/                    Dataset attribution and source SHA-256 manifest
docs/html/               Generated pdoc API pages (committed and checked in CI)
docs/                    Method, testing, development and AI-use notes
evidence/reported/       Original metrics, predictions, plots and test evidence
evidence/verification/   Fresh-clone verification and current test reports
reports/                 Written report PDF and editable text
scripts/                 Documentation, reproduction and zipped-clone utilities
.github/workflows/       Tests and documentation synchronization checks
outputs/                 Ignored regenerated results
```

## Read the API documentation

GitHub displays HTML source files as source code. To read the rendered pages:

```powershell
python -m http.server 8000 --directory docs/html
```

Open `http://localhost:8000/`. Rebuild after changing any Python API:
`python scripts/build_docs.py`. The source manifest includes all package and
support scripts; `--check` verifies fingerprints and committed HTML. Docstrings
use Google style with inputs, returns, exceptions and usage. See
[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).

## Version history and submission

The earlier reference implementation used different features and models. It is
preserved on `archive/reference-implementation`, while `main` contains this
current four-model workflow. [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) explains
the transition and the actual packaging commits; no past development dates or
student actions have been invented.

Final accessible repository: https://github.com/Shanwei0111/MMA3001Project

To produce the required zipped clone after committing and pushing:

```powershell
python scripts/package_submission.py --destination ../MMA3001_Chua_Shanwei_35393343_Repository_Clone.zip --data-source data/raw
```

This clones remote `main` into a fresh directory, verifies and adds the locally
supplied raw FD001 files, and zips the clone including its Git metadata. No
virtual environment, caches or generated rerun output are included. Submit
that ZIP through Moodle and submit the written PDF separately as required.

## Licensing and acknowledgements

[MIT](LICENSE) covers original project code and documentation. NASA data and
third-party packages have their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and [data/README.md](data/README.md). Data access is public; the inspected NASA
catalog does not specify a dataset license. AI-assisted implementation and
drafting are disclosed in the report and [docs/AI_USE.md](docs/AI_USE.md).
