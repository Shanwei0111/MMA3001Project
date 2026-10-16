# Turbofan remaining useful life - NASA C-MAPSS FD001

**Chua Shanwei, 35393343 - MMA3001 Numerical Methods and Machine Learning**

Estimate remaining operational cycles from age and six causal sensor histories.
This project compares a random forest, standardized ridge regression and an
age-only baseline using engine-level validation, uncertainty analysis and
measured computational cost. The scope is simulated FD001 only.

## Measured result

The frozen internal holdout rule selected a 200-tree random forest, depth 8,
minimum leaf 10, feature fraction 0.7 and a 30-cycle window. Training uses every
fifth eligible cycle and equal-engine weights. Official test results (100 engines):

| Model | MAE (cycles) | RMSE (cycles) | NASA score |
|---|---:|---:|---:|
| Age-only | 28.83 | 34.89 | 8770.3 |
| Sensor ridge | 18.26 | 22.29 | 1670.4 |
| Random forest (selected) | 14.69 | 20.04 | 2174.6 |

The forest improves average error but has a **worse official NASA score than
ridge**, revealing optimistic tail errors. Selection remains frozen. Its RMSE
bootstrap interval is [16.33, 23.56] cycles; the paired forest-minus-ridge interval
[-4.87, 0.29] includes zero. These are metric uncertainty intervals, not individual
prediction bounds. No real-engine maintenance suitability is claimed.

## Deliverables and evidence

- [Technical report](output/pdf/Jet_Engine_RUL_Report.pdf): nine main pages plus references, all seven required sections.
- [Editable presentation](presentation/Jet_Engine_RUL_Presentation.pptx): seven slides and 300 seconds of suggested narration.
- [Speaker notes](presentation/SPEAKER_NOTES.md) and [technical defense guide](presentation/DEFENSE.md).
- [Generated HTML source documentation](docs/html/jet_rul.html): download/clone and open locally, or serve as described below.
- [Method and I/O specification](docs/METHOD.md), [AI disclosure](docs/AI_USE.md), and [brief coverage matrix](docs/REQUIREMENTS.md).
- [Test report](artifacts/test-results.xml), saved [model selection](artifacts/selection.json), [predictions](artifacts/test_predictions.csv) and [performance evidence](artifacts/performance.json).

## Quick start (Python 3.12)

The tested environment uses Python 3.12 on Windows; exact installed Python
packages are pinned in requirements-lock.txt. A Linux CI job runs the synthetic
tests without NASA data. Internet access is needed to install dependencies.

```powershell
git clone https://github.com/Shanwei0111/MMA3001Project.git
cd MMA3001Project
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
```

On Linux/macOS, activate with `source .venv/bin/activate` instead. If PowerShell
activation is restricted, use `.venv\Scripts\python.exe` in place of `python`.

Obtain the NASA archive linked by the [official catalog](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
Extract `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` into `data/raw/`.
Do not edit the source files. The workflow checks them against data/manifest.json.
See [data provenance and terms](data/README.md). A local submission archive also
contains the user's supplied FD001 files, so this extraction step can be skipped.

```powershell
python -m pytest --junitxml=artifacts/test-results.xml
python -m jet_rul reproduce --data-dir data/raw
python -m jet_rul predict --data-dir data/raw --engine 1
python -m pdoc --docformat numpy -o docs/html jet_rul
python scripts/build_report.py
```

The complete experiment searches 40 configurations across five folds, then
performs diagnostic studies, refits, evaluates, profiles and draws figures.
Allow several minutes on a modern CPU; timing depends on machine load. It writes
to artifacts/ and replaces the corresponding saved results. For an independent
rerun, supply `--output-dir artifacts/rerun`; report building uses artifacts/ by
default, so copy chosen rerun evidence there only deliberately.

Individual stages are available:

The committed code was checked from a clean clone: all 37 tests passed and
refitting the three frozen configurations reproduced all 100 endpoint predictions
with maximum absolute difference 0.0 in the same environment. This verification
repeated final fitting/evaluation, not the whole tuning search. See
artifacts/reproduction_check.json. Repeat this check with:

```powershell
python scripts/verify_reproduction.py --data-dir data/raw
```

For individual workflow stages:

```powershell
python -m jet_rul develop --data-dir data/raw
python -m jet_rul evaluate --data-dir data/raw
python -m jet_rul profile --data-dir data/raw
python -m jet_rul figures --data-dir data/raw
```

`develop` only parses training observations. It saves partitions and model choice
before `evaluate` reads official labels. Keep configs/fd001.json unchanged between
stages. Test evaluation is reproducible, but must not be used to tune subsequent
models while claiming it remains an untouched test. Model files are generated
locally under artifacts/models/ and excluded from Git. Load only trusted joblib
files; they use Python pickle.

## Input and prediction example

Source rows have 26 fields: ID, cycle, three settings, 21 sensors. The predictor
uses age plus s2/s3/s4/s7/s9/s11 current values, trailing means and slopes. Units
are degrees Rankine, psia and rpm. Missing data, gaps, duplicates, impossible
primary sensor values and fewer than 30 observations raise errors. Values beyond
empirical sensor ranges return warnings. See the full [I/O contract](docs/METHOD.md).

```python
from jet_rul.data import read_cmapss
from jet_rul.models import load_model

observations = read_cmapss('data/raw/test_FD001.txt')
model = load_model('artifacts/models/forest.joblib')
result = model.predict(observations.loc[observations.engine_id == 1])
print(result['predicted_rul_cycles'], result['flags'])
```

The official target is uncapped RUL. Scoring uses error = prediction - truth,
with exponential denominators 13 for early and 10 for late predictions. Never
infer test truth from the last observed cycle or compare capped-target scores
without stating the changed convention.

## Repository layout

```text
src/jet_rul/       parser, causal features, models, metrics, experiments, profiling
tests/            analytic, boundary, leakage and synthetic workflow tests
configs/          fixed search and selection settings
data/             provenance, SHA-256 manifest, local untracked raw files
artifacts/        numerical evidence, figures, profiler and JUnit test report
docs/             engineering method, disclosure, requirement mapping, pdoc HTML
notebooks/        optional training-only exploration
scripts/          report/presentation builders and reproducibility verification
output/pdf/       assessed report and editable Markdown companion
presentation/     editable slides, narration, defense guide and source content
.github/          automatic test/documentation checks
```

## Documentation and artifacts

Open docs/html/jet_rul.html in a browser after cloning. GitHub's file viewer does
not execute raw HTML. Alternatively run `python -m http.server 8000 --directory
docs/html` and visit http://localhost:8000/jet_rul.html. Regenerate HTML after
source changes to keep docstrings synchronized.

The report and scientific figures regenerate with ordinary pinned Python
packages. The editable PowerPoint is included directly. Its optional JavaScript
builder uses the Codex bundled `@oai/artifact-tool` runtime and requires that
runtime plus ARTIFACT_SKILL_DIR, RUNTIME_PYTHON and RUNTIME_NODE_MODULES; it is not a public npm
dependency. The deck's content and narration also exist as plain JSON/Markdown,
so a reader can edit the slides without that authoring runtime. All model results
and validation evidence reproduce independently of presentation tooling.

## Development and Git history

The repository preserves the original README commit and adds actual implementation,
experiment and documentation stages. Commits are not backdated. For personal
changes, use a branch such as `git switch -c improve-input-checks`, run pytest,
regenerate docs, and commit a single meaningful change with a descriptive prefix
(`feat:`, `fix:`, `test:`, `docs:`, `experiment:`). Do not commit environments,
downloaded data, temporary files or credentials. Keep main functional. The
GitHub Actions workflow builds tests/docs and uploads their evidence.

## Student completion and disclosure

Substantial code and writing were generated with Codex assistance. The student
must review and understand them, run the work personally, add a genuine reflection
to [AI_USE.md](docs/AI_USE.md) and the report, and rehearse the oral defense without
AI. Confirm teaching-team approval and the actual deadline in the unit overview.
The full brief says Week 5 for topic selection, while the key-points document says
Week 6; neither supplies an exact final submission date. Upload the final report,
accessible GitHub link and zipped clone via Moodle according to the unit instructions.

Original code/documentation use the [MIT license](LICENSE). NASA data and third-party
materials retain their own terms.
