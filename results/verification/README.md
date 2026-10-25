# Reproducibility verification

The current check results are in reproduction_check.json and
organized_repository_check.json. They record the tested source hashes, pytest
outcome, environment and deterministic comparison. Previous packaging checks
are explicitly separated in previous_package/ for history.

Reproduce from a fresh clone and a Python 3.14 virtual environment:

```powershell
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m rul_project data --data-source ../CMAPSSData
python -m pytest --junitxml=results/generated/pytest_results.xml
python -m rul_project all
python tools/verify_reproducibility.py
python tools/build_api_documentation.py --check
```

The comparison aligns engine/model keys before checking numeric columns, using
atol=1e-8 and rtol=1e-10. It includes four complete validation-prediction tables,
all official test predictions, extended metrics, both CVs and both bootstrap
tables. Hardware-sensitive timings, profiles and serialization estimates are not
required to equal old benchmark measurements. The frozen results remain intact.
