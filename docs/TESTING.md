# Tests and reproducibility evidence

Run from the installed repository root:

```powershell
python -m pytest --junitxml=outputs/pytest_results.xml
python -m doctest src/rul_project/rul_utils.py
python -m rul_project all
python scripts/verify_reproduction.py --output-dir outputs
python scripts/build_docs.py --check
```

The original Assignment XML recorded 36 passing cases. The packaged suite removes
a duplicate `split_by_engine` definition from the test module, ensuring its tests
exercise the production function. Eight additional checks bring the current total
to **44 cases**:

- 16 RUL-label cases: known values, row/index preservation, duplicates, missing
  columns and zero/negative/fractional/nonfinite keys.
- 10 grouped-split cases: complete engine histories, disjoint groups, deterministic
  sampling and invalid split sizes or IDs.
- 10 metric cases: zero error, hand-calculated MAE/RMSE/bias, asymmetric 13/10
  penalty, corrupt shapes/values and exponential overflow.
- 8 repository cases: production test-function identity, frozen HGB parameters,
  source fingerprints, safe local acquisition, side-effect-free stage imports
  and unsupported estimator memory inspection.

The test configuration places temporary files inside ignored `outputs/pytest-tmp`.
The API examples in `rul_utils.py` are also executable doctests. The current test
report is stored separately in `evidence/verification/`; the historical XML in
`evidence/reported/` is retained for traceability to the written report.

`verify_reproduction.py` compares all 400 model/engine predictions, extended test
metrics, both grouped CV tables and both bootstrap tables with the frozen report
snapshot. It checks IDs before numbers and uses atol=1e-8, rtol=1e-10. Timings,
profiles and private-array/serialization measurements are excluded from numerical
equality because they depend on hardware, library implementation and run load.

This suite targets software correctness and recorded numerical reproduction.
Sensor-noise, missing-channel, alternate-regime and real-engine performance are
not established by these tests; those limitations remain in the report.
