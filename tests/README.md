# Test functions and test reports

The current organized suite contains **61 cases**. The original 36 tests and
eight earlier repository checks are supplemented by 17 factory, endpoint,
paired-bootstrap and explicit-path checks.

Open reports/pytest_report.html in a browser for the recorded readable report.
reports/pytest_results.xml is the original machine-readable pytest output for
that run. Run the current source tests with:

```powershell
python -m pytest --junitxml=results/generated/pytest_results.xml
python -m doctest src/rul_project/prepare_dataset.py src/rul_project/rul_metrics.py src/rul_project/model_configuration.py
```

| File | What is checked |
|---|---|
| test_data_and_metrics.py | Known RUL labels, input preservation, invalid keys, engine grouping and hand-calculated metrics |
| test_repository_contracts.py | Production test targets, fixed HGB settings, source bytes and safe imports |
| test_workflow_functions.py | All model factories, test-label alignment, paired-bootstrap integrity, path isolation and operation units |

Tests use small fixtures and known answers. Full experimental reproducibility
is checked separately by tools/verify_reproducibility.py after the complete
workflow. All four models' validation predictions, 400 official predictions,
grouped CV and bootstrap results are compared with the original report evidence.
The historical 36-case report remains in results/reference/historical.

Temporary pytest files use ignored results/generated-tests. Cached test metadata
uses results/generated/pytest-cache. Stored reports are evidence snapshots; new
runs do not silently overwrite tests/reports. Update the recorded report after
reviewing a code change and retain its source fingerprint record.
