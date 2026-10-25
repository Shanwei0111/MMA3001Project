# Development history and source organization

The current main branch contains the student's four-model FD001 comparison.
Earlier windowed/ridge-reference results are preserved on
archive/reference-implementation; they are not the source of the current report.

The first local Assignment scripts were developed without their own Git history.
Subsequent packaging and refactoring commits record work actually performed at
that point. No earlier dates or independent student actions have been invented.

## Learning-script map

| Original local learning file | Organized source and command |
|---|---|
| step1_preparedata.py | prepare_dataset.py; prepare-data |
| step2_splitdata.py | prepare_dataset.py; split-engines |
| step3_linear_models.py | train_models.py; train-linear |
| step4_randomforest.py | train_models.py; train-forest |
| step5_histgradientboosting.py | train_models.py; train-hgb |
| step6_testevaluation.py | evaluate_test_set.py and analyse_uncertainty.py |
| step7_performance.py | benchmark_performance.py; benchmark |
| step8_plotbaseline.py | plot_model_results.py; plot-results |
| rul_utils.py | prepare_dataset.py, rul_metrics.py and model_configuration.py |

The refactor extracts reusable functions, centralizes estimator settings, passes
paths explicitly and gives every module a purpose-specific name. Report equations,
training targets, sensor order, seeds, model settings and validation populations
are preserved. Full prediction/CV/bootstrap comparison checks the refactor.

## Actual branch and commit practice

The previous current-workflow packaging used submission/current-fd001-workflow.
This organization pass uses refactor/organized-submission and retains scoped
source, documentation and verification commits. Main is updated only after
tests, documentation checks and reproduction pass. Future experiments should
use a focused branch and record changed settings and results in a descriptive
commit; official test performance is not a replacement for grouped development
validation. Examples: refactor: centralize model configuration; test: reject
misaligned endpoint labels; docs: synchronize public API contracts.

## AI assistance

ChatGPT/Codex assisted with substantial initial code, explanations, debugging,
test/validation suggestions and report drafting. The student selected the
application, ran experiments, supplied outputs and questioned interpretations.
This repository organization, comments, docstrings, documentation and verification
were also AI-assisted. Source comparisons and executable tests check those changes.
The student remains responsible for reviewing and explaining the submitted work.

The report's Section 7 provides the personal reflection. Neither the Git history
nor this guide claims that AI only provided syntax references or that all
implementation was written independently by the student.
