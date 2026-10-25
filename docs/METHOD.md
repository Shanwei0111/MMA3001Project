# Method and I/O contract

FD001 is restricted to one operating regime and one HPC degradation mode.
RUL labels are `max(cycle within training engine) - cycle`, uncapped. The
engine-level development split uses pandas sampling of 80 engines with seed
42, retaining 16,340 training and 4,291 validation records with no shared IDs.
Five-fold GroupKFold operates within those 80 engines (64/16 per fold).

Models use cycle alone or the ordered features cycle, sensor_2, sensor_3,
sensor_4, sensor_7, sensor_9, sensor_11. Temperatures (T24/T30/T50) are Rankine,
pressures (P30/Ps30) psia, core speed (Nc) rpm and age/RUL cycles. Full observed
ranges and limitations are given in report Section 2. IDs are never predictors.

LinearRegression uses its fitted intercept. RF has 100 trees, depth 8, minimum
leaf 10, default full feature fraction, bootstrap enabled, seed 42 and n_jobs=1.
HGB has squared error, learning rate 0.05, 200 iterations, 15 leaves, depth 4,
minimum leaf 20, L2 1.0, default max_bins=255, no internal early stopping and
seed 42. All models are refitted on the complete 100-engine training set for
official evaluation. NASA labels apply only to the final observed test cycles.

Raw predictions are retained and delivered predictions are clipped at zero.
Signed error is prediction minus truth. The score is the sum of exp(-e/13)-1
for negative e and exp(e/10)-1 for nonnegative e. Metrics are unrounded in CSV.
Bootstrap samples 100 endpoint IDs 2,000 times with seed 42 and keeps model
comparisons paired. It does not produce individual-engine prediction intervals.

Raw-file hash checks reject corrupt or changed source files. Shared functions
reject invalid keys, duplicate records and invalid metric arrays. Arbitrary
units, regimes and missing-sensor input remain outside the validated pipeline.
HGB can internally handle NaNs, but no uniform imputation policy is implemented.

API parameter/return/exception contracts and examples are generated from the
same source in docs/html/. The report describes test reuse and development-
informed model comparison. Packaging does not reset the test's independence.
