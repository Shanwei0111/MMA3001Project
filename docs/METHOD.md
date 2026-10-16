# Engineering and numerical specification

## Scope and engineering use

Estimate remaining useful life (RUL), measured in operational cycles, from a
single simulated turbofan's observed sensor history. The supported population is
NASA C-MAPSS FD001: one operating condition and high-pressure compressor (HPC)
degradation. The result can support an illustrative maintenance review. It is not
an airworthiness decision, a validated real-engine forecast, or a conversion to
flight hours. Each simulated trajectory represents a different initial engine
condition and degradation history. Actual latent wear and failure thresholds are
not observed by the regressor.

The project deliberately excludes FD002-FD004, multiple fault modes, operating
regime normalization, online learning and deep recurrent networks. Six physically
meaningful channels are fixed before fitting. No feature is selected using the
official test targets. No physical conservation residual can be calculated from
these selected outputs alone: mass flow, full thermodynamic state and boundary
conditions would be needed. Physical credibility therefore concerns units,
causality, regime limits, degradation sensitivity and honest failure analysis.

## Input/output contract

The source is a headerless, whitespace-delimited TXT file with 26 columns, in this
exact order: engine ID, cycle, three operational settings, sensors 1 through 21.
Trailing whitespace is allowed. IDs and cycles are positive integers. Each
engine's cycles must be consecutive, with one row per engine/cycle. A prediction
requires one engine and at least 30 observed consecutive rows. A trailing history
may begin after cycle 1, but the original absolute cycle numbers must be retained.

| Field | Meaning | Unit | Accepted domain / use |
|---|---|---|---|
| engine_id | Trajectory identifier | none | Positive integer, metadata only |
| cycle = t | Observed operational age | cycles | Positive integer, predictor |
| setting_1..3 | Operating settings | Undeclared in supplied README | Finite, parsed metadata, not predictors |
| s2 = T24 | LPC outlet total temperature | degrees Rankine | Positive finite, predictor |
| s3 = T30 | HPC outlet total temperature | degrees Rankine | Positive finite, predictor |
| s4 = T50 | LPT outlet total temperature | degrees Rankine | Positive finite, predictor |
| s7 = P30 | HPC outlet total pressure | psia | Positive finite, predictor |
| s9 = Nc | Physical core speed | rpm | Positive finite, predictor |
| s11 = Ps30 | HPC outlet static pressure | psia | Positive finite, predictor |
| Other 15 sensors | Unused source channels | Native source units | Finite, parsed to validate schema |
| raw_rul_prediction | Unconstrained regression output | cycles | Finite real number, retained for audit |
| predicted_rul_cycles | max(0, raw prediction) | cycles | Nonnegative point estimate |
| input_status, flags | Empirical support and clipping warnings | categorical/list | No calibrated confidence interpretation |
| model_version | Configuration SHA-256 prefix | text | Traceability, not a model-file checksum |

Nonfinite values, missing columns, nonnumeric data, duplicate rows, gaps,
nonpositive primary measurements and short histories raise errors. The software
does not silently impute, interpolate or discard an engine. Sensor values outside
the observed training envelope generate soft warnings; these limits are empirical
and are not physical validity limits. A caller must declare FD001. The software
cannot reliably infer an unknown operating regime from the opaque settings.

## Labels and causal features

For a verified run-to-failure training engine i, RUL(i,t) = T_i - t. Labels are
uncapped. A commonly used 125-cycle cap would change the estimand and is not used.
For official test engines, the final observed cycle is censored; the separate
RUL_FD001.txt value supplies endpoint truth. Never label a test endpoint as zero
using its last observed cycle.

For each chosen channel s and trailing window length W:

    mean_s(t) = sum(s[t-W+1:t+1]) / W
    q_j = j - (W-1)/2, j = 0,...,W-1
    slope_s(t) = sum(q_j * s[t-W+1+j]) / sum(q_j**2)
    x(t) = [t, six current values, six means, six slopes]

There are 19 predictors. Slope units are sensor-unit/cycle. Centered local
positions reduce cancellation compared with fitting a line to large absolute
cycle numbers. No centered window or future observation is allowed. All W=10,
20,30 candidates use endpoints at cycle >=30 to make validation support equal.

Training takes every fifth supported cycle and the last cycle of each engine.
Every model uses the same rule. Per-row weights are proportional to 1/n_i, then
normalized to mean one, giving each engine equal total fitting weight. This
reduces the dominance of long trajectories. A full-stride ablation quantifies
the effect of the computational shortcut.

## Mathematical models

Ridge standardizes each feature with training-partition mean and standard
deviation. The scaler is unweighted. It solves

    min_(b,beta) sum_j w_j (y_j - b - z_j^T beta)^2 + alpha ||beta||_2^2

with an unpenalized intercept and an SVD solver. Scaling and regularization make
the fit more stable for strongly correlated current/mean sensor values. Alpha
is selected from 0.1,1,10,100. There is no iterative convergence tolerance for
this direct solver. The age-only baseline uses the same procedure with age as
its sole predictor, including alpha tuning.

The random forest averages B=200 bootstrapped regression trees. Each split
reduces weighted squared error; maximum depth is 8 or 12, minimum leaf count is
5 or 10, and the candidate feature fraction is 0.7 or 1.0. Each tree's prediction
is a leaf weighted mean. The ensemble reduces variance and represents nonlinear
sensor/age interactions, but cannot reliably extrapolate beyond learned leaf
responses. Tree growth stops at configured depth/leaf constraints, not at a PDE
residual or a gradient convergence tolerance. One thread and seed 3001 are used.

## Selection, validation and uncertainty

1. Freeze configuration and SHA-256 manifest. Audit file shape/count/finite values.
   The preliminary data audit inspected test label summaries, but no model was
   selected from them. The model-selection routine only checksums test files.
2. Randomly split the 100 training engines into 80 development and 20 holdout
   engines, using a saved seed and explicit ID lists.
3. Assign each engine a single reproducible pseudo-endpoint, uniformly selected
   from cycle 30 through T_i-1. Only its causal history supplies features.
   The choice uses known training lifetime to define a retrospective scenario;
   its horizon distribution need not match the official test truncations.
4. Five-fold CV splits the 80 development engines, with 64 fit and 16 validation
   engines per fold. Pool the 80 validation predictions and minimize RMSE.
   Search 12 sensor-ridge and 24 forest configurations, plus four age baselines.
   Never split individual rows randomly across folds.
5. Fit each CV winner on the 80 development engines. Select the forest only if
   the 20-engine holdout RMSE is at least 5% below ridge AND its NASA score is no
   worse. Otherwise choose ridge. Persist selection.json before official test
   evaluation. This holdout is a model-selection set, not an unbiased final test.
6. Freeze the selection. Development-fold tree-count, feature/stride, seed and
   measurement-noise studies describe behavior without changing the selection.
7. Refit all three frozen configurations on all 100 training engines. Evaluate
   exactly one endpoint per each of 100 official test engines. Report every
   predeclared comparator; do not switch the selected model using test results.
8. Paired bootstrap resamples engines (2,000 replicates, fixed seed). Report
   percentile 95% intervals for MAE/RMSE and forest-minus-ridge differences.
   These are uncertainty intervals for dataset-level metrics conditional on the
   fitted models, not per-engine prediction intervals or total training uncertainty.

Errors use e = prediction - truth. RMSE = sqrt(mean(e^2)); MAE = mean(abs(e));
bias = mean(e). Positive error overestimates life. The NASA-style score is

    S = sum(expm1(-e/13) if e < 0 else expm1(e/10))

The score is dimensionless and lower is better. `expm1` avoids cancellation near
zero and the implementation reports overflow explicitly. Its mean is also
provided to compare sets with different engine counts. The supplied local paper
prints contradictory constants on p.7; the NASA NTRS version supports the stated
early=13, late=10 convention. The explicit formula and a penalty-direction test
remove ambiguity. Compare published scores only when target capping, endpoints,
sample counts and preprocessing conventions match.

Engineering targets proposed before evaluation are MAE <=20 and RMSE <=25 cycles,
not certification thresholds. Report honestly if they are missed. Report errors
by true-RUL band (<=30,31-80,>80), signed residuals, outliers, clipping and
out-of-envelope warnings. Noise sensitivity uses 1% and 5% of development sensor
SD, and a +1 SD T30 bias. SD is an empirical scale, not a calibrated sensor error.

## Performance and optimization

Profile the same machine, single-thread, warm in-memory estimator. Report 30
timing repetitions and median/IQR for one prediction, a 100-engine batch, and
history-to-output processing. Report three refit timings, compressed model size,
node-array storage, and process working set (including libraries/data). File
loading is excluded from warm prediction latency. Hardware timing is not portable.

The explicit reference feature loop and vectorized sliding windows compute the
same formula. Compare maximum absolute difference and timings on the full
training set. Vectorization changes execution overhead, not the mathematical
model. Naive window work is O(N*6*W); a future rolling recurrence could reach
O(N*6), with additional floating-point drift considerations.

For ridge with d predictors, a dot product plus intercept uses about 2d FLOPs;
separate standardization adds 2d. A hypothetical fused implementation has
approximate cold bytes 8(2d+2), giving 2d/[8(2d+2)] FLOP/byte. This is a scoped
arithmetic estimate, not a hardware counter and not a claim that sklearn fuses
the operations. It excludes validation and Python/library overhead.

Forest traversal is primarily comparisons and irregular memory access. Count
actual non-leaf nodes visited on the 100 test endpoints; comparisons are not
FLOPs. B predictions require B-1 additions and one division (~B FLOPs). Report
actual total nodes and array bytes. Approximate traversal bytes assume one node
record per visited node and ignore cache reuse. Training is roughly
O(B*m*N*log(N)) for balanced trees with m candidate features; implementation,
sorting and unbalanced trees alter this estimate. Memory/branch overhead dominates
these small prediction kernels. No manufactured GPU or peak-FLOP claim is made.

## Sources

- NASA catalog: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- Saxena et al., Damage Propagation Modeling for Aircraft Engine Run-to-Failure
  Simulation (2008): https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf
- Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- Random forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- Leakage: https://scikit-learn.org/stable/common_pitfalls.html
- pdoc: https://pdoc.dev/docs/pdoc.html

Accessed during preparation in October 2026. Supplied module briefs govern the
submission format; this document is implementation detail supporting the report.
