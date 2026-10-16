# Three-minute technical defense practice

Use these as technical explanations to understand, not a script to recite without
understanding. The examiner may ask only two or three questions. Answer directly,
then point to a formula, test or actual result. AI cannot answer during the defense.

## What system and assumptions do you represent?

A simulated turbofan undergoing HPC degradation in FD001's single operating
condition. The assumption is that unseen FD001 histories follow the training
population's relationships between age, gas-path measurements and remaining
cycles. I do not claim to reconstruct unique physical damage, model another fault
mode, or generalize to a real fleet. The source simulator is physics-based; the
regressor itself is empirical.

## Why does test truth come from another file?

Training trajectories reach failure, so RUL is maximum training cycle minus
current cycle. Test trajectories are censored. Their last observed cycle still
has remaining life. NASA supplies that endpoint label separately, ordered by
engine ID. Treating the last observation as failure would create a false zero.

## How do you prevent leakage?

All rows from one engine stay in one fitting or validation partition. Features
use only a trailing window. The scaler learns from fit rows only. Test labels
enter after the selection file freezes. Tests mutate future observations and
ensure earlier features stay unchanged. A synthetic workflow uses deliberately
unparseable official files to prove development never parses them. Preliminary
source audit statistics were inspected and disclosed, but did not select models.

## Why a random forest, and why compare ridge?

The forest represents nonlinear interactions and averages bootstrap trees to
reduce variance. Ridge is a stable low-cost linear alternative for correlated
features and small fleets. The internal holdout rule required a 5% RMSE benefit
without a worse NASA score. The forest achieved 10.6% lower holdout RMSE and a
lower score. This evidence selected it before the official test.

## What are your numerical stability and convergence criteria?

The slope uses centered local time positions. Ridge standardizes its predictors
and adds an L2 penalty, then uses a direct SVD solve instead of explicitly
inverting normal equations. That solver has no iterative convergence tolerance.
Trees stop at depth eight and minimum leaf size ten; two hundred trees average
their predictions. I studied 50-400 trees and three random seeds to characterize
cost and variability. It would be incorrect to invent a mesh-convergence or PDE
residual criterion for this model.

## What errors dominate, and how are they controlled?

Model bias and finite-fleet sampling matter more than floating-point rounding.
The forest test RMSE is 20.04 cycles, whereas vectorized/reference features differ
by at most 1.88e-13. Test uncertainty is quantified by paired engine bootstrap;
noise/bias perturbations investigate input sensitivity. The forest-versus-ridge
RMSE difference interval includes zero. Training uncertainty and real-world
distribution shift are not fully captured.

## Is the forest safer because its RMSE is lower?

No. Its test NASA score is 2174.6 versus ridge's 1670.4, even though its RMSE is
20.04 versus 22.29. The NASA-style score penalizes optimistic life estimates
exponentially. Some tail errors therefore outweigh average improvement. The
holdout score advantage did not generalize. I disclose this limitation and keep
the frozen selection rather than using the test set for another choice.

## Why are the score denominators 13 and 10?

With e=prediction-truth, the stated convention uses exp(-e/13)-1 for early errors
and exp(e/10)-1 for late errors. A +10 error gives 1.718 and a -10 error 1.158.
The supplied local paper prints inconsistent constants, so I checked the NASA
NTRS version and made the formula explicit. The penalty-direction unit test
guards this interpretation. Scores require the same uncapped target convention.

## What does the bootstrap interval mean?

The forest metric RMSE interval [16.33,23.56] describes variation from sampling
different sets of engines from this test sample, conditional on the fitted model.
It is not a range for an individual engine's remaining life, and does not include
retraining on a new fleet. The paired comparison resamples identical engines for
both models, preserving correspondence.

## Did trends and more data always help?

No. On the diagnostic development fold, stride-five current-only forest features
give RMSE 27.83 versus 29.30 with trends. The full-stride trend model takes 12.25 s
versus 2.41 s at stride five and has worse RMSE on that fold. This is limited
one-fold evidence, not a general proof. It motivates a future controlled feature
study while leaving the predeclared selected model unchanged.

## What is the computational bottleneck? How did you optimize it?

The cProfile trace records feature construction overhead. Vectorized rolling
windows compute the same means/slopes in 0.183 s versus the 0.291 s reference
loop, a 1.59x speedup. End-to-end prediction also includes parsing checks and
feature assembly, which dominate a tiny ridge dot product. Forest prediction
visits about 1,503 comparison nodes per engine. Those comparisons and irregular
memory accesses differ from floating-point arithmetic.

## Explain FLOPs and arithmetic intensity for this project.

A 19-feature dot product plus intercept uses about 38 FLOPs. Separate scaling
adds about 38. A hypothetical fused cold calculation has about 320 input/parameter
bytes and 0.119 FLOP/byte for its dot product. This is an analytical estimate,
not a measured hardware counter or an exact description of sklearn execution.
Forest averaging uses about 200 FLOPs, but its traversal uses comparisons; I
report them separately. Serialized forest size is about 1.13 MB, node/value
arrays 3.32 MB, while process working set also includes Python, libraries and data.

## What is most vulnerable to failure, and what mitigates it?

Wrong regimes, broken cycle sequences, missing sensors, misaligned labels and
unfamiliar degradation can invalidate a forecast. Input contracts reject known
structural errors and warn about empirical sensor excursions. The application
cannot infer all unsupported regimes automatically. Source hashes, saved IDs,
causal tests and explicit metadata protect traceability. A future deployment
would need calibrated uncertainty, monitoring and independent field validation.

## How is the solution physically credible?

The channels correspond to gas-path temperatures, pressures and core speed,
all in declared native units. Predictions use causal histories and nonnegative
remaining cycles. Noise, bias and operating-scope limitations are explicit.
However, I do not claim conservation-law validation: the selected channels lack
the full state and boundary information needed for a mass/energy balance. That
limitation distinguishes an empirical prognostic model from a thermodynamic solver.

## What did AI do, and what did you personally learn?

State that Codex substantially assisted design, code, tests and writing. Explain
how tests and benchmark evidence checked it, and describe the concrete pandas
fixture correction or score-formula verification. Then add your own genuine
experience after reviewing and rerunning the project. Do not claim independent
authorship, personal experiments or understanding that you have not yet gained.

## Rehearsal exercise

1. Compute a trailing slope by hand for [2,4,6,8] and explain why it is 2 units/cycle.
2. Locate one engine in artifacts/test_predictions.csv, calculate its signed error
   and explain whether it is optimistic or conservative.
3. Explain each stage from source TXT to the returned JSON in under one minute.
4. Open one pytest test and explain what bug it would catch.
5. Deliver the seven slides using the 300-second timings in SPEAKER_NOTES.md.
