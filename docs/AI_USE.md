# AI use and student review record

Author/submitting student: Chua Shanwei, 35393343.

## Actual assistance in this project

OpenAI Codex assisted with interpreting the uploaded brief, selecting a tractable
FD001 scope, designing experiments, generating the Python implementation and
pytest suite, running experiments, preparing documentation, report and slides,
and organizing Git history. The student selected jet-engine lifespan prediction,
provided the C-MAPSS files and brief, and supplied submission identity/repository.
Implementation choices and most written material were generated with substantial
AI assistance. They must not be described as entirely student-written.

| Stage | Prompt/request summary (paraphrase) | AI contribution | Validation/evidence | Student review |
|---|---|---|---|---|
| Scope | Complete the project using uploaded NASA C-MAPSS and the unit brief | FD001 scope, causal features, baseline and forest comparison | Brief coverage matrix and source documentation | Pending personal review |
| Code | Implement the selected project reproducibly | Loader, labels, feature transforms, estimators, CLI | Analytic feature tests, leakage tests, malformed-input tests, synthetic workflow | Pending line-by-line understanding |
| Experiments | Compare methods with validation and profiling | Grouped engine partitions, saved selection, benchmark and sensitivity analysis | Saved CSV/JSON outputs, seed/config/data hashes, reproducibility check | Pending interpretation |
| Report/deck | Produce the assessed deliverables | Report wording, plots, slide notes and defense guide | Numbers derived from saved evidence, layout inspection | Pending editing and rehearsal |
| Repository | Prepare GitHub submission | Packaging, CI, HTML documentation and meaningful commits | Test report and repository history | Pending final submission decision |

These rows summarize real activities. They are not fabricated verbatim prompt
transcripts. Keep the original conversation as the full prompt record if the
teaching team requests it. Do not invent tool model-version strings that the
interface does not expose.

## Technical reflections supported by the work

- A model can appear highly accurate if adjacent cycles from the same engine
  occur in training and validation. Engine-level partitions prevent that error.
- The final observed test cycle is not failure. External endpoint labels must be
  aligned with engine IDs; training-label logic must never be reused blindly.
- The supplied paper's scoring constants conflict with its prose. The explicit
  formula, authoritative source and numeric unit test are better evidence than
  trusting an AI-generated formula.
- Strongly correlated current and averaged signals justify standardization and
  ridge regularization. A direct SVD solve needs different numerical reasoning
  from an iterative PDE solver or neural network.
- More complex models need measured evidence. A forest may fail a predeclared
  selection rule even if it is the original proposed primary model.
- Automatic code generation still required corrections: invalid-value tests
  initially failed while constructing integer-typed fixtures under pandas 3.
  Converting those deliberately corrupt fixtures to float allowed the intended
  validation checks to execute. No production-data corruption was introduced.

## Student-owned reflection to complete before submission

The following questions require the student's own answers. They are intentionally
unanswered, because generated first-person claims would misrepresent learning.

1. Which two implementation decisions can you now explain without assistance?
2. Which AI suggestion did you verify or change, and what evidence justified it?
3. Run the workflow yourself. Record the command, date, environment and result.
4. Explain one failure case using an actual engine ID from test_predictions.csv.
5. What would you implement differently with another week, and why?

Append genuine answers and update the report reflection before assessment. Keep
the disclosure of substantial AI assistance even after personally revising code.
