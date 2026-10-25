# Clean-clone verification - 25 October 2026

The current implementation was committed on the submission branch and cloned
with `git clone --no-local`, preventing an accidental shared working directory.
The checked source revision was **b9c30c6b8fe93b0826ea5394c52b805141865018**.
Later changes add the verification evidence and dependency inventory without
changing the checked package code. The full source fingerprints are recorded in
[clean_clone_verification.json](../evidence/verification/clean_clone_verification.json).

A separate CPython 3.14.5 virtual environment was created **without system site
packages**. All pinned dependencies were downloaded and installed into that
environment, followed by an editable installation of the independent clone.
The exact installed dependency versions are in `requirements-lock.txt`.

The following checks passed:

1. All **44 pytest cases**, including production-function identity and import
   safety. [XML](../evidence/verification/pytest_results.xml) and
   [readable HTML](../evidence/verification/pytest_report.html) are retained.
2. Executable examples in the shared utility docstrings.
3. Raw FD001 file fingerprints after copying the student's original files.
4. The complete CLI workflow: label construction, grouped split, both linear
   baselines, RF depth study, five grouped folds for each ensemble, official
   model refitting, figures, runtime/profiles and HGB uncertainty extension.
5. All **400 official model/engine predictions**, four extended metric rows,
   both five-fold CV tables and both paired-bootstrap tables matched the report
   evidence with **maximum absolute difference 0.0**. See
   [reproduction_check.json](../evidence/verification/reproduction_check.json).
6. The HTML source manifest check and regeneration; regenerating pages in the
   independent clone produced no Git differences.
7. The official NASA download command, separately exercised in an empty data
   directory, retrieved the same three FD001 files and passed all SHA-256 checks.

Timings and profile summaries from the rerun are deliberately not substituted
for the original hardware measurements in the written report. They vary with
load and instrumentation. The final Moodle ZIP is produced from a fresh clone
of the published remote main branch and contains verified raw FD001 files and
Git history; environments, caches and regenerated output directories are omitted.

The earlier reference project remains on its archive branch. Its 37-test result
and different model metrics are separate from the current 44-case suite and
four-model report. Packaging history records actual subsequent work rather
than inventing a Git timeline for the earlier local Assignment development.
