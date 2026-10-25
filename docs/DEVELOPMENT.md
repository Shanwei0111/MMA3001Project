# Development and repository history

This repository originally held an earlier reference implementation using
windowed features, ridge regression and a different forest. Commit a69e7f7
and its ancestors remain reachable on archive/reference-implementation.
Those results are not the source of the current report's four-model table.

The student subsequently developed and ran the current eight-script workflow
in Assignment. At the start of repository preparation, that folder had no
Git history. The packaging commits therefore record actual work performed
now; they are not backdated evidence of continuous earlier Git use.

The branch submission/current-fd001-workflow records actual changes to:
package current scripts, remove machine-specific paths/import-time execution,
fix the test's accidentally shadowed split function, add meaningful checks,
generate API documentation and verify reproduction. It is integrated into
main without squashing away those changes. No prior commits are rewritten.

Future changes: create a focused branch, for example experiment/grouped-hgb,
change one defensible aspect, run tests and record results, then merge only
after review. Use concise commits such as fix: validate endpoint alignment,
test: cover corrupt source fingerprints, or docs: explain metric units.
Do not tune on official test results or relabel unperformed experiments as work.
