# Repository submission evidence

| Requirement | Repository evidence |
|---|---|
| Purpose, usage, environment and dependency orientation | README.md, pyproject.toml, requirements.txt |
| Reproducible computational workflow | src/rul_project/steps, CLI stages, data/manifest.json |
| Inline comments and consistent API docstrings | src/rul_project, scripts; Google style |
| Complete generated HTML in sync with source | docs/html, docs/source_manifest.json, scripts/build_docs.py, CI |
| Original code license and separate NASA attribution | LICENSE, THIRD_PARTY_NOTICES.md, data/README.md |
| Test functions and reports | tests, evidence/reported/pytest_results.xml, evidence/verification |
| Engineering validation and method selection | reports PDF, evidence/reported, docs/METHOD.md |
| Actual branches and meaningful commits | archive/reference-implementation; submission/current-fd001-workflow; main |
| Fresh-clone check | evidence/verification and docs/VERIFICATION.md |
| Zipped clone for Moodle | scripts/package_submission.py; generated ZIP delivered separately |

Earlier local development did not have its own Git history. Packaging commits
record actual subsequent repository work without backdating; the report's
original numerical evidence remains identifiable. Rubric completion also depends
on student understanding and the separate presentation/technical questions.
