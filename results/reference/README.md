# Frozen report results

| Folder | Contents |
|---|---|
| metrics/ | Model accuracy, grouped CV and forest-depth comparison |
| predictions/ | Validation/test predictions and retained error analysis |
| uncertainty/ | Original and extended paired engine-bootstrap intervals |
| performance/ | Timings, selected-array storage, profiles and operation counts |
| figures/ | Original report/diagnostic PNG figures |
| historical/ | The original 36-case pytest report and its provenance note |

These files preserve the student's saved Assignment results. New runs write to
results/generated rather than replacing this snapshot. Original CSV names map
to these subfolders through file_index.json, used by the reproduction checker.
The report PDF is unchanged; only its editable Markdown figure links were updated.

The historical 36-case test file accidentally included a helper that masked the
production split function. The organized suite directly tests production code
and adds checks for the reusable functions. Its current reports are under
tests/reports, separate from this historical evidence.
