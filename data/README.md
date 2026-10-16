# Dataset provenance and access

NASA C-MAPSS FD001 files are required: train_FD001.txt, test_FD001.txt and
RUL_FD001.txt. Put them in data/raw/ without renaming or editing them.

Official catalog: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Archive linked by the catalog: https://data.nasa.gov/docs/legacy/CMAPSSData.zip

The supplied copy is verified by SHA-256 in manifest.json. The raw files are
excluded from public Git history. The catalog marks public access but does not
state a specific license, so this repository does not assign an MIT license to
NASA data. Read NASA's current terms before redistribution. The MIT license
covers this project's original code and documentation only. A local submission
archive may include the user's supplied FD001 files for examiner reproduction.

FD001 has 100 training engines / 20,631 rows, and 100 truncated test engines /
13,096 rows, with 100 separate endpoint RUL values. All observations have 26
columns: two identifiers, three settings and 21 sensors. The source README's
final sensor-number description is inconsistent with its actual 26-column files.

The preliminary audit checked all four subsets, but only FD001 is modeled.
Source summaries are not model-selection evidence. Source data are not cleaned
by silently dropping rows; the supplied FD001 passes finite/sequence checks.
