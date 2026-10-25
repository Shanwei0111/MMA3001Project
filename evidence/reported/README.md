# Frozen report evidence

These files are the student's saved Assignment results, not a new experiment.
The written report uses this snapshot. Large derived training/split tables,
binary profiles, the excluded sensor_4 model and a stale three-model linear
table were omitted. Their current forms are regenerated into outputs/.

The original pytest_results.xml recorded 36 passing cases, but its test file
included a duplicate split_by_engine helper. The current production-targeted
suite corrects that coverage issue. Its result is stored separately under
evidence/verification, preserving the historical report evidence.

Report-specific RF/HGB figures and the extended HGB paired-bootstrap table
are included. Running the uncertainty stage reproduces them from the saved
final predictions. Timing/profile evidence is specific to the recorded machine.
