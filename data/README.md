# NASA C-MAPSS FD001 data

The predictors are cycle (cycles), sensor_2/T24, sensor_3/T30 and sensor_4/T50
(Rankine), sensor_7/P30 and sensor_11/Ps30 (psia), and sensor_9/Nc (rpm).
IDs are grouping keys, not predictors. Training RUL is uncapped cycles to the
last recorded failure cycle; official labels refer only to each engine's final
observed test record. The report's Section 2 gives observed input ranges.

Source: National Aeronautics and Space Administration, Prognostics Center of
Excellence. [Official catalog](https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data).
[NASA archive](https://data.nasa.gov/docs/legacy/CMAPSSData.zip).

The original files are `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`.
SHA-256 fingerprints and byte counts are in [manifest.json](manifest.json).
Training has 20,631 records from 100 complete engine histories; testing has
13,096 records from 100 truncated histories and 100 endpoint RUL labels.
Source records have 26 columns: ID, cycle, three settings and 21 sensors.

Copy exact source files into `data/raw/` or use the acquisition commands in
the README. The project checks bytes before modelling and does not edit the
original files. Raw data is excluded from the public source repository. The
local Moodle clone ZIP includes only the student's supplied FD001 subset,
accompanied by this attribution and the same hash manifest.

Public accessibility is stated in NASA's catalog, but a specific dataset
license was not given in the inspected record. The project's MIT license
does not apply to NASA data. Source-derived numerical evidence is retained
for review with NASA attribution, without assigning a new data license.

Cite: Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation
modeling for aircraft engine run-to-failure simulation. In *2008 International
Conference on Prognostics and Health Management* (pp. 1-9). IEEE.
https://doi.org/10.1109/PHM.2008.4711414
