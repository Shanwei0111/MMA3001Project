"""Small independently specified fixtures; NASA files are unnecessary for CI."""
import numpy as np
import pandas as pd
import pytest
from jet_rul.data import COLUMNS, SENSORS


@pytest.fixture
def trajectories():
    """Two positive sensor trajectories with known slopes and different offsets."""
    rows = []
    for engine in [1, 2]:
        for cycle in range(1, 51):
            row = {name: 1.0 for name in COLUMNS}
            row.update(engine_id=engine, cycle=cycle)
            for j, sensor in enumerate(SENSORS):
                row[sensor] = 200*engine + (j+1)*cycle
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)
