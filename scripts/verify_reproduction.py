"""Refit frozen configurations and compare every prediction to saved evidence.

Run from a clean clone after installing dependencies:
    python scripts/verify_reproduction.py --data-dir data/raw

This checks reproducible final fitting/evaluation, not the full tuning search.
It creates a separate output directory and never alters the reference artifacts.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('.private/reproduction'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.output_dir.resolve()
    if destination == (root/'artifacts').resolve():
        raise ValueError('Use a separate output directory to preserve reference evidence')
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root/'artifacts/selection.json', destination/'selection.json')
    subprocess.run([sys.executable,'-m','jet_rul','evaluate','--data-dir',str(args.data_dir.resolve()),
                    '--output-dir',str(destination)],cwd=root,check=True)
    expected = pd.read_csv(root/'artifacts/test_predictions.csv')
    actual = pd.read_csv(destination/'test_predictions.csv')
    pd.testing.assert_frame_equal(expected, actual, check_exact=False, rtol=1e-10, atol=1e-10)
    columns = ['age','ridge','forest']
    delta = float(np.max(np.abs(expected[columns].to_numpy()-actual[columns].to_numpy())))
    result = {'scope':'frozen-config refit/evaluation from clean clone; not repeated search',
              'engines':len(actual),'models':columns,'max_absolute_prediction_difference':delta,
              'relative_tolerance':1e-10,'absolute_tolerance':1e-10,'passed':True}
    (destination/'reproduction_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
