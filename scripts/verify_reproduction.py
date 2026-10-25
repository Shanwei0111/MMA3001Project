"""Compare a fresh workflow run with the frozen student report evidence.

Run after ``python -m rul_project all``. Compares deterministic quantities;
hardware-dependent timings, profiles and binary serialization sizes are not
required to match. A JSON verification report is written into the run directory.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


def main() -> None:
    """Check engine alignment, predictions, CV and bootstrap values.

    Returns:
        None: Writes reproduction_check.json, or raises on a mismatch.

    Raises:
        FileNotFoundError: If the reference or a generated CSV is absent.
        AssertionError: If IDs or deterministic values differ beyond tolerance.
        ValueError: If a CSV schema cannot be parsed.

    Usage: python scripts/verify_reproduction.py --output-dir outputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--reference-dir", type=Path, default=Path("evidence/reported"))
    args = parser.parse_args()
    checks = []
    maximum_difference = 0.0
    for name, keys, columns in [
        ("official_test_predictions.csv", ["model", "engine_id"], ["cycle", "RUL", "raw_predicted_RUL", "clipped_predicted_RUL"]),
        ("official_test_extended_metrics.csv", ["model"], ["test_engines", "mae", "rmse", "bias", "overprediction_pct", "nasa_score"]),
        ("random_forest_group_cv.csv", ["fold"], ["training_engines", "validation_engines", "mae", "rmse"]),
        ("hist_gradient_boosting_group_cv.csv", ["fold"], ["training_engines", "validation_engines", "mae", "rmse"]),
        ("bootstrap_summary.csv", ["metric"], ["estimate", "ci_lower", "ci_upper"]),
        ("extended_bootstrap_summary.csv", ["comparison"], ["mae", "mae_lower", "mae_upper", "rmse", "rmse_lower", "rmse_upper"]),
    ]:
        reference = pd.read_csv(args.reference_dir / name).sort_values(keys).reset_index(drop=True)
        reproduced = pd.read_csv(args.output_dir / name).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(reference[keys], reproduced[keys])
        np.testing.assert_allclose(reference[columns], reproduced[columns], rtol=1e-10, atol=1e-8)
        difference = float(np.max(np.abs(reference[columns].to_numpy() - reproduced[columns].to_numpy())))
        maximum_difference = max(maximum_difference, difference)
        checks.append({"file": name, "rows": len(reference), "max_absolute_difference": difference})
    result = {"status": "passed", "checks": checks, "maximum_absolute_difference": maximum_difference,
              "tolerance": {"rtol": 1e-10, "atol": 1e-8}, "excluded": ["timings", "profiles", "serialized size", "selected private arrays"]}
    (args.output_dir / "reproduction_check.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
