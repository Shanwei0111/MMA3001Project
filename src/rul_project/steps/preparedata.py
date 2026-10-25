"""Prepare uncapped training labels from train_FD001.txt.

Reads the prepared/source files specified in the README under configured
DATA_DIR/OUTPUT_DIR and writes train_FD001_labeled.csv. Importing this module performs
no experiment. CLI usage: ``python -m rul_project preparedata``.
"""
from rul_project.settings import DATA_DIR, OUTPUT_DIR


def main() -> None:
    """Prepare uncapped training labels from train_FD001.txt.

    Returns:
        None: Writes train_FD001_labeled.csv to OUTPUT_DIR and prints summaries.

    Raises:
        FileNotFoundError: If a required source or preceding-stage CSV is absent.
        ValueError: If parsing, schema, alignment or estimator validation fails.
        OSError: If input/output access is unavailable.

    No arguments are accepted. Configure paths through the CLI before import.
    Model settings and output names are preserved from the student's script.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from rul_project.rul_utils import add_training_rul
    from pathlib import Path
    import pandas as pd

    # Set the path to the run-to-failure training data.
    file_path = DATA_DIR / "train_FD001.txt"

    # Read the whitespace-separated file without a header.
    data = pd.read_csv(file_path, sep=r"\s+", header=None)

    # Check the expected number of columns.
    if data.shape[1] != 26:
        raise ValueError(
            f"Expected 26 columns, but found {data.shape[1]}."
        )

    # Assign names to the source columns.
    column_names = [
        "engine_id",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
    ]

    sensor_names = [f"sensor_{i}" for i in range(1, 22)]

    data.columns = column_names + sensor_names

    # Calculate labels using the shared, testable function.
    data = add_training_rul(data)

    # Display a dataset summary.
    print("Training dataset summary")
    print("------------------------")
    print("Number of records:", len(data))
    print("Number of engines:", data["engine_id"].nunique())
    print("Number of columns after adding RUL:", data.shape[1])
    print("Minimum RUL:", data["RUL"].min())

    # Select the records belonging to engine 1.
    engine_1 = data[data["engine_id"] == 1]

    # Display the beginning and end of engine 1's history.
    print("\nEngine 1 - first three records:")
    print(
        engine_1[["engine_id", "cycle", "RUL"]]
        .head(3)
        .to_string(index=False)
    )

    print("\nEngine 1 - last three records:")
    print(
        engine_1[["engine_id", "cycle", "RUL"]]
        .tail(3)
        .to_string(index=False)
    )

    print("\nEngine 1 - failure cycle:", engine_1["cycle"].max())

    # Calculate the lifetime of each training engine.
    engine_lifetimes = data.groupby("engine_id")["cycle"].max()

    print("\nTraining engine lifetimes")
    print("------------------------")
    print("Minimum failure cycle:", engine_lifetimes.min())
    print("Maximum failure cycle:", engine_lifetimes.max())
    print("Mean failure cycle:", round(engine_lifetimes.mean(), 2))

    # Save the labeled dataset beside this script.
    output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "train_FD001_labeled.csv"
    data.to_csv(output_path, index=False)

    print("\nLabeled dataset saved to:", output_path)

if __name__ == "__main__":
    main()
