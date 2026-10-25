"""Split 100 training engines into 80/20 complete-history groups.

Reads the prepared/source files specified in the README under configured
DATA_DIR/OUTPUT_DIR and writes train_split.csv and validation_split.csv. Importing this module performs
no experiment. CLI usage: ``python -m rul_project splitdata``.
"""
from rul_project.settings import DATA_DIR, OUTPUT_DIR


def main() -> None:
    """Split 100 training engines into 80/20 complete-history groups.

    Returns:
        None: Writes train_split.csv and validation_split.csv to OUTPUT_DIR and prints summaries.

    Raises:
        FileNotFoundError: If a required source or preceding-stage CSV is absent.
        ValueError: If parsing, schema, alignment or estimator validation fails.
        OSError: If input/output access is unavailable.

    No arguments are accepted. Configure paths through the CLI before import.
    Model settings and output names are preserved from the student's script.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from rul_project.rul_utils import split_by_engine
    from pathlib import Path

    import pandas as pd

    # Load the labeled training dataset.
    output_dir = OUTPUT_DIR
    data_path = output_dir / "train_FD001_labeled.csv"

    data = pd.read_csv(data_path)

    # Split complete engine histories using the shared function.
    train_data, validation_data = split_by_engine(
        data,
        n_train=80,
        random_state=42,
    )

    # Check the actual engine groups in the resulting tables.
    train_ids = set(train_data["engine_id"])
    validation_ids = set(validation_data["engine_id"])

    shared_engine_count = len(
        train_ids.intersection(validation_ids)
    )

    print("Dataset split summary")
    print("---------------------")
    print("Training engines:", train_data["engine_id"].nunique())
    print("Validation engines:", validation_data["engine_id"].nunique())
    print("Shared engines:", shared_engine_count)
    print("Training records:", len(train_data))
    print("Validation records:", len(validation_data))

    # Save the two datasets.
    train_data.to_csv(output_dir / "train_split.csv", index=False)
    validation_data.to_csv(
        output_dir / "validation_split.csv",
        index=False,
    )

if __name__ == "__main__":
    main()
