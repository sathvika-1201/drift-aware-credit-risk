import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import PROCESSED_DATA_PATH, REPORTS_DIR, TARGET_COLUMN, DATE_COLUMN
from src.features.build_features import prepare_features


def load_drift_samples(
    path=PROCESSED_DATA_PATH,
    chunksize=100_000,
    max_reference_rows=50_000,
    max_current_rows=50_000,
):
    reference_chunks = []
    current_chunks = []

    reference_rows = 0
    current_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(path, chunksize=chunksize, low_memory=True),
        start=1,
    ):
        print(f"Reading chunk {chunk_number}...")

        chunk[DATE_COLUMN] = pd.to_datetime(chunk[DATE_COLUMN], errors="coerce")

        reference_chunk = chunk[chunk[DATE_COLUMN] < "2016-01-01"]
        current_chunk = chunk[chunk[DATE_COLUMN] >= "2018-01-01"]

        if reference_rows < max_reference_rows and not reference_chunk.empty:
            needed = max_reference_rows - reference_rows
            selected_reference = reference_chunk.head(needed)
            reference_chunks.append(selected_reference)
            reference_rows += len(selected_reference)

        if current_rows < max_current_rows and not current_chunk.empty:
            needed = max_current_rows - current_rows
            selected_current = current_chunk.head(needed)
            current_chunks.append(selected_current)
            current_rows += len(selected_current)

        print(f"Collected reference={reference_rows:,}, current={current_rows:,}")

        if reference_rows >= max_reference_rows and current_rows >= max_current_rows:
            break

    if not reference_chunks:
        raise ValueError("No reference rows found before 2016-01-01.")

    if not current_chunks:
        raise ValueError("No current rows found from 2018-01-01 onward.")

    reference = pd.concat(reference_chunks, ignore_index=True)
    current = pd.concat(current_chunks, ignore_index=True)

    return reference, current


def generate_drift_report():
    reference, current = load_drift_samples()

    reference = reference.drop(columns=[TARGET_COLUMN], errors="ignore")
    current = current.drop(columns=[TARGET_COLUMN], errors="ignore")

    reference = prepare_features(reference)
    current = prepare_features(current)

    common_columns = reference.columns.intersection(current.columns)
    reference = reference[common_columns]
    current = current[common_columns]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = Report(
        [
            DataDriftPreset(),
        ]
    )

    result = report.run(
        reference_data=reference,
        current_data=current,
    )

    output_path = REPORTS_DIR / "data_drift_report.html"

    result.save_html(str(output_path))

    print(f"Saved drift report to {output_path}")


if __name__ == "__main__":
    generate_drift_report()