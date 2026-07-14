import pandas as pd

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.data.clean_data import basic_cleaning


def process_and_save_data(
    input_path=RAW_DATA_PATH,
    output_path=PROCESSED_DATA_PATH,
    chunksize=100_000,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_chunk = True
    total_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(input_path, chunksize=chunksize, low_memory=True),
        start=1,
    ):
        print(f"Processing chunk {chunk_number}...")

        cleaned_chunk = basic_cleaning(chunk)

        cleaned_chunk.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        total_rows += len(cleaned_chunk)
        first_chunk = False

        print(f"Saved {total_rows:,} rows so far")

    print(f"Finished processing. Final saved rows: {total_rows:,}")
    print(f"Saved processed data to: {output_path}")


if __name__ == "__main__":
    process_and_save_data()