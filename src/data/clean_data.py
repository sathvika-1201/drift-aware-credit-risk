import pandas as pd

from src.config import DATE_COLUMN, TARGET_COLUMN


GOOD_STATUSES = ["Fully Paid"]

BAD_STATUSES = [
    "Charged Off",
    "Default",
    "Late (31-120 days)",
]


LEAKAGE_COLUMNS = [
    "loan_status",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "out_prncp",
    "out_prncp_inv",
    "debt_settlement_flag",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
]


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df["loan_status"].isin(GOOD_STATUSES + BAD_STATUSES)]

    df[TARGET_COLUMN] = df["loan_status"].apply(
        lambda status: 0 if status in GOOD_STATUSES else 1
    )

    return df


def clean_percentage_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    percentage_columns = ["int_rate", "revol_util"]

    for col in percentage_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .replace("nan", None)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_term_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "term" in df.columns:
        df["term"] = (
            df["term"]
            .astype(str)
            .str.replace("months", "", regex=False)
            .str.strip()
        )
        df["term"] = pd.to_numeric(df["term"], errors="coerce")

    return df


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    existing_leakage_cols = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    df = df.drop(columns=existing_leakage_cols)

    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = create_target(df)
    df = clean_dates(df)
    df = clean_percentage_columns(df)
    df = clean_term_column(df)
    df = drop_leakage_columns(df)

    df = df.dropna(subset=[DATE_COLUMN, TARGET_COLUMN])

    return df