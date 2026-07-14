import pandas as pd

from src.config import DATE_COLUMN, TARGET_COLUMN


def time_based_split(df: pd.DataFrame):
    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    train_df = df[df[DATE_COLUMN] < "2016-01-01"]
    valid_df = df[
        (df[DATE_COLUMN] >= "2016-01-01")
        & (df[DATE_COLUMN] < "2017-01-01")
    ]
    test_df = df[
        (df[DATE_COLUMN] >= "2017-01-01")
        & (df[DATE_COLUMN] < "2018-01-01")
    ]
    production_df = df[df[DATE_COLUMN] >= "2018-01-01"]

    return train_df, valid_df, test_df, production_df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y