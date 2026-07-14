import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SELECTED_FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
]


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    available_features = [col for col in SELECTED_FEATURES if col in df.columns]

    return df[available_features]


def clean_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line"] = pd.to_datetime(
            df["earliest_cr_line"],
            format="%b-%Y",
            errors="coerce",
        )

        df["earliest_cr_line_year"] = df["earliest_cr_line"].dt.year
        df = df.drop(columns=["earliest_cr_line"])

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = select_model_features(df)
    df = clean_feature_types(df)

    return df


def get_feature_columns(df: pd.DataFrame):
    numeric_features = df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_columns(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    max_categories=30,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0.3,
    )

    return preprocessor