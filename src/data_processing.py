import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------
# Custom Transformers
# ---------------------------
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], errors="coerce")
        df["transaction_hour"] = df[self.datetime_col].dt.hour
        df["transaction_day"] = df[self.datetime_col].dt.day
        df["transaction_month"] = df[self.datetime_col].dt.month
        df["transaction_year"] = df[self.datetime_col].dt.year
        return df.drop(columns=[self.datetime_col])


class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col="CustomerId"):
        self.customer_id_col = customer_id_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        agg_df = df.groupby(self.customer_id_col).agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("Amount", "count"),
            std_transaction_amount=("Amount", "std")
        ).reset_index()
        agg_df["std_transaction_amount"] = agg_df["std_transaction_amount"].fillna(0)
        return agg_df


# ---------------------------
# Full Task 3 Feature Engineering Workflow
# ---------------------------
def feature_engineering_workflow(data_path):
    # Load raw data
    df = pd.read_csv(data_path)

    # Extract date/time features
    df = DateFeatureExtractor().fit_transform(df)

    # Aggregate per customer
    customer_df = CustomerAggregator().fit_transform(df)

    # Identify numerical and categorical columns
    numerical_cols = [
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount"
    ]
    categorical_cols = customer_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Preprocessing for numerical and categorical features
    preprocessing = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), numerical_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ])

    # Full pipeline
    full_pipeline = Pipeline([
        ("preprocess", preprocessing)
    ])

    processed_matrix = full_pipeline.fit_transform(customer_df)

    return processed_matrix, full_pipeline, customer_df
