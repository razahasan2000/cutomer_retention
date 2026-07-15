from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, Dict, Optional


class BaseProcessor(ABC):
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.numeric_cols: list = []
        self.categorical_cols: list = []
        self.encoded_col_names: list = []
        self.target_col: str = "Churn"
        self.fitted = False

    @abstractmethod
    def load_data(self, path: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        df = self.clean(df)
        y = df[self.target_col].values

        if self.numeric_cols:
            X_num = self.scaler.fit_transform(df[self.numeric_cols].fillna(0))
        else:
            X_num = np.empty((len(df), 0))

        if self.categorical_cols:
            encoded = self.encoder.fit_transform(df[self.categorical_cols].fillna("missing"))
            self.encoded_col_names = self.encoder.get_feature_names_out(self.categorical_cols).tolist()
        else:
            encoded = np.empty((len(df), 0))
            self.encoded_col_names = []

        X = np.hstack([X_num, encoded]) if X_num.size and encoded.size else (X_num if X_num.size else encoded)
        self.fitted = True
        return pd.DataFrame(X, columns=self.numeric_cols + self.encoded_col_names), y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Processor not fitted. Call fit_transform first.")
        df = self.clean(df)

        if self.numeric_cols:
            X_num = self.scaler.transform(df[self.numeric_cols].fillna(0))
        else:
            X_num = np.empty((len(df), 0))

        if self.categorical_cols:
            encoded = self.encoder.transform(df[self.categorical_cols].fillna("missing"))
        else:
            encoded = np.empty((len(df), 0))

        X = np.hstack([X_num, encoded]) if X_num.size and encoded.size else (X_num if X_num.size else encoded)
        return pd.DataFrame(X, columns=self.numeric_cols + self.encoded_col_names)

    def get_feature_names(self) -> list:
        return self.numeric_cols + self.encoded_col_names
