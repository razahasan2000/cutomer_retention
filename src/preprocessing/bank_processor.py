import pandas as pd
from src.preprocessing.base_processor import BaseProcessor


class BankProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.numeric_cols = [
            "CreditScore", "Age", "Tenure", "Balance",
            "NumOfProducts", "EstimatedSalary", "Point Earned"
        ]
        self.categorical_cols = [
            "Geography", "Gender", "HasCrCard", "IsActiveMember",
            "Card Type"
        ]
        self.target_col = "Exited"

    def load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for drop_col in ["RowNumber", "CustomerId", "Surname", "Complain", "Satisfaction Score"]:
            if drop_col in df.columns:
                df = df.drop(columns=[drop_col])
        for col in self.categorical_cols:
            df[col] = df[col].astype(str)
        return df
