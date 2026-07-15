import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from typing import Optional, Dict


class CoxPHAnalysis:
    def __init__(self):
        self.cph = CoxPHFitter()
        self.fitted = False

    def fit(self, df: pd.DataFrame, duration_col: str, event_col: str,
            formula: Optional[str] = None):
        if formula:
            from lifelines import CoxPHFitter as CPH
            self.cph = CPH()
            self.cph.fit(df, duration_col=duration_col, event_col=event_col, formula=formula)
        else:
            self.cph.fit(df, duration_col=duration_col, event_col=event_col)
        self.fitted = True
        return self

    def summary(self) -> pd.DataFrame:
        if not self.fitted:
            return pd.DataFrame()
        return self.cph.summary

    def plot(self, ax=None, **kwargs):
        if self.fitted:
            self.cph.plot(ax=ax, **kwargs)

    def predict_survival(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("CoxPH not fitted.")
        return self.cph.predict_survival_function(X).values

    def concordance_index(self) -> float:
        if not self.fitted:
            return 0.0
        return self.cph.concordance_index_
