import pandas as pd
import numpy as np
from src.preprocessing.base_processor import BaseProcessor


class Cell2CellProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.numeric_cols = [
            "MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge",
            "DirectorAssistedCalls", "OverageMinutes", "RoamingCalls",
            "PercentChangeMinutes", "PercentChangeRevenues",
            "DroppedCalls", "BlockedCalls", "UnansweredCalls",
            "CustomerCareCalls", "ThreewayCalls", "ReceivedCalls",
            "OutboundCalls", "InboundCalls", "PeakCallsInOut",
            "PeakCallsIn", "PeakCallsOut", "OffPeakCallsInOut",
            "OffPeakCallsIn", "OffPeakCallsOut", "DroppedBlockedRate",
            "CallForwardingCalls", "CallWaitingCalls",
            "MonthsInService", "UniqueSubs", "ActiveSubs",
            "ServiceArea", "Handsets", "HandsetModels",
            "CurrentEquipmentDays", "AgeHH1", "AgeHH2",
            "ChildrenInHH", "IncomeGroup",
            "RetentionCalls", "RetentionOffersAccepted",
            "ResponseTime", "ServiceAdj", "AdjustmentsToCreditRating"
        ]
        self.categorical_cols = [
            "CreditRating", "PrizmCode", "Occupation", "MaritalStatus",
            "Homeowner", "HandsetRefurbished", "HandsetWebCapable",
            "TruckOwner", "RVOwner", "BuysViaMailOrder",
            "RespondsToMailOffers", "OptOutMailings",
            "NonUSTravel", "OwnsComputer", "HasCreditCard",
            "NewCellphoneUser", "NotNewCellphoneUser",
            "OwnsMotorcycle", "MadeCallToRetentionTeam"
        ]
        self.target_col = "Churn"

    def load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        drop_cols = ["CalendarID", "CalibrationGroup", "CalibrationStatus",
                     "CustomerID", "PrizmCode", "PrizmCodeDetailed",
                     "AdultsInHH", "HHstTenure", "HHstTenure_Days",
                     "DroppedBlockedRate_Calc", "CallWaiting_Calls",
                     "CallForwarding_Calls", "Threeway_Calls",
                     "CreditRating_Detailed"]
        drop_cols = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        id_cols = [c for c in df.columns if c.lower().startswith("customer") or c.lower().startswith("calend")]
        df = df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")

        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.fillna(0)

        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("missing")

        actual_num = [c for c in self.numeric_cols if c in df.columns]
        actual_cat = [c for c in self.categorical_cols if c in df.columns]
        self.numeric_cols = actual_num
        self.categorical_cols = actual_cat

        if self.target_col in df.columns:
            df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce").fillna(0).astype(int)

        return df
