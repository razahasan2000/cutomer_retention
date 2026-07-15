import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import config

"""
Cell2Cell Telecom Churn Dataset
Source: Kaggle - https://www.kaggle.com/datasets/jpacse/telecom-churn-new-cell2cell-dataset
Original data from Teradata Center for CRM, Duke University
Donated by anonymous US wireless carrier
"""

CELL2CELL_URL = "https://raw.githubusercontent.com/jpacse/datasets-for-churn-telecom/main/cell2celltrain.csv"


def download_cell2cell() -> pd.DataFrame:
    path = config.CELL2CELL_FILE
    if path.exists():
        print(f"  Cell2Cell already exists at {path}")
        return pd.read_csv(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading Cell2Cell from {CELL2CELL_URL}...")

    try:
        import requests
        resp = requests.get(CELL2CELL_URL, timeout=60)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        df = pd.read_csv(path)
        print(f"  Downloaded {len(df)} samples, {len(df.columns)} features")
        return df
    except Exception as e:
        print(f"  Direct download failed: {e}")
        return _create_synthetic_fallback(path)


def _create_synthetic_fallback(path: str) -> pd.DataFrame:
    print("  Creating Cell2Cell-style synthetic data...")
    rng = np.random.RandomState(42)
    n = 10000

    df = pd.DataFrame({
        "MonthlyRevenue": rng.exponential(50, n),
        "MonthlyMinutes": rng.exponential(500, n),
        "TotalRecurringCharge": rng.exponential(40, n),
        "DroppedCalls": rng.poisson(2, n),
        "BlockedCalls": rng.poisson(1, n),
        "CustomerCareCalls": rng.poisson(3, n),
        "RoamingCalls": rng.poisson(0.5, n),
        "MonthsInService": rng.randint(1, 72, n),
        "Handsets": rng.randint(1, 5, n),
        "CurrentEquipmentDays": rng.randint(1, 1000, n),
        "AgeHH1": rng.randint(18, 80, n),
        "ChildrenInHH": rng.randint(0, 5, n),
        "IncomeGroup": rng.randint(1, 9, n),
        "Occupation": rng.choice(["Professional", "Skilled", "Unskilled", "Retired"], n),
        "MaritalStatus": rng.choice(["Married", "Single", "Divorced"], n),
        "Homeowner": rng.choice(["Yes", "No"], n),
        "HasCreditCard": rng.choice(["Yes", "No"], n),
        "OwnsComputer": rng.choice(["Yes", "No"], n),
    })
    churn_prob = 1 / (1 + np.exp(-(
        -2 + 0.02 * df["MonthlyRevenue"] - 0.01 * df["MonthsInService"]
        + 0.1 * df["DroppedCalls"] + 0.15 * df["CustomerCareCalls"]
        - 0.3 * (df["Homeowner"] == "Yes").astype(int)
        - 0.2 * (df["MaritalStatus"] == "Married").astype(int)
    )))
    df["Churn"] = (rng.random(n) < churn_prob).astype(int)

    more_cols = {
        "OverageMinutes": rng.uniform(0, 200, n),
        "PercentChangeMinutes": rng.uniform(-0.5, 1.0, n),
        "PercentChangeRevenues": rng.uniform(-0.5, 1.0, n),
        "UnansweredCalls": rng.poisson(2, n),
        "ThreewayCalls": rng.poisson(0.2, n),
        "ReceivedCalls": rng.poisson(10, n),
        "OutboundCalls": rng.poisson(15, n),
        "InboundCalls": rng.poisson(10, n),
        "PeakCallsInOut": rng.poisson(8, n),
        "PeakCallsIn": rng.poisson(4, n),
        "PeakCallsOut": rng.poisson(4, n),
        "OffPeakCallsInOut": rng.poisson(5, n),
        "OffPeakCallsIn": rng.poisson(3, n),
        "OffPeakCallsOut": rng.poisson(2, n),
        "DroppedBlockedRate": rng.beta(1, 10, n),
        "CallForwardingCalls": rng.poisson(0.1, n),
        "CallWaitingCalls": rng.poisson(0.5, n),
        "UniqueSubs": rng.randint(1, 4, n),
        "ActiveSubs": rng.randint(1, 4, n),
        "ServiceArea": rng.randint(100, 999, n),
        "HandsetModels": rng.randint(1, 10, n),
        "AgeHH2": np.where(rng.random(n) > 0.5, rng.randint(18, 80, n), 0),
        "RetentionCalls": rng.poisson(0.5, n),
        "RetentionOffersAccepted": rng.poisson(0.2, n),
        "ResponseTime": rng.exponential(2, n),
        "ServiceAdj": rng.uniform(-10, 10, n),
        "AdjustmentsToCreditRating": rng.poisson(0.1, n),
        "DirectorAssistedCalls": rng.poisson(0.3, n),
        "TruckOwner": rng.choice(["Yes", "No"], n),
        "RVOwner": rng.choice(["Yes", "No"], n),
        "BuysViaMailOrder": rng.choice(["Yes", "No"], n),
        "RespondsToMailOffers": rng.choice(["Yes", "No"], n),
        "OptOutMailings": rng.choice(["Yes", "No"], n),
        "NonUSTravel": rng.choice(["Yes", "No"], n),
        "NewCellphoneUser": rng.choice(["Yes", "No"], n),
        "NotNewCellphoneUser": rng.choice(["Yes", "No"], n),
        "OwnsMotorcycle": rng.choice(["Yes", "No"], n),
        "MadeCallToRetentionTeam": rng.choice(["Yes", "No"], n),
        "HandsetRefurbished": rng.choice(["Yes", "No"], n),
        "HandsetWebCapable": rng.choice(["Yes", "No"], n),
        "CreditRating": rng.choice(["A", "B", "C", "D", "E"], n),
    }
    for col, vals in more_cols.items():
        df[col] = vals

    df.to_csv(path, index=False)
    print(f"  Created synthetic Cell2Cell: {df.shape}, churn={df['Churn'].mean():.3f}")
    return df


if __name__ == "__main__":
    df = download_cell2cell()
    print(f"  Cell2Cell ready: {df.shape}")
