import os
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = PROJECT_ROOT / "data"
TELCO_DIR = DATA_DIR / "telco"
IRANIAN_DIR = DATA_DIR / "iranian"
BANK_DIR = DATA_DIR / "bank"
CELL2CELL_DIR = DATA_DIR / "cell2cell"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

TELCO_FILE = TELCO_DIR / "telco_customer_churn.csv"
IRANIAN_FILE = IRANIAN_DIR / "iranian_churn.csv"
BANK_FILE = BANK_DIR / "bank_churn.csv"
CELL2CELL_FILE = CELL2CELL_DIR / "cell2celltrain.csv"

SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5
SMOTE_K_NEIGHBORS = 5
N_BOOTSTRAP = 1000
VERBOSE = True

LR_PARAMS = {"max_iter": 1000, "random_state": SEED, "C": 1.0}
DT_PARAMS = {"max_depth": 6, "random_state": SEED}
RF_PARAMS = {"n_estimators": 200, "random_state": SEED, "n_jobs": -1}
XGB_PARAMS = {
    "eval_metric": "logloss", "random_state": SEED, "n_jobs": -1,
    "learning_rate": 0.1, "max_depth": 6, "n_estimators": 200
}
LGB_PARAMS = {
    "random_state": SEED, "n_jobs": -1, "verbose": -1,
    "learning_rate": 0.1, "max_depth": 6, "n_estimators": 200
}
CAT_PARAMS = {
    "random_state": SEED, "verbose": 0,
    "learning_rate": 0.1, "depth": 6, "iterations": 200
}
TABNET_PARAMS = {
    "n_d": 8, "n_a": 8, "n_steps": 3, "gamma": 1.3,
    "lambda_sparse": 0.001, "optimizer_fn": "torch.optim.Adam",
    "optimizer_params": {"lr": 2e-2},
    "scheduler_params": {"step_size": 50, "gamma": 0.9},
    "scheduler_fn": "torch.optim.lr_scheduler.StepLR",
    "epsilon": 1e-15, "seed": SEED
}

TABNET_EPOCHS = 50
TABNET_PATIENCE = 10
TABNET_BATCH_SIZE = 256

SYNTHETIC_DATASETS = {"bank": True, "telco": False, "iranian": False, "cell2cell": False}

SENSITIVE_ATTRIBUTES = ["gender", "SeniorCitizen", "Partner", "Dependents"]
FEATURE_GROUPS = {
    "demographic": ["gender", "SeniorCitizen", "Partner", "Dependents"],
    "service": [
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ],
    "account": ["Contract", "PaperlessBilling", "PaymentMethod", "tenure"],
    "billing": ["MonthlyCharges", "TotalCharges"],
    "engineered": [
        "AvgMonthlySpend", "ChargeAcceleration", "ContractRisk",
        "ServiceCount", "TenureDecile"
    ]
}

DATASET_INFO = {
    "telco": {
        "name": "IBM Telco Customer Churn",
        "source": "Kaggle / IBM Sample Data Sets",
        "url": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
        "license": "CC0: Public Domain",
        "n_samples": 7043,
        "n_features": 20,
        "churn_rate": 0.265,
        "domain": "Telecommunications"
    },
    "iranian": {
        "name": "Iranian Churn Dataset",
        "source": "UCI Machine Learning Repository",
        "url": "https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset",
        "license": "CC BY 4.0",
        "n_samples": 3150,
        "n_features": 13,
        "churn_rate": 0.22,
        "domain": "Telecommunications"
    },
    "bank": {
        "name": "Bank Customer Churn Dataset",
        "source": "Kaggle",
        "url": "https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn",
        "license": "Other (specified in description)",
        "n_samples": 10000,
        "n_features": 14,
        "churn_rate": 0.20,
        "domain": "Banking"
    },
    "cell2cell": {
        "name": "Cell2Cell Telecom Churn",
        "source": "Teradata Center / Duke University (via Kaggle)",
        "url": "https://www.kaggle.com/datasets/jpacse/telecom-churn-new-cell2cell-dataset",
        "license": "Research purposes (anonymous US carrier data)",
        "n_samples": 71047,
        "n_features": 75,
        "churn_rate": 0.02,
        "domain": "Telecommunications"
    }
}
