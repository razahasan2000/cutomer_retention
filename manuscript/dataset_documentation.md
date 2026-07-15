# ESACRIF Dataset Documentation

## Overview
Three publicly available customer churn datasets from different domains and geographies, enabling cross-dataset generalization analysis.

---

## Dataset 1: IBM Telco Customer Churn

| Property | Value |
|---|---|
| **Source** | Kaggle / IBM Sample Data Sets |
| **URL** | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| **License** | CC0: Public Domain |
| **Samples** | 7,043 |
| **Features** | 20 (10 numeric, 10 categorical after encoding) |
| **Target** | `Churn` (Yes/No → 1/0) |
| **Churn Rate** | 26.5% (1,869 churned / 5,174 stayed) |
| **Domain** | Telecommunications (California, USA) |
| **Time Period** | Last month of observation |

### Feature Description
| Feature | Type | Description |
|---|---|---|
| customerID | ID | Unique customer identifier |
| gender | Binary | Male / Female |
| SeniorCitizen | Binary | 0 = No, 1 = Yes |
| Partner | Binary | Has partner (Yes/No) |
| Dependents | Binary | Has dependents (Yes/No) |
| tenure | Numeric | Months with company (0-72) |
| PhoneService | Binary | Has phone service |
| MultipleLines | Categorical | No, Yes, No phone service |
| InternetService | Categorical | DSL, Fiber optic, No |
| OnlineSecurity | Categorical | No, Yes, No internet service |
| OnlineBackup | Categorical | No, Yes, No internet service |
| DeviceProtection | Categorical | No, Yes, No internet service |
| TechSupport | Categorical | No, Yes, No internet service |
| StreamingTV | Categorical | No, Yes, No internet service |
| StreamingMovies | Categorical | No, Yes, No internet service |
| Contract | Categorical | Month-to-month, One year, Two year |
| PaperlessBilling | Binary | Yes/No |
| PaymentMethod | Categorical | Electronic check, Mailed check, Bank transfer, Credit card |
| MonthlyCharges | Numeric | Monthly charge ($18-119) |
| TotalCharges | Numeric | Total charged |
| Churn | Binary | Target: Yes/No |

### Known Limitations
- Zero-tenure customers have blank TotalCharges (11 rows, imputed with median)
- Class imbalance (26.5% churn rate)
- Single geography (California)
- No temporal/churn timing information
- No customer lifetime value column in raw data

---

## Dataset 2: Iranian Churn Dataset

| Property | Value |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset |
| **DOI** | 10.24432/C5JW3Z |
| **License** | CC BY 4.0 |
| **Samples** | 3,150 |
| **Features** | 13 |
| **Target** | `Churn` (1 = churn, 0 = non-churn) |
| **Churn Rate** | ~15.7% |
| **Domain** | Telecommunications (Iran) |
| **Time Period** | 12 months (9 months observation, 3 months gap) |

### Feature Description
| Feature | Type | Description |
|---|---|---|
| Call Failure | Integer | Number of call failures |
| Complains | Binary | 0 = No complaint, 1 = complaint |
| Subscription Length | Integer | Total months of subscription |
| Charge Amount | Ordinal | 0 = lowest amount, 9 = highest amount |
| Seconds of Use | Integer | Total seconds of calls |
| Frequency of use | Integer | Total number of calls |
| Frequency of SMS | Integer | Total number of text messages |
| Distinct Called Numbers | Integer | Total number of distinct phone calls |
| Age Group | Ordinal | 1 = younger age, 5 = older age |
| Tariff Plan | Binary | 1 = Pay as you go, 2 = contractual |
| Status | Binary | 1 = active, 2 = non-active |
| Customer Value | Float | Calculated value of customer |
| Age | Integer | Customer age |
| Churn | Binary | 1 = churn, 0 = non-churn |

### Known Limitations
- Smaller sample size (3,150)
- Aggregated data (no monthly breakdown)
- Limited categorical features
- Different churn definition (12-month window)
- Single country (Iran)

---

## Dataset 3: Bank Customer Churn Dataset

| Property | Value |
|---|---|
| **Source** | Kaggle |
| **URL** | https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn |
| **License** | Other (specified in description) |
| **Samples** | 10,000 |
| **Features** | 14 (after cleaning) |
| **Target** | `Exited` (1 = left, 0 = stayed) |
| **Churn Rate** | ~20.5% |
| **Domain** | Banking (Multinational) |
| **Time Period** | Cross-sectional |

### Feature Description
| Feature | Type | Description |
|---|---|---|
| CreditScore | Integer | Customer credit score (350-850) |
| Geography | Categorical | France, Germany, Spain |
| Gender | Binary | Male, Female |
| Age | Integer | Customer age (18-92) |
| Tenure | Integer | Years as customer (0-10) |
| Balance | Float | Account balance |
| NumOfProducts | Integer | Products purchased (1-4) |
| HasCrCard | Binary | Has credit card |
| IsActiveMember | Binary | Active member flag |
| EstimatedSalary | Float | Estimated annual salary |
| Card Type | Categorical | SILVER, GOLD, PLATINUM, DIAMOND |
| Point Earned | Integer | Credit card points |
| Exited | Binary | Target: 1 = churned |

### Known Limitations
- Synthetic proxy used when direct download unavailable
- No temporal features (cross-sectional)
- Limited service-related features
- Complaint/satisfaction data requires careful handling

---

## Dataset Comparison Summary

| Criterion | Telco | Iranian | Bank |
|---|---|---|---|
| Samples | 7,043 | 3,150 | 10,000 |
| Features | 20 | 13 | 14 |
| Churn Rate | 26.5% | 15.7% | 20.5% |
| Domain | Telecom | Telecom | Banking |
| Geography | USA | Iran | Multinational |
| Time Info | Tenure (months) | Subscription length | Tenure (years) |
| Imbalance | Moderate | Moderate | Moderate |
| Service Info | Yes (9 services) | No | Limited |
| Contract Info | Yes (3 types) | Yes (tariff) | No |

## Data Preprocessing Notes
- All datasets: one-hot encode categoricals, standardize numerics
- Telco: TotalCharges coerced to numeric, blanks imputed with median
- Iranian: Column names normalized (removed double spaces)
- Bank: RowNumber, CustomerId, Surname dropped; Exited used as target

## Citation Information
```bibtex
@misc{telco_churn,
  author = {IBM},
  title = {Telco Customer Churn},
  year = {2019},
  url = {https://www.kaggle.com/datasets/blastchar/telco-customer-churn}
}

@misc{iranian_churn,
  title = {Iranian Churn},
  year = {2020},
  doi = {10.24432/C5JW3Z},
  url = {https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset}
}
```
