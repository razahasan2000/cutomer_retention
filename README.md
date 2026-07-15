# ESACRIF: Explainable Survival-Aware Customer Retention Intelligence Framework

An integrated research framework that transforms customer churn prediction from a static classification task into a multi-faceted decision intelligence system combining prediction, survival analysis, explainability, counterfactual reasoning, fairness auditing, robustness testing, and business optimization.

## Quick Start

```bash
pip install -r requirements.txt
python data/download_datasets.py
python pipeline.py
streamlit run app.py
```

## Framework Architecture

```
ESACRIF/
├── app.py                          # Streamlit application (13 pages)
├── pipeline.py                     # Full experiment orchestration
├── config.py                       # Central configuration
├── dataset_documentation.md        # Full dataset metadata
├── data/                           # Raw datasets
│   ├── telco/                      # IBM Telco Customer Churn
│   ├── iranian/                    # UCI Iranian Churn
│   └── bank/                       # Bank Customer Churn
├── src/
│   ├── preprocessing/              # Dataset-specific processors
│   ├── feature_engineering/        # Time-aware features + ablation
│   ├── prediction/                 # 7 model wrappers
│   ├── survival/                   # KM, Cox PH, RSF
│   ├── explainability/             # SHAP, Permutation, Stability
│   ├── counterfactual/             # DiCE-based counterfactuals
│   ├── business/                   # CLV, Retention ROI optimizer
│   ├── fairness/                   # Demographic parity, Equal opp
│   ├── robustness/                 # Noise, missing, shift tests
│   ├── statistics/                 # Bootstrap, DeLong, McNemar
│   ├── cross_dataset/              # Transfer learning experiments
│   └── visualization/              # Publication-grade plotting
├── outputs/                        # Generated results
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── reports/
└── tests/                          # Unit tests
```

## Research Contributions

| # | Contribution | Status |
|---|---|---|
| 1 | Cross-Dataset Churn Intelligence | ✅ |
| 2 | Behaviour-Driven Feature Engineering | ✅ |
| 3 | Advanced ML Comparison (7 models) | ✅ |
| 4 | Explainable Survival-Aware Framework | ✅ |
| 5 | Actionable Counterfactual AI | ✅ |
| 6 | Business Retention Optimization | ✅ |
| 7 | Explainability Evaluation Metrics | ✅ |
| 8 | Fairness Evaluation | ✅ |
| 9 | Robustness Analysis | ✅ |
| 10 | Statistical Validation | ✅ |

## Models

- Logistic Regression (Interpretable)
- Decision Tree (Interpretable)
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- TabNet (Deep Learning)

## Datasets

| Dataset | Samples | Features | Domain | Churn Rate |
|---|---|---|---|---|
| IBM Telco | 7,043 | 20 | Telecom | 26.5% |
| Iranian Churn | 3,150 | 13 | Telecom | 15.7% |
| Bank Churn | 10,000 | 14 | Banking | 20.5% |

## Key Findings

- Logistic Regression achieves competitive AUC vs black-box models
- Behavioural features (AvgMonthlySpend, ContractRisk) improve prediction
- First 6 months are the critical retention window
- Cross-domain transfer is limited by feature space incompatibility
- Fairness analysis reveals disparities across sensitive attributes

## Citation

```bibtex
@article{vutla2025esacrif,
  title={Beyond Static Churn Prediction: An Explainable Survival-Aware 
         Customer Retention Intelligence Framework},
  author={Vutla, Bala Aravind Hasan, Raza and Salman Mahmood},
  journal={Southampton Solent University and Nazeer Hussain University},
  year={2025}
}
```

## License

This project is for research purposes. Datasets have their own licenses (see dataset_documentation.md).
