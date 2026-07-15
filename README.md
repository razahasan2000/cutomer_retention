# ESACRIF: Explainable Survival-Aware Customer Retention Intelligence Framework

**ESACRIF** reframes customer churn from a static classification problem into a
reproducible **decision-intelligence framework** that answers the five questions
retention managers actually face: **who** will leave, **when**, **why**, **what**
to do, and **which** interventions are worth funding.

It integrates seven model families with behavioural feature engineering,
Kaplan–Meier / Cox / RMST survival modelling, SHAP explainability with
stability auditing, DiCE counterfactual recommendations, and an **adaptive
profit-and-ROI retention optimiser** that *learns* the profit-maximising
intervention threshold from data. All experiments use a fixed seed (42), SMOTE
applied only within training folds, and bootstrap / DeLong / McNemar statistical
validation.

> Manuscript: *Beyond Churn Prediction: An Explainable Survival-Aware Customer
> Retention Intelligence Framework with Counterfactual Recommendations and
> Business Optimisation* — see `outputs/MANUSCRIPT.md` (excluded from this repo;
> contact the authors for the draft).

## Quick Start

```bash
pip install -r requirements.txt
# datasets are included under data/ ; to refresh:
# python data/download_datasets.py
python pipeline.py          # reproduces all 13 tables + 9 figures (seed 42)
streamlit run app.py        # interactive Streamlit UI (prediction, SHAP, counterfactuals, ROI simulator)
```

The full pipeline runs end-to-end in roughly 15–25 minutes on a CPU and writes
all results under `outputs/`.

## The Five Intelligence Layers

| Layer | What it does | Key outputs |
|---|---|---|
| 1. Prediction | 7 model families (LR → TabNet), SMOTE train-only, multi-objective selection | AUC, PR-AUC, calibration (ECE) |
| 2. Survival | Kaplan–Meier, RMST, Cox PH on tenure-as-duration | RMST, hazard ratios, timing by segment |
| 3. Explainability | SHAP global importance + **5-fold stability** (Spearman ρ, Kendall τ) + cross-model agreement | stability scores 0.86–1.00 |
| 4. Counterfactual | DiCE minimal, actionable feature changes | intervention recommendations |
| 5. Optimisation | CLV + principled expected-profit model; **adaptive threshold optimiser** (learns τ*) | campaign ROI, threshold comparison |

Responsible-AI auditing (fairness across gender / SeniorCitizen / Partner /
Dependents; robustness to noise, label-flip, missing data) and statistical
validation (DeLong, McNemar, bootstrap CI) are interleaved throughout.

## Research Contributions

1. **Explainable predictive modelling** across seven families with statistical validation (RQ1).
2. **Time-aware churn intelligence** via Kaplan–Meier, RMST, and Cox PH (RQ2).
3. **Stable explainability** with explicit 5-fold SHAP stability and cross-model agreement (RQ3).
4. **Actionable intervention** via DiCE counterfactual recommendations (RQ4).
5. **Business-aware retention optimisation** with an adaptive threshold optimiser that learns the profit-maximising cutoff (RQ5).
6. **Formal pipeline specification**, a **component ablation study**, and a **baseline-framework comparison** (capability matrix + DeLong/McNemar).

## Models

- Logistic Regression (interpretable baseline)
- Decision Tree, Random Forest (bagging)
- XGBoost, LightGBM, CatBoost (gradient boosting)
- TabNet (deep tabular)

## Datasets

| Dataset | Samples | Features | Domain | Churn rate | Notes |
|---|---|---|---|---|---|
| IBM Telco | 7,043 | 46 | Telecom | 26.5% | Primary dataset |
| Iranian Telecom | 3,150 | 19 | Telecom | 15.7% | External validation |
| Cell2Cell | 10,000 | 82 | Telecom | 30.4% | Synthetic fallback (original 71k source unavailable) |
| Bank (synthetic) | 10,000 | 20 | Banking | 20.5% | Software testing only — excluded from all claims |

## Key Findings (Telco, seed=42)

- **Prediction:** Logistic Regression reaches the highest AUC (**0.8397**) and is **not significantly outperformed** by any black-box model (DeLong p > 0.05 for all 21 pairwise comparisons). CatBoost leads calibration (ECE **0.052**).
- **Survival:** RMST = **54.49 months** (72-month horizon); month-to-month customers have less than half the expected retention duration of two-year-contract customers (**36.3 vs 71.5 months**); Cox concordance **0.9036**.
- **Explainability:** SHAP rankings are highly stable across 5 folds (Stability Score **0.86–1.00**); top drivers are tenure, contract type, and charges.
- **Counterfactuals:** DiCE translates risk into actionable contract/tenure changes.
- **Business:** Under a principled expected-profit model, the **adaptive threshold optimiser** learns τ* ≈ **$30** (cost breakeven) and yields **+48.0% ROI (+$3,957.54)** — matching the best hand-tuned rule while dominating random (−40.3%) and fixed high-risk (−24.3%) targeting; best-case calibrated ROI ≈ **258%**.
- **Fairness:** Gender passes demographic parity (0.003); SeniorCitizen (0.241) and Partner fail — reported honestly as a finding.
- **Generalisation:** Cross-dataset transfer AUCs of 0.38–0.76 evidence behavioural domain shift, motivating domain-adaptive retention models.

## Formal Specification, Ablation & Baselines

- **Algorithm 1** (in the manuscript, §3.8) formally specifies the offline training and online deployment phases, including the correctness guarantees (τ* dominates any fixed-percentile rule; SMOTE train-only ⇒ no leakage).
- **Component ablation (Table 4):** a cumulative study where each configuration adds exactly one layer. Feature engineering lifts AUC marginally (+0.005); explainability and survival add capabilities without changing point prediction; only the full configuration reaches strong calibration (ECE 0.0571) and positive ROI (+48.0%).
- **Baseline comparison (Tables 12 / 12b):** a capability matrix against four literature archetypes (Vanilla ML, +SHAP, Survival, ROI/Prescriptive) plus a quantitative DeLong/McNemar test showing ESACRIF's deployed model is statistically non-inferior to every baseline model (all DeLong p > 0.99).

## Reproducibility

- **Seed:** 42. **SMOTE:** k=5, train-only. **Split:** 80/20 stratified.
- **Command:** `python pipeline.py` → `outputs/tables/` (13 tables) and `outputs/figures/` (9 figures).
- All numbers in the manuscript are taken directly from the generated CSVs.

## Project Structure

```
ESACRIF/
├── app.py                     # Streamlit UI
├── pipeline.py                # Full experiment orchestration
├── experiments.py             # All Q1 experiments (models, stability, ablation, business, stats)
├── config.py                  # Central configuration
├── data/                     # Raw datasets (telco, iranian, cell2cell, bank)
├── src/
│   ├── preprocessing/         # Dataset-specific processors
│   ├── feature_engineering/   # Time-aware features + ablation
│   ├── prediction/            # 7 model wrappers
│   ├── survival/              # KM, Cox PH, RSF
│   ├── explainability/        # SHAP, Permutation, Stability
│   ├── counterfactual/        # DiCE-based counterfactuals
│   ├── business/              # CLV, Retention ROI optimizer
│   ├── fairness/              # Demographic parity, Equal opportunity
│   ├── robustness/            # Noise, missing, shift tests
│   ├── statistics/            # Bootstrap, DeLong, McNemar
│   └── cross_dataset/         # Transfer learning experiments
├── outputs/                   # Generated tables/figures (manuscript docs excluded)
└── tests/                     # Unit tests
```

## Citation

```bibtex
@article{vutla2025esacrif,
  title={Beyond Static Churn Prediction: An Explainable Survival-Aware
         Customer Retention Intelligence Framework},
  author={Vutla, Bala Aravind and Hasan, Raza},
  journal={Southampton Solent University},
  year={2025}
}
```

## License

Research purposes. Datasets carry their own licenses — see `dataset_documentation.md`.
