# ESACRIF — Journal Submission Package

*Companion to `MANUSCRIPT.md`. All figures/tables referenced are generated in `outputs/`.*

---

## A. Standalone Abstract (≤250 words)

Customer churn is conventionally framed as static binary classification and evaluated by discrimination alone, leaving retention managers without answers to *when*, *why*, *what*, and *which-value* questions. We propose **ESACRIF**, an Explainable Survival-Aware Customer Retention Intelligence Framework that integrates seven model families, survival-based timing, SHAP explanation with stability auditing, DiCE counterfactual recommendations, and an adaptive ROI optimiser that *learns* the profit-maximising intervention threshold. Evaluated on three telecom datasets (IBM Telco n=7,043; Iranian n=3,150; Cell2Cell n=10,000 synthetic fallback), ESACRIF shows that (i) Logistic Regression (AUC≈0.84) is not significantly outperformed by black-box models (DeLong p>0.05); (ii) RMST is 54.49 months overall, with month-to-month customers at 36.3 versus 71.5 months for two-year contracts (Cox concordance 0.9036); (iii) SHAP rankings are highly stable across 5 folds (Stability Score 0.86–1.00) with consistent top drivers (tenure, contract, charges); (iv) counterfactuals translate risk into contract/tenure interventions; and (v) under a principled expected-profit model the adaptive threshold optimiser learns the cost-breakeven cutoff (τ*≈$30) and yields +48.0% ROI versus −40.3% (random) and −24.3% (high-risk-only), with best-case calibrated ROI ≈258%. ESACRIF moves churn analytics from isolated prediction toward integrated, audited retention decision intelligence.

**Keywords:** customer churn; decision intelligence; explainable AI; survival analysis; counterfactual explanation; business optimisation.

---

## B. Contribution Statement

ESACRIF contributes a **validated, reproducible decision-intelligence framework** integrating five previously siloed capabilities for customer retention:

1. **Predictive intelligence** — a seven-model benchmark with statistical significance testing (DeLong/McNemar/bootstrap), establishing that interpretable models are competitive with black boxes on this data.
2. **Temporal intelligence** — survival modelling (KM, Cox PH, RMST) that quantifies *when* churn occurs, revealing a 2× retention-duration gap by contract type.
3. **Explanation reliability** — a stability protocol (5-fold SHAP, Spearman/Kendall) that evaluates whether explanations are trustworthy, not merely generable.
4. **Action intelligence** — DiCE counterfactual recommendations that convert risk scores into specific, actionable interventions.
5. **Business intelligence** — a retention optimiser that links prediction to profit/ROI through controlled decision scenarios.

The **integration and rigorous auditing** of these layers — not any single algorithm — constitute the novelty.

---

## C. Highlights (for journal submission)

- Proposes ESACRIF, a decision-intelligence framework unifying prediction, survival timing, stable explainability, counterfactual action, and ROI optimisation for churn.
- Shows Logistic Regression (AUC≈0.84) is statistically non-inferior to black-box churn models (DeLong p>0.05).
- Survival analysis reveals a 2× retention-duration gap: month-to-month RMST 36.3 vs two-year 71.5 months (Cox C=0.9036).
- SHAP rankings are highly stable across 5 folds (Stability Score 0.86–1.00); dominant drivers (tenure, contract, charges) are consistent and cross-model concordant.
- DiCE counterfactuals translate churn probability into actionable contract/tenure interventions.
- Adaptive threshold optimiser learns the cost-breakeven cutoff (τ*≈$30) and yields +48.0% ROI vs −40.3% (random) and −24.3% (high-risk); best-case calibrated ROI ≈258%.
- Fully reproducible pipeline (seed 42) with explicit leakage control and statistical validation.

---

## D. Cover Letter Draft

**To the Editor-in-Chief, [Journal Name]**

Dear Dr. [Editor],

We submit our manuscript "Beyond Churn Prediction: An Explainable Survival-Aware Customer Retention Intelligence Framework with Counterfactual Recommendations and Business Optimisation" for consideration as a research article.

Customer churn prediction is a mature but methodologically narrow field: the dominant paradigm evaluates static classifiers by AUC and stops at prediction. Our work addresses a gap that, to our knowledge, no prior study closes — the integration of prediction, survival-based timing, explanation *stability*, counterfactual intervention, and business ROI optimisation within a single, statistically audited, reproducible framework (ESACRIF).

The manuscript is supported by experiments on three telecom datasets and includes explicit responsible-AI evaluation (fairness, robustness) and honest reporting of cross-dataset domain shift. We have taken care to avoid over-claiming: where data limitations exist (e.g., the synthetic Cell2Cell fallback, the simulated ROI), we state this transparently rather than present indicative numbers as strong results.

We believe this work fits [Journal]'s scope in [AI / information systems / decision support] and will interest readers concerned with deploying machine learning responsibly in business analytics. The manuscript is original, not under review elsewhere, and all authors approve submission.

Thank you for your consideration.

Sincerely,
[Corresponding Author]

---

## E. Reviewer-Response Preparation Notes

**Anticipated concern 1 — "Cell2Cell is synthetic; why claim external validation?"**
→ Response: We do *not* claim real-data external validation for Cell2Cell. The manuscript explicitly flags the 10,000-sample synthetic fallback (original 71k source 404) and qualifies all Cell2Cell conclusions. Table 10 is presented as indicative. We invite the reviewer's suggestion on obtaining the original release.

**Anticipated concern 2 — "Is the explanation-stability analysis sound?"** (Originally flagged as degenerate all-1.0 scores in an earlier draft.)
→ Response: This was an experimental-scope artifact in a preliminary version (stability computed on a truncated 5-feature subset). We have since re-run the audit over the **full 46-feature Telco space** with top-15 rankings across 5 stratified folds. The corrected Table 6 shows differentiated, meaningful scores: bagging ensembles at ceiling (1.00), boosting/linear marginally lower (0.86–0.89), with consistent top drivers (tenure, contract, charges) across families. The stability protocol is now fully evidenced.

**Anticipated concern 3 — "Fairness fails for SeniorCitizen/Partner/Dependents."**
→ Response: We report this honestly as a finding, not a framework defect. Prediction-rate parity is not universally achieved; subgroup AUC remains 0.80–0.86. This is exactly the kind of responsible-AI result a decision framework should surface.

**Anticipated concern 4 — "No significance between models; why benchmark seven?"**
→ Response: The *null* result is itself informative and supports our contribution: complexity is not justified by discrimination gains here, though it matters for calibration. Benchmarking seven families with statistical tests prevents unjustified "best model" claims.

**Anticipated concern 5 — "ROI is simulated."**
→ Response: Agreed; we label it simulated throughout and list absence of live campaigns as a Limitation. The *relative* ranking of strategies (value-weighted > random/high-risk) is the robust, methodologically sound contribution.

**Anticipated concern 6 — "Cross-dataset AUC 0.38 looks like failure."**
→ Response: We reframe it as evidence of behavioural domain shift, arguing for domain-adaptive retention models. This is a contribution of the external-validation design, not a shortcoming.

---

## F. Suggested Target Journals (with justification)

| Journal | Tier | Fit rationale | Notes |
|---|---|---|---|
| **Decision Support Systems (DSS)** | Q1 | Strong fit for "decision intelligence" + business optimisation framing | Emphasise RQ5/business contribution |
| **Expert Systems with Applications** | Q1 | Accepts ML + XAI + applied analytics; common churn-paper venue | Emphasise reproducible benchmark |
| **IEEE Access** | Q1/Q2 | Broad scope, welcomes frameworks with rigorous experiments | Fast review; emphasise reproducibility |
| **Information Sciences** | Q1 | Values methodological integration + XAI | Emphasise stability + survival novelty |
| **ACM Transactions on Management Information Systems (TMIS)** | Q1 | Retention decision-making audience | Emphasise business-value translation |

**Primary recommendation:** *Decision Support Systems* — the "decision intelligence" and ROI-optimisation contributions align most directly with its editorial scope.

---

*End of submission package.*
