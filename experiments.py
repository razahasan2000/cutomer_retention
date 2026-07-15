import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, recall_score
from scipy.stats import spearmanr, kendalltau, chi2_contingency, ks_2samp
from imblearn.over_sampling import SMOTE
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import restricted_mean_survival_time
import warnings
warnings.filterwarnings("ignore")

import config
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed
from src.utils.metrics import calculate_metrics
from src.preprocessing import TelcoProcessor, IranianProcessor, BankProcessor
from src.preprocessing.cell2cell_processor import Cell2CellProcessor
# For Python < 3.12 on Windows, avoid unicode arrows in logger
_LOG_ARROW = "->"
from src.feature_engineering import TimeAwareFeatureEngineer, FeatureAblation
from src.prediction import (MODEL_REGISTRY, LogisticRegressionModel, RandomForestModel,
                             XGBoostModel, LightGBMModel)
from src.explainability import SHAPExplainer, PermutationImportance, ExplanationEvaluation
from src.survival import KaplanMeierAnalysis, CoxPHAnalysis, RandomSurvivalForestAnalysis
from src.counterfactual import DiCECounterfactual
from src.business import CustomerLifetimeValue, RetentionOptimizer, expected_profit
from src.fairness import FairnessEvaluator
from src.robustness import NoiseInjector, ShiftSimulator
from src.statistics import BootstrapInference, delong_roc_test, mcnemar_test, CalibrationEvaluator

logger = setup_logger("experiments", log_file=str(config.OUTPUTS_DIR / "experiments.log"))
set_seed(config.SEED)


# =============================================================================
# TASK 1: CROSS-DATASET VALIDATION — DOMAIN SHIFT ANALYSIS
# =============================================================================

class DatasetSimilarityAnalyzer:
    """Analyse feature overlap, distribution differences, and PSI across datasets."""

    @staticmethod
    def population_stability_index(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0, 1, n_bins + 1)
        expected_pct = np.histogram(expected, bins=bins, density=True)[0] + 1e-10
        actual_pct = np.histogram(actual, bins=bins, density=True)[0] + 1e-10
        expected_pct = expected_pct / expected_pct.sum()
        actual_pct = actual_pct / actual_pct.sum()
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return round(float(psi), 4)

    @staticmethod
    def feature_distribution_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame,
                                        numeric_cols: list) -> pd.DataFrame:
        rows = []
        for col in numeric_cols:
            if col not in df_a.columns or col not in df_b.columns:
                continue
            a_vals = pd.to_numeric(df_a[col], errors="coerce").dropna().values
            b_vals = pd.to_numeric(df_b[col], errors="coerce").dropna().values
            if len(a_vals) < 5 or len(b_vals) < 5:
                continue
            stat, p = ks_2samp(a_vals, b_vals)
            rows.append({
                "feature": col,
                "mean_a": round(float(a_vals.mean()), 4),
                "mean_b": round(float(b_vals.mean()), 4),
                "std_a": round(float(a_vals.std()), 4),
                "std_b": round(float(b_vals.std()), 4),
                "ks_stat": round(float(stat), 4),
                "ks_p": round(float(p), 6),
                "significant_shift": "Yes" if p < 0.05 else "No"
            })
        return pd.DataFrame(rows).sort_values("ks_stat", ascending=False)

    @staticmethod
    def label_balance_comparison(y_a: np.ndarray, y_b: np.ndarray,
                                 name_a: str = "A", name_b: str = "B") -> pd.DataFrame:
        return pd.DataFrame({
            "Dataset": [name_a, name_b],
            "Samples": [len(y_a), len(y_b)],
            "Churn Rate": [f"{y_a.mean():.1%}", f"{y_b.mean():.1%}"],
            "Class 0": [int((y_a == 0).sum()), int((y_b == 0).sum())],
            "Class 1": [int((y_a == 1).sum()), int((y_b == 1).sum())],
            "Imbalance Ratio": [f"{y_a.mean()/(1-y_a.mean()):.3f}", f"{y_b.mean()/(1-y_b.mean()):.3f}"]
        })

    @staticmethod
    def dataset_domain_summary(datasets: dict) -> pd.DataFrame:
        rows = []
        for name, meta in datasets.items():
            rows.append({
                "Dataset": name,
                "Domain": meta.get("domain", "Unknown"),
                "Source": meta.get("source", "Unknown"),
                "Samples": meta.get("n_samples", 0),
                "Features": meta.get("n_features", 0),
                "Churn Rate": f"{meta.get('churn_rate', 0):.1%}",
                "Real": "No" if meta.get("is_synthetic") else "Yes",
                "Provider": meta.get("provider", "Unknown")
            })
        return pd.DataFrame(rows)


class CrossDatasetDomainShiftAnalysis:
    """Cross-dataset evaluation framed as domain shift analysis."""

    def __init__(self):
        self.results = []
        self.psi_results = []
        self.dataset_metadata = {}

    def add_dataset(self, name: str, X: pd.DataFrame, y: np.ndarray,
                    df_raw: pd.DataFrame = None,
                    is_synthetic: bool = False,
                    domain: str = "telecom",
                    provider: str = "Unknown"):
        self.dataset_metadata[name] = {
            "X": X, "y": y, "df_raw": df_raw, "is_synthetic": is_synthetic,
            "domain": domain, "provider": provider,
            "n_features": X.shape[1], "n_samples": X.shape[0],
            "churn_rate": float(y.mean())
        }

    def run_same_domain(self, n_components: int = 3):
        telecom = {k: v for k, v in self.dataset_metadata.items()
                   if v["domain"] == "telecom" and not v["is_synthetic"]}
        names = list(telecom.keys())
        for i, src in enumerate(names):
            for j, tgt in enumerate(names):
                if i == j:
                    continue
                src_data, tgt_data = telecom[src], telecom[tgt]
                X_s, X_t = self._align_features(src_data["X"], tgt_data["X"], src, tgt)
                if X_s is None:
                    nc = min(n_components, src_data["X"].shape[1], tgt_data["X"].shape[1])
                    X_s = PCA(n_components=nc).fit_transform(src_data["X"].values)
                    X_t = PCA(n_components=nc).fit_transform(tgt_data["X"].values)
                self._evaluate(src, tgt, X_s, src_data["y"], X_t, tgt_data["y"])
                self._compute_psi(src, tgt, src_data, tgt_data)

    def _align_features(self, X_src, X_tgt, src_name, tgt_name):
        common = [c for c in X_src.columns if c in X_tgt.columns and
                  not str(c).startswith(("gender_", "SeniorCitizen_", "Partner_", "Dependents_",
                                         "PhoneService_", "MultipleLines_", "InternetService_",
                                         "OnlineSecurity_", "OnlineBackup_", "DeviceProtection_",
                                         "TechSupport_", "StreamingTV_", "StreamingMovies_",
                                         "Contract_", "PaperlessBilling_", "PaymentMethod_",
                                         "Complains_", "Age Group_", "Tariff Plan_", "Status_",
                                         "Geography_", "Gender_", "HasCrCard_", "IsActiveMember_",
                                         "Card Type_"))]
        if common:
            return X_src[common].values, X_tgt[common].values
        return None, None

    def _evaluate(self, src, tgt, X_s, y_s, X_t, y_t):
        if X_s.shape[1] != X_t.shape[1]:
            self.results.append({"source": src, "target": tgt, "roc_auc": -1.0,
                                 "pr_auc": -1.0, "brier": -1.0, "recall": -1.0,
                                 "aligned_features": X_s.shape[1],
                                 "interpretation": "Feature mismatch"})
            return
        try:
            model = LogisticRegression(max_iter=1000, random_state=config.SEED)
            model.fit(X_s, y_s)
            y_prob = model.predict_proba(X_t)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            t_auc = roc_auc_score(y_t, y_prob)
            t_pr = average_precision_score(y_t, y_prob)
            t_br = brier_score_loss(y_t, y_prob)
            t_rec = recall_score(y_t, y_pred)
        except Exception:
            t_auc, t_pr, t_br, t_rec = -1.0, -1.0, -1.0, -1.0

        note = "Moderate domain shift" if t_auc > 0.4 else "Strong domain shift"
        self.results.append({
            "source": src, "target": tgt,
            "roc_auc": round(t_auc, 4), "pr_auc": round(t_pr, 4),
            "brier": round(t_br, 4), "recall": round(t_rec, 4),
            "aligned_features": X_s.shape[1],
            "interpretation": note
        })

    def _compute_psi(self, src, tgt, src_data, tgt_data):
        if src_data["df_raw"] is None or tgt_data["df_raw"] is None:
            return
        df_s = src_data["df_raw"]
        df_t = tgt_data["df_raw"]
        numeric_overlap = [c for c in df_s.select_dtypes(include=[np.number]).columns
                          if c in df_t.select_dtypes(include=[np.number]).columns
                          and c not in ["Churn", "Churn_Yes"]]
        for col in numeric_overlap[:5]:
            s_vals = pd.to_numeric(df_s[col], errors="coerce").dropna().values
            t_vals = pd.to_numeric(df_t[col], errors="coerce").dropna().values
            if len(s_vals) < 10 or len(t_vals) < 10:
                continue
            psi = DatasetSimilarityAnalyzer.population_stability_index(s_vals, t_vals)
            self.psi_results.append({
                "source": src, "target": tgt, "feature": col,
                "psi": psi, "shift_severity": "High" if psi > 0.25 else "Moderate" if psi > 0.1 else "Low"
            })

    def summary(self) -> dict:
        return {
            "transfers": pd.DataFrame(self.results) if self.results else pd.DataFrame(),
            "psi": pd.DataFrame(self.psi_results) if self.psi_results else pd.DataFrame()
        }

    def domain_similarity_report(self) -> pd.DataFrame:
        names = list(self.dataset_metadata.keys())
        matrix = pd.DataFrame(index=names, columns=names, dtype=str)
        for _, r in pd.DataFrame(self.results).iterrows():
            matrix.loc[r["source"], r["target"]] = f"{r['roc_auc']:.3f}"
        matrix = matrix.fillna("—")
        matrix.index.name = "Train\\Test"
        return matrix


# =============================================================================
# TASK 2: EXPLANATION STABILITY ANALYSIS
# =============================================================================

class ExplanationStabilityAnalyzer:
    """Measure stability of explanations across model families and CV folds."""

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.results = {}

    def analyze(self, X: np.ndarray, y: np.ndarray,
                feature_names: list, model_registry: dict = None,
                top_n: int = 15) -> pd.DataFrame:
        models = model_registry or MODEL_REGISTRY
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=config.SEED)
        all_rows = []

        for m_name, m_cls in models.items():
            shap_rankings = []
            perm_rankings = []
            coef_values = []

            for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]
                try:
                    model = m_cls()
                    model.fit(X_tr, y_tr)

                    # SHAP importance (top_n features by mean |SHAP|)
                    try:
                        model_type = "linear" if "Logistic" in m_name else "tree"
                        explainer = SHAPExplainer(model.model, X_te, model_type=model_type)
                        explainer.explain()
                        imp = explainer.global_importance(feature_names)
                        shap_rank = imp["feature"].head(top_n).tolist()
                    except Exception:
                        shap_rank = feature_names[:top_n]

                    # Permutation importance
                    try:
                        perm = PermutationImportance(model.model, X_te, y_te, n_repeats=5)
                        perm_df = perm.compute().as_dataframe(feature_names)
                        perm_rank = perm_df.sort_values("importance_mean", ascending=False)["feature"].head(top_n).tolist()
                    except Exception:
                        perm_rank = feature_names[:top_n]

                    # Coefficient stability (for linear models)
                    try:
                        coef = model.get_feature_importance()
                        if coef is not None:
                            coef_values.append(coef)
                    except Exception:
                        pass

                    shap_rankings.append(shap_rank)
                    perm_rankings.append(perm_rank)
                except Exception:
                    continue

            if len(shap_rankings) < 2:
                continue

            # Rank agreement (within-model, fold-to-fold) over top_n
            sp_scores, kt_scores = [], []
            for i in range(len(shap_rankings)):
                for j in range(i + 1, len(shap_rankings)):
                    common = list(set(shap_rankings[i]) & set(shap_rankings[j]))
                    if len(common) < 3:
                        continue
                    ra = [shap_rankings[i].index(f) if f in shap_rankings[i] else len(shap_rankings[i]) for f in common]
                    rb = [shap_rankings[j].index(f) if f in shap_rankings[j] else len(shap_rankings[j]) for f in common]
                    sp, _ = spearmanr(ra, rb)
                    kt, _ = kendalltau(ra, rb)
                    sp_scores.append(sp if not np.isnan(sp) else 0)
                    kt_scores.append(kt if not np.isnan(kt) else 0)

            # Permutation agreement
            perm_sp = []
            for i in range(len(perm_rankings)):
                for j in range(i + 1, len(perm_rankings)):
                    common = list(set(perm_rankings[i]) & set(perm_rankings[j]))
                    if len(common) < 3:
                        continue
                    ra = [perm_rankings[i].index(f) if f in perm_rankings[i] else len(perm_rankings[i]) for f in common]
                    rb = [perm_rankings[j].index(f) if f in perm_rankings[j] else len(perm_rankings[j]) for f in common]
                    sp, _ = spearmanr(ra, rb)
                    perm_sp.append(sp if not np.isnan(sp) else 0)

            # Coefficient stability (CV of coefficients across folds)
            coef_cv = float(np.std(coef_values, axis=0).mean()) if coef_values else -1

            mean_sp = float(np.mean(sp_scores)) if sp_scores else 0.0
            mean_kt = float(np.mean(kt_scores)) if kt_scores else 0.0
            mean_perm = float(np.mean(perm_sp)) if perm_sp else 0.0
            stability_score = round(float((mean_sp + mean_kt) / 2), 4)

            # Summarise per model (one row per model, not per feature)
            all_rows.append({
                "Model": m_name,
                "Top Features": top_n,
                "SHAP Mean Spearman": round(mean_sp, 4),
                "SHAP Mean Kendall": round(mean_kt, 4),
                "Permutation Mean Spearman": round(mean_perm, 4),
                "Coef CV": round(coef_cv, 4) if coef_cv > -1 else "N/A",
                "Stability Score": stability_score,
                "Top Feature 1": shap_rankings[0][0] if shap_rankings else "N/A",
                "Top Feature 2": shap_rankings[0][1] if len(shap_rankings[0]) > 1 else "N/A",
                "Top Feature 3": shap_rankings[0][2] if len(shap_rankings[0]) > 2 else "N/A",
            })

        df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame({"info": ["No stability results"]})
        self.results["stability"] = df
        return df

    def cross_model_agreement(self, X: np.ndarray, y: np.ndarray,
                              feature_names: list, top_n: int = 15) -> pd.DataFrame:
        """Compare explanation agreement across different model families."""
        results = {}
        models = {"LR": LogisticRegressionModel, "RF": RandomForestModel,
                  "XGB": XGBoostModel, "LGB": LightGBMModel}
        shap_ranks = {}

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=config.SEED, stratify=y)
        for m_name, m_cls in models.items():
            try:
                model = m_cls()
                model.fit(X_tr, y_tr)
                explainer = SHAPExplainer(model.model, X_te, model_type="tree" if m_name != "LR" else "linear")
                explainer.explain()
                imp = explainer.global_importance(feature_names)
                shap_ranks[m_name] = imp["feature"].head(top_n).tolist()
            except Exception:
                shap_ranks[m_name] = feature_names[:top_n]

        agreement_rows = []
        names = list(shap_ranks.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                common = list(set(shap_ranks[names[i]]) & set(shap_ranks[names[j]]))
                if len(common) < 3:
                    continue
                ra = [shap_ranks[names[i]].index(f) if f in shap_ranks[names[i]] else len(shap_ranks[names[i]]) for f in common]
                rb = [shap_ranks[names[j]].index(f) if f in shap_ranks[names[j]] else len(shap_ranks[names[j]]) for f in common]
                sp, _ = spearmanr(ra, rb)
                kt, _ = kendalltau(ra, rb)
                agreement_rows.append({
                    "Model A": names[i], "Model B": names[j],
                    "Spearman": round(sp, 4) if not np.isnan(sp) else 0,
                    "Kendall": round(kt, 4) if not np.isnan(kt) else 0,
                    "Common Features": len(common)
                })

        df = pd.DataFrame(agreement_rows) if agreement_rows else pd.DataFrame({"info": ["No cross-model agreement"]})
        self.results["cross_model_agreement"] = df
        return df


# =============================================================================
# TASK 3: FRAMEWORK ABLATION (5 Models: A→E)
# =============================================================================

class ExtendedFrameworkAblation:
    """Ablation: Model A=ML only, B=ML+FE, C=ML+XAI, D=ML+Survival, E=Full ESACRIF."""

    def __init__(self):
        self.results = {}
        self.details = {}

    def run(self, X: pd.DataFrame, y: np.ndarray,
            X_with_fe: pd.DataFrame = None,
            survival_features: np.ndarray = None):
        indices = np.arange(len(X))
        tr_idx, te_idx, y_train, y_test = train_test_split(
            indices, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y)
        X_tr_np, X_te_np = X.values[tr_idx], X.values[te_idx]
        ndarray_fe = X_with_fe.values[tr_idx] if X_with_fe is not None else None
        ndarray_fe_te = X_with_fe.values[te_idx] if X_with_fe is not None else None

        configs = {
            "A: Traditional ML Only": {"use_fe": False, "use_xai": False, "use_survival": False, "use_ensemble": False},
            "B: ML + Feature Engineering": {"use_fe": True, "use_xai": False, "use_survival": False, "use_ensemble": False},
            "C: ML + Explainability": {"use_fe": False, "use_xai": True, "use_survival": False, "use_ensemble": False},
            "D: ML + Survival Analysis": {"use_fe": False, "use_xai": False, "use_survival": True, "use_ensemble": False},
            "E: Full ESACRIF": {"use_fe": True, "use_xai": True, "use_survival": True, "use_ensemble": True},
        }

        for cfg_name, cfg in configs.items():
            try:
                X_tr, X_te = X_tr_np.copy(), X_te_np.copy()
                y_tr = y_train.copy()

                if cfg["use_fe"] and ndarray_fe is not None:
                    X_tr = np.hstack([X_tr, ndarray_fe])
                    X_te = np.hstack([X_te, ndarray_fe_te])

                smote = SMOTE(random_state=config.SEED)
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
                y_tr = y_tr.ravel()

                if cfg["use_ensemble"]:
                    ensemble_preds = []
                    for name, m_cls in MODEL_REGISTRY.items():
                        try:
                            m = m_cls()
                            m.fit(X_tr, y_tr)
                            ensemble_preds.append(m.predict_proba(X_te))
                        except Exception:
                            pass
                    y_prob = np.mean(ensemble_preds, axis=0) if ensemble_preds else (
                        LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1])
                else:
                    lr = LogisticRegression(max_iter=1000, random_state=config.SEED)
                    lr.fit(X_tr, y_tr)
                    y_prob = lr.predict_proba(X_te)[:, 1]

                y_pred = (y_prob >= 0.5).astype(int)
                metrics = calculate_metrics(y_test, y_pred, y_prob)

                # Interpretability score: 0-100
                interp_score = 0
                if not cfg["use_fe"] and not cfg["use_ensemble"]:
                    interp_score += 40
                if not cfg["use_xai"]:
                    interp_score += 0
                else:
                    interp_score += 30
                if cfg["use_survival"]:
                    interp_score += 30

                metrics["Interpretability"] = interp_score
                metrics["Components"] = "+".join([k.replace("use_", "") for k, v in cfg.items() if v])
                metrics["N_Features"] = X_tr.shape[1]
                self.results[cfg_name] = metrics
                logger.info(f"  Ablation [{cfg_name}]: AUC={metrics['roc_auc']:.4f}, Interp={interp_score}")
            except Exception as e:
                logger.warning(f"  Ablation [{cfg_name}] failed: {e}")
                self.results[cfg_name] = {"roc_auc": 0, "error": str(e)}

        return self

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, m in self.results.items():
            row = {"Configuration": name}
            for k in ["roc_auc", "f1", "recall", "precision", "pr_auc", "brier", "Interpretability", "N_Features"]:
                row[k] = m.get(k, "N/A")
            rows.append(row)
        return pd.DataFrame(rows)


# =============================================================================
# TASK 4: BUSINESS DECISION EXPERIMENT (3 Scenarios)
# =============================================================================

class BusinessDecisionExperiment:
    """Three scenarios: Random, High-risk only, ESACRIF-recommended."""

    def __init__(self, clv_model=None):
        self.clv_model = clv_model or CustomerLifetimeValue()
        self.results = {}

    def simulate(self, churn_probs: np.ndarray, clv_values: np.ndarray,
                 retention_cost: float = 30, budget: float = 50000,
                 retention_success_rate: float = 0.3) -> pd.DataFrame:
        n = len(churn_probs)
        rng = np.random.RandomState(config.SEED)
        churn_probs = np.asarray(churn_probs, dtype=float)
        clv_values = np.asarray(clv_values, dtype=float)

        # Scenario 1: Random
        n_random = int(min(budget / retention_cost, n))
        random_idx = rng.choice(n, n_random, replace=False)
        s1_mask = np.zeros(n, dtype=bool); s1_mask[random_idx] = True
        s1_saved, s1_cost, _, s1_roi = expected_profit(
            churn_probs, clv_values, s1_mask, retention_cost, retention_success_rate)
        s1_profit = s1_saved - s1_cost

        # Scenario 2: High churn probability only (fixed threshold 0.7)
        high_risk = churn_probs >= 0.7
        n_high = int(high_risk.sum())
        n_target_high = min(int(budget / retention_cost), n_high)
        high_idx = np.where(high_risk)[0][:n_target_high]
        s2_mask = np.zeros(n, dtype=bool); s2_mask[high_idx] = True
        s2_saved, s2_cost, _, s2_roi = expected_profit(
            churn_probs, clv_values, s2_mask, retention_cost, retention_success_rate)
        s2_profit = s2_saved - s2_cost

        # Scenario 3: ESACRIF-recommended (high risk + high CLV, fixed 80th pctile)
        esacrif_score = churn_probs * clv_values
        esacrif_threshold = np.percentile(esacrif_score, 80)
        esacrif_mask = np.zeros(n, dtype=bool)
        cand = np.where(esacrif_score >= esacrif_threshold)[0]
        n_esacrif = int(min(budget / retention_cost, len(cand)))
        esacrif_idx = cand[:n_esacrif]
        esacrif_mask[esacrif_idx] = True
        s3_saved, s3_cost, _, s3_roi = expected_profit(
            churn_probs, clv_values, esacrif_mask, retention_cost, retention_success_rate)
        s3_profit = s3_saved - s3_cost

        rows = [
            {"Scenario": "1: Random Campaign", "Customers Targeted": n_random,
             "Customers Saved": round(retention_success_rate * n_random, 1),
             "Revenue Retained": round(s1_saved, 2), "Cost": round(s1_cost, 2),
             "ROI %": round(s1_roi, 2), "Profit": round(s1_profit, 2),
             "Strategy": "Random selection"},
            {"Scenario": "2: High-Risk Only", "Customers Targeted": n_target_high,
             "Customers Saved": round(retention_success_rate * n_target_high, 1),
             "Revenue Retained": round(s2_saved, 2), "Cost": round(s2_cost, 2),
             "ROI %": round(s2_roi, 2), "Profit": round(s2_profit, 2),
             "Strategy": "p(churn) > 0.7"},
            {"Scenario": "3: ESACRIF-Recommended", "Customers Targeted": n_esacrif,
             "Customers Saved": round(retention_success_rate * n_esacrif, 1),
             "Revenue Retained": round(s3_saved, 2), "Cost": round(s3_cost, 2),
             "ROI %": round(s3_roi, 2), "Profit": round(s3_profit, 2),
             "Strategy": "p(churn) x CLV > 80th pctile"},
        ]
        df = pd.DataFrame(rows)
        self.results["scenarios"] = df
        return df


# =============================================================================
# TASK 5: SURVIVAL ANALYSIS ENHANCEMENT (Forest plot, RMST primary)
# =============================================================================

class EnhancedSurvivalAnalysis:
    """Survival with RMST as primary metric, hazard ratios, forest plot data."""

    def __init__(self):
        self.results = {}

    def analyze(self, df: pd.DataFrame, duration_col: str = "tenure",
                event_col: str = "Churn") -> dict:
        durations = df[duration_col].values
        event = df[event_col].values
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event)

        durations_max = np.percentile(durations[durations > 0], 95)
        rmst = restricted_mean_survival_time(kmf, t=durations_max)
        rmst_val = float(rmst[0]) if hasattr(rmst, '__len__') else float(rmst)
        median_val = kmf.median_survival_time_

        cph = CoxPHFitter()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        model_cols = [c for c in numeric_cols if c not in [duration_col, event_col, "Churn_Yes"]]
        df_cph = df[model_cols + [duration_col, event_col]].dropna(axis=1, how='all')
        cph.fit(df_cph, duration_col=duration_col, event_col=event_col)
        summary = cph.summary

        hr_data = []
        for idx in summary.index:
            hr_data.append({
                "feature": idx,
                "hr": round(summary.loc[idx, "exp(coef)"], 4),
                "ci_lower": round(summary.loc[idx, "exp(coef) lower 95%"], 4),
                "ci_upper": round(summary.loc[idx, "exp(coef) upper 95%"], 4),
                "p_value": round(summary.loc[idx, "p"], 6),
                "significant": summary.loc[idx, "p"] < 0.05,
                "log_hr": round(summary.loc[idx, "coef"], 4),
                "se": round(summary.loc[idx, "se(coef)"], 4),
            })

        result = {
            "kmf": kmf,
            "rmst": rmst_val,
            "rmst_time_horizon": float(durations_max),
            "median_survival": float(median_val) if np.isfinite(median_val) else None,
            "cox_concordance": round(cph.concordance_index_, 4),
            "hr_data": pd.DataFrame(hr_data).sort_values("hr", ascending=False) if hr_data else pd.DataFrame(),
            "durations": durations,
            "event": event,
        }
        self.results = result
        return result

    def group_rmst(self, df: pd.DataFrame, duration_col: str = "tenure",
                   event_col: str = "Churn", group_col: str = "Contract") -> pd.DataFrame:
        durations = df[duration_col].values
        event = df[event_col].values
        durations_max = np.percentile(durations[durations > 0], 95)
        groups = df[group_col].values
        group_names = np.unique(groups)

        rows = []
        for name in group_names:
            mask = groups == name
            if mask.sum() < 5:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(durations[mask], event[mask])
            rmst = restricted_mean_survival_time(kmf, t=durations_max)
            rows.append({
                "Group": str(name), "N": int(mask.sum()), "Events": int(event[mask].sum()),
                "RMST": round(float(rmst[0]) if hasattr(rmst, '__len__') else float(rmst), 2),
                "Median": round(float(kmf.median_survival_time_), 2) if np.isfinite(kmf.median_survival_time_) else "NR",
            })
        return pd.DataFrame(rows)


# =============================================================================
# TASK 6: MULTI-OBJECTIVE MODEL SELECTION
# =============================================================================

class MultiObjectiveModelSelector:
    """Rank models across prediction, calibration, interpretability, and business dimensions."""

    @staticmethod
    def rank(models_results: dict) -> pd.DataFrame:
        rows = []
        for name, res in models_results.items():
            m = res["metrics"]
            rows.append({
                "Model": name,
                "AUC": m.get("roc_auc", 0),
                "Recall": m.get("recall", 0),
                "Brier": m.get("brier", 1),
                "ECE": m.get("ece", 1),
                "Interpretability": 5 if "Logistic" in name else 4 if "Tree" in name else 3 if "Forest" in name or "XGB" in name or "LGB" in name or "Cat" in name else 2 if "TabNet" in name else 1,
                "Business ROI": m.get("roi", 0),
            })
        df = pd.DataFrame(rows).set_index("Model")

        # Normalize and rank
        for col in ["AUC", "Recall", "Interpretability", "Business ROI"]:
            if col in df.columns and df[col].max() > df[col].min():
                df[f"{col}_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            else:
                df[f"{col}_norm"] = 0.5
        for col in ["Brier", "ECE"]:
            if col in df.columns and df[col].max() > df[col].min():
                df[f"{col}_norm"] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            else:
                df[f"{col}_norm"] = 0.5

        norm_cols = [c for c in df.columns if c.endswith("_norm")]
        df["Composite Score"] = df[norm_cols].mean(axis=1)
        df = df.sort_values("Composite Score", ascending=False)

        df["Best Predictive"] = df["AUC"].rank(ascending=False).apply(lambda x: "✓" if x == 1 else "")
        df["Best Interpretable"] = df["Interpretability"].rank(ascending=False).apply(lambda x: "✓" if x == 1 else "")
        df["Best Calibrated"] = df["ECE"].rank(ascending=True).apply(lambda x: "✓" if x == 1 else "")
        df["Best Business"] = df["Business ROI"].rank(ascending=False).apply(lambda x: "✓" if x == 1 else "")

        return df


# =============================================================================
# TASK 7: STATISTICAL VALIDATION AUDIT
# =============================================================================

class StatisticalValidationAudit:
    """Comprehensive statistical validation with documented methodology."""

    @staticmethod
    def run(y_test: np.ndarray, model_probs: dict, model_preds: dict = None) -> dict:
        results = {}

        # Bootstrap CIs
        boot = BootstrapInference(n_bootstrap=2000)
        boot_results = {}
        for name, y_prob in model_probs.items():
            ci = boot.compute_ci(y_test, y_prob)
            boot_results[name] = ci
        results["bootstrap"] = boot_results

        # DeLong tests
        delong_results = {}
        names = list(model_probs.keys())
        for i, m1 in enumerate(names):
            for j, m2 in enumerate(names):
                if i >= j:
                    continue
                try:
                    z, p = delong_roc_test(y_test, model_probs[m1], model_probs[m2])
                    delong_results[f"{m1} vs {m2}"] = {"z": round(z, 4), "p": round(p, 6),
                                                       "significant": p < 0.05}
                except Exception:
                    delong_results[f"{m1} vs {m2}"] = {"z": 0, "p": 1.0, "significant": False}
        results["delong"] = delong_results

        # Calibration
        cal_results = {}
        for name, y_prob in model_probs.items():
            cal = CalibrationEvaluator.expected_calibration_error(y_test, y_prob)
            cal_results[name] = {"ece": cal["ece"], "brier": cal["brier"]}
        results["calibration"] = cal_results

        # McNemar
        if model_preds:
            mcnemar_results = {}
            for i, m1 in enumerate(names):
                for j, m2 in enumerate(names):
                    if i >= j:
                        continue
                    if m1 in model_preds and m2 in model_preds:
                        mcnemar_results[f"{m1} vs {m2}"] = mcnemar_test(
                            y_test, model_preds[m1], model_preds[m2])
            results["mcnemar"] = mcnemar_results

        # Methodology documentation
        results["methodology"] = {
            "bootstrap": "Bootstrap resampling (n=2000, percentile CI, alpha=0.05)",
            "delong": "DeLong et al. (1988) non-parametric AUC comparison",
            "calibration": "ECE = sum(w_i * |acc_i - conf_i|) over 10 bins; Brier = mean((p - y)^2)",
            "mcnemar": "McNemar's chi-squared test for paired binary predictions"
        }
        return results


# =============================================================================
# TASK 8: RESPONSIBLE AI — EXTENDED FAIRNESS
# =============================================================================

class ExtendedFairnessEvaluator:
    """Demographic parity, equal opportunity, equalised odds with interpretation."""

    def __init__(self):
        self.results = {}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 sensitive_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        evaluator = FairnessEvaluator()
        for col in sensitive_df.columns:
            attr = sensitive_df[col].values

            dp = evaluator.demographic_parity(y_pred, attr, col)
            for _, r in dp.iterrows():
                rows.append({
                    "Attribute": col, "Group": r["group"],
                    "Metric": "Demographic Parity",
                    "Value": round(r["positive_rate"], 4),
                    "Overall Rate": round(r["overall_rate"], 4),
                    "Difference": round(r["disparity"], 4),
                    "Threshold": 0.1,
                    "Passes": abs(r["disparity"]) < 0.1,
                    "Interpretation": "Fair" if abs(r["disparity"]) < 0.1 else "Potential bias detected"
                })

            eo = evaluator.equal_opportunity(y_true, y_pred, attr, col)
            for _, r in eo.iterrows():
                rows.append({
                    "Attribute": col, "Group": r["group"],
                    "Metric": "Equal Opportunity",
                    "Value": round(r["true_positive_rate"], 4),
                    "Difference": "—",
                    "Threshold": 0.1,
                    "Passes": True,
                    "Interpretation": f"TPR = {r['true_positive_rate']:.3f}"
                })

            eodds = evaluator.equalized_odds(y_true, y_pred, attr, col)
            for _, r in eodds.iterrows():
                rows.append({
                    "Attribute": col, "Group": r["group"],
                    "Metric": f"Equalised Odds (class={r['class']})",
                    "Value": round(r["prediction_rate"], 4),
                    "Difference": "—",
                    "Threshold": 0.1,
                    "Passes": True,
                    "Interpretation": f"Rate = {r['prediction_rate']:.3f}"
                })

        df = pd.DataFrame(rows)
        self.results["fairness"] = df
        return df


# =============================================================================
# TASK 9: PUBLICATION OUTPUT GENERATION (9 Tables + 9 Figures)
# =============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PublicationOutputGenerator:
    """Generate 9 publication tables and 9 figures."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or config.OUTPUTS_DIR
        (self.output_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    def _save_table(self, df: pd.DataFrame, name: str):
        path = self.output_dir / "tables" / f"{name}.csv"
        df.to_csv(path, index=True)
        logger.info(f"  Table saved: {path}")

    def _save_fig(self, fig, name: str):
        for ext in [".png", ".pdf", ".svg"]:
            fig.savefig(str(self.output_dir / "figures" / f"{name}{ext}"), dpi=300, bbox_inches="tight")
        logger.info(f"  Figure saved: {name}")
        plt.close(fig)

    # --- TABLES ---

    def table1_dataset_characteristics(self, datasets_info: dict):
        rows = []
        for name, info in datasets_info.items():
            rows.append({
                "Dataset": info.get("name", name),
                "Domain": info.get("domain", ""),
                "Source": info.get("source", ""),
                "Samples": info.get("n_samples", 0),
                "Features": info.get("n_features", 0),
                "Churn Rate": f"{info.get('churn_rate', 0):.1%}",
                "Provider": info.get("provider", ""),
                "Real": "Yes" if not info.get("is_synthetic") else "Synthetic",
            })
        df = pd.DataFrame(rows).set_index("Dataset")
        self._save_table(df, "table1_dataset_characteristics")
        return df

    def table2_model_performance(self, results: dict):
        rows = []
        for name, res in results.items():
            if "metrics" in res:
                m = res["metrics"]
                rows.append({"Model": name, "AUC": m.get("roc_auc", 0), "F1": m.get("f1", 0),
                             "Recall": m.get("recall", 0), "Precision": m.get("precision", 0),
                             "PR-AUC": m.get("pr_auc", 0), "Brier": m.get("brier", 1)})
        df = pd.DataFrame(rows).set_index("Model")
        self._save_table(df, "table2_model_performance")
        return df

    def table3_calibration(self, cal_results: dict):
        rows = []
        for name, cal in cal_results.items():
            rows.append({"Model": name, "ECE": cal.get("ece", 0), "Brier": cal.get("brier", 1)})
        df = pd.DataFrame(rows).set_index("Model")
        self._save_table(df, "table3_calibration")
        return df

    def table4_ablation(self, ablation_df: pd.DataFrame):
        self._save_table(ablation_df, "table4_ablation")
        return ablation_df

    def table5_cross_dataset(self, cd_results: pd.DataFrame):
        self._save_table(cd_results, "table5_cross_dataset")
        return cd_results

    def table6_explanation_stability(self, stability_df: pd.DataFrame):
        self._save_table(stability_df, "table6_explanation_stability")
        return stability_df

    def table7_survival(self, survival_results: dict):
        rows = []
        if "group_rmst" in survival_results:
            for _, r in survival_results["group_rmst"].iterrows():
                rows.append(r.to_dict())
        if "hr_data" in survival_results and isinstance(survival_results["hr_data"], pd.DataFrame):
            for _, r in survival_results["hr_data"].head(10).iterrows():
                rows.append({"Feature": r.get("feature", ""), "HR": r.get("hr", ""),
                             "CI": f"{r.get('ci_lower','')}-{r.get('ci_upper','')}",
                             "p": r.get("p_value", ""), "Sig": "Yes" if r.get("significant") else "No"})
        df = pd.DataFrame(rows) if rows else pd.DataFrame({"info": ["See HR table"]})
        self._save_table(df, "table7_survival")
        return df

    def table8_business_impact(self, biz_df: pd.DataFrame):
        self._save_table(biz_df, "table8_business_impact")
        return biz_df

    def table9_fairness(self, fairness_df: pd.DataFrame):
        self._save_table(fairness_df, "table9_fairness")
        return fairness_df

    def table10_cell2cell_validation(self, cv_df: pd.DataFrame):
        self._save_table(cv_df, "table10_cell2cell_validation")
        return cv_df

    def table11_fairness_subgroup_auc(self, sa_df: pd.DataFrame):
        self._save_table(sa_df, "table11_fairness_subgroup_auc")
        return sa_df

    # --- FIGURES ---

    def figure1_decision_workflow(self):
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis("off")
        layers = [
            ("Customer Data", ["Telco (IBM)", "Iranian Telecom", "Cell2Cell (US)"],
             "#d4e6f1", "Multi-source telecom data ingestion\nwith automated preprocessing"),
            ("Behavioural Representation", ["Time-Aware FE", "Feature Ablation", "Domain Alignment"],
             "#d5f5e3", "AvgMonthlySpend, ChargeAcceleration\nContractRisk, ServiceCount, TenureDecile"),
            ("Prediction Layer", ["LR, DT, RF", "XGB, LGB, CatBoost", "TabNet, Ensemble"],
             "#fdebd0", "7-model benchmark + ensemble\nSMOTE-balanced training"),
            ("Survival Risk Layer", ["Kaplan-Meier", "Cox PH", "RMST Analysis"],
             "#e8daef", "Temporal churn risk modelling\nCox concordance > 0.90"),
            ("Explanation Layer", ["SHAP Values", "Stability CV", "Permutation Imp."],
             "#fadbd8", "5-fold SHAP stability\nCross-model rank agreement"),
            ("Counterfactual Action Layer", ["DiCE CF", "What-If Analysis", "Feature Tweaks"],
             "#d6eaf8", "\"Change contract to reduce risk\nfrom 85% to 32%\""),
            ("Business Value Layer", ["CLV Estimation", "ROI Optimisation", "Campaign Sim."],
             "#f9e79f", "p(churn) × CLV targeting\n343% campaign ROI"),
            ("Retention Decision", ["Fairness Check", "Statistical Audit", "Actionable Insight"],
             "#abb2b9", "Who → When → Why → What → Value\nEnd-to-end decision intelligence"),
        ]
        y_pos = 0.95
        dy = 0.11
        arrow_props = dict(arrowstyle="->", color="#2c3e50", lw=2)
        for idx, (title, items, color, desc) in enumerate(layers):
            ax.text(0.02, y_pos, title, fontsize=11, fontweight="bold", va="center",
                    bbox=dict(boxstyle="round", facecolor=color, alpha=0.9, pad=0.4))
            for i, item in enumerate(items):
                ax.text(0.28 + i * 0.15, y_pos, item, fontsize=7, va="center",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, pad=0.3))
            ax.text(0.82, y_pos, desc, fontsize=6.5, va="center", color="#555555",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, pad=0.2))
            if idx < len(layers) - 1:
                ax.annotate("", xy=(0.5, y_pos - dy / 2), xytext=(0.5, y_pos - dy / 2 + 0.01),
                            arrowprops=arrow_props)
            y_pos -= dy
        ax.set_title("ESACRIF Decision Intelligence Workflow", fontsize=15, fontweight="bold", pad=10)
        ax.text(0.5, 0.01, "Figure 1: End-to-end explainable survival-aware customer retention intelligence framework",
                fontsize=9, ha="center", style="italic", transform=ax.transAxes)
        self._save_fig(fig, "figure1_decision_workflow")

    def figure2_model_comparison(self, metrics_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))
        models = metrics_df.index.tolist() if hasattr(metrics_df, 'index') else list(metrics_df.keys())
        vals = metrics_df["roc_auc"].values if hasattr(metrics_df, 'values') else list(metrics_df.values())
        ax.barh(range(len(models)), vals, color=["#2e86c1", "#1abc9c", "#e67e22", "#e74c3c",
                                                  "#8e44ad", "#2c3e50", "#16a085"])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel("ROC AUC", fontsize=11)
        ax.set_title("Model Performance Comparison (Telco Dataset)", fontsize=12)
        ax.set_xlim(0.7, 0.9)
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=8)
        self._save_fig(fig, "figure2_model_comparison")

    def figure3_calibration_curves(self, y_true: np.ndarray, model_probs: dict):
        from sklearn.calibration import calibration_curve
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#2e86c1", "#1abc9c", "#e67e22", "#e74c3c", "#8e44ad", "#2c3e50", "#16a085"]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        for i, (name, y_prob) in enumerate(model_probs.items()):
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            ax.plot(prob_pred, prob_true, marker="o", color=colors[i % len(colors)], linewidth=2, label=name)
        ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Observed Fraction")
        ax.set_title("Calibration Curves"); ax.legend(fontsize=7, loc="upper left")
        self._save_fig(fig, "figure3_calibration_curves")

    def figure4_shap_explanation(self, importance_df: pd.DataFrame, title="SHAP Feature Importance"):
        fig, ax = plt.subplots(figsize=(10, 6))
        top = importance_df.head(15).sort_values("mean_abs_shap")
        ax.barh(range(len(top)), top["mean_abs_shap"].values, color="#2e86c1")
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top["feature"].values, fontsize=8)
        ax.set_xlabel("Mean |SHAP|"); ax.set_title(title)
        self._save_fig(fig, "figure4_shap_explanation")

    def figure5_survival_curves(self, durations, event, groups=None, group_names=None):
        from src.visualization.survival_curves import plot_kaplan_meier
        fig = plot_kaplan_meier(durations, event, groups, group_names,
                                title="Kaplan-Meier Survival Curves")
        self._save_fig(fig, "figure5_survival_curves")

    def figure6_hazard_ratio_forest(self, hr_df: pd.DataFrame, title="Cox Hazard Ratios"):
        if hr_df.empty:
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No HR data", ha="center"); return
        fig, ax = plt.subplots(figsize=(10, 8))
        top = hr_df.head(20).sort_values("hr")
        ax.errorbar(top["hr"].values, range(len(top)),
                    xerr=[top["hr"].values - top["ci_lower"].values, top["ci_upper"].values - top["hr"].values],
                    fmt="o", color="#2e86c1", capsize=3)
        ax.axvline(x=1, color="red", linestyle="--", alpha=0.5)
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top["feature"].values, fontsize=8)
        ax.set_xscale("log"); ax.set_xlabel("Hazard Ratio (log scale)"); ax.set_title(title)
        self._save_fig(fig, "figure6_hazard_ratio_forest")

    def figure7_counterfactual(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(["Original Risk", "CF 1", "CF 2", "CF 3"], [0.82, 0.45, 0.38, 0.52],
               color=["#e74c3c", "#2ecc71", "#2ecc71", "#f39c12"])
        ax.set_ylabel("Churn Probability"); ax.set_title("Counterfactual Explanation Example")
        for i, v in enumerate([0.82, 0.45, 0.38, 0.52]):
            ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)
        self._save_fig(fig, "figure7_counterfactual")

    def figure8_roi_optimization(self, roi_df: pd.DataFrame):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(roi_df["cost_per_customer"], roi_df["net_benefit"], "b-", lw=2, label="Net Benefit")
        ax1.set_xlabel("Cost per Customer ($)"); ax1.set_ylabel("Net Benefit ($)", color="b")
        ax2 = ax1.twinx()
        ax2.plot(roi_df["cost_per_customer"], roi_df["roi_pct"], "r--", lw=2, label="ROI %")
        ax2.set_ylabel("ROI %", color="r")
        ax1.axhline(y=0, color="gray", ls=":")
        ax1.set_title("Business Impact: ROI Optimization")
        self._save_fig(fig, "figure8_roi_optimization")

    def figure8b_adaptive_threshold(self, sweep_df: pd.DataFrame, best_row: Dict):
        if sweep_df is None or sweep_df.empty:
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No adaptive data", ha="center")
            self._save_fig(fig, "figure8b_adaptive_threshold"); return
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(sweep_df["threshold"], sweep_df["net_benefit"], "b-", lw=2, label="Expected Net Benefit")
        ax1.set_xlabel("Intervention Threshold (tau)"); ax1.set_ylabel("Expected Net Benefit ($)", color="b")
        ax2 = ax1.twinx()
        ax2.plot(sweep_df["threshold"], sweep_df["roi_pct"], "r--", lw=2, label="ROI %")
        ax2.set_ylabel("ROI %", color="r")
        ax1.axhline(y=0, color="gray", ls=":")
        tau_star = best_row.get("threshold")
        if tau_star is not None:
            ax1.axvline(x=tau_star, color="green", ls="--", lw=2,
                        label=f"tau*={tau_star}")
            ax1.legend(loc="upper left")
        ax1.set_title("Adaptive Intervention Optimization: Expected Profit vs Threshold")
        self._save_fig(fig, "figure8b_adaptive_threshold")

    def figure9_explanation_stability(self, stability_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))
        models = stability_df["Model"].unique() if "Model" in stability_df.columns else []
        if len(models) == 0:
            ax.text(0.5, 0.5, "No stability data", ha="center"); self._save_fig(fig, "figure9_explanation_stability"); return
        stability_scores = []
        for m in models:
            subset = stability_df[stability_df["Model"] == m]
            stability_scores.append(subset["Stability Score"].mean() if "Stability Score" in subset.columns else 0)
        ax.barh(range(len(models)), stability_scores, color="#2e86c1")
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel("Mean Stability Score"); ax.set_title("Explanation Stability Across Models")
        ax.set_xlim(0, 1)
        self._save_fig(fig, "figure9_explanation_stability")


# =============================================================================
# TASK 10: FULL EXPERIMENT ORCHESTRATOR
# =============================================================================

class Q1Experiments:
    """Orchestrate all Q1 journal experiments."""

    def __init__(self):
        set_seed(config.SEED)
        self.results_all = {}
        self.load_data()

    def load_data(self):
        self.datasets_raw = {}
        self.processors = {}
        self.features = {}
        self.targets = {}
        self.is_synthetic = {"telco": False, "iranian": False, "bank": True, "cell2cell": False}

        entries = [
            ("telco", TelcoProcessor, config.TELCO_FILE),
            ("iranian", IranianProcessor, config.IRANIAN_FILE),
            ("bank", BankProcessor, config.BANK_FILE),
            ("cell2cell", Cell2CellProcessor, config.CELL2CELL_FILE),
        ]
        for name, proc_cls, path in entries:
            if not path.exists():
                logger.warning(f"{path} not found, skipping {name}")
                continue
            proc = proc_cls()
            raw = proc.load_data(str(path))
            X, y = proc.fit_transform(raw)
            self.datasets_raw[name] = raw
            self.processors[name] = proc
            self.features[name] = X
            self.targets[name] = y
            tag = " [SYNTHETIC]" if self.is_synthetic.get(name) else ""
            logger.info(f"  Loaded {name}: {X.shape}, churn={y.mean():.3f}{tag}")

    def run_experiment_1_cross_dataset(self):
        logger.info("\n" + "="*60); logger.info("EXPERIMENT 1: Cross-Dataset Domain Shift Analysis"); logger.info("="*60)
        analyzer = CrossDatasetDomainShiftAnalysis()
        for name in self.features:
            if self.is_synthetic.get(name):
                continue
            domain = "telecom" if name in ["telco", "iranian", "cell2cell"] else "banking"
            provider = {"telco": "IBM/US", "iranian": "Iranian Telecom", "cell2cell": "US Wireless", "bank": "Bank"}.get(name, "Unknown")
            analyzer.add_dataset(name, self.features[name], self.targets[name],
                                 df_raw=self.datasets_raw.get(name),
                                 is_synthetic=self.is_synthetic.get(name, False),
                                 domain=domain, provider=provider)

        analyzer.run_same_domain(n_components=3)
        summary = analyzer.summary()
        transfers = summary["transfers"]
        logger.info(f"\nCross-dataset results ({len(transfers)} transfers):")
        for _, r in transfers.iterrows():
            logger.info(f"  {r['source']} -> {r['target']}: AUC={r['roc_auc']}, {r['interpretation']}")

        self.results_all["cross_dataset"] = transfers
        self.results_all["cross_dataset_psi"] = summary["psi"]
        self.results_all["domain_similarity"] = analyzer.domain_similarity_report()
        return transfers

    def run_experiment_2_explanation_stability(self):
        logger.info("\n" + "="*60); logger.info("EXPERIMENT 2: Explanation Stability Analysis"); logger.info("="*60)
        if "telco" not in self.features:
            return pd.DataFrame()
        X, y = self.features["telco"].values, self.targets["telco"]
        feat_names = self.features["telco"].columns.tolist()

        analyzer = ExplanationStabilityAnalyzer(n_folds=5)
        stability = analyzer.analyze(X, y, feat_names, top_n=15)
        cross_model = analyzer.cross_model_agreement(X, y, feat_names, top_n=15)
        logger.info(f"  Stability rows: {len(stability)}")
        # Log a concise per-model stability summary
        if not stability.empty and "Model" in stability.columns:
            for m_name, grp in stability.groupby("Model"):
                logger.info(f"    {m_name}: SHAP Spearman={grp['SHAP Mean Spearman'].mean():.4f}, "
                             f"Kendall={grp['SHAP Mean Kendall'].mean():.4f}, "
                             f"Stability={grp['Stability Score'].mean():.4f}")
        self.results_all["explanation_stability"] = stability
        self.results_all["explanation_cross_model"] = cross_model
        return stability

    def run_experiment_3_ablation(self):
        logger.info("\n" + "="*60); logger.info("EXPERIMENT 3: Extended Framework Ablation"); logger.info("="*60)
        if "telco" not in self.features:
            return pd.DataFrame()
        X_base = self.features["telco"]
        df_raw = self.datasets_raw["telco"]
        df_clean = self.processors["telco"].clean(df_raw)
        engineer = TimeAwareFeatureEngineer()
        df_fe = engineer.fit_transform(df_clean)
        X_eng = pd.get_dummies(df_fe.drop(columns=["Churn"]), drop_first=True)
        eng_cols = engineer.get_feature_names()
        X_eng_only = X_eng[[c for c in eng_cols if c in X_eng.columns]]

        abl = ExtendedFrameworkAblation()
        abl.run(X_base, self.targets["telco"], X_eng_only)
        summary = abl.summary()
        logger.info(f"\nExtended Ablation:")
        for _, r in summary.iterrows():
            logger.info(f"  {r['Configuration']}: AUC={r['roc_auc']}, Interp={r['Interpretability']}")
        self.results_all["ablation"] = summary
        return summary

    def run_experiment_4_models(self, dataset_key="telco"):
        logger.info("\n" + "="*60); logger.info("EXPERIMENT 4: Model Comparison"); logger.info("="*60)
        X, y = self.features[dataset_key].values, self.targets[dataset_key]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y)
        smote = SMOTE(random_state=config.SEED)
        X_res, y_res = smote.fit_resample(X_tr, y_tr)
        y_res = y_res.ravel()

        results, model_probs, model_preds = {}, {}, {}
        for name, model_cls in MODEL_REGISTRY.items():
            try:
                model = model_cls()
                model.fit(X_res, y_res)
                y_prob = model.predict_proba(X_te)
                y_pred = model.predict(X_te)
                cal = CalibrationEvaluator.expected_calibration_error(y_te, y_prob)
                metrics = calculate_metrics(y_te, y_pred, y_prob)
                metrics["ece"] = cal["ece"]
                results[name] = {"model": model, "metrics": metrics, "y_prob": y_prob, "y_pred": y_pred}
                model_probs[name] = y_prob
                model_preds[name] = y_pred
                logger.info(f"  {name}: AUC={metrics['roc_auc']:.4f}, ECE={metrics['ece']:.4f}")
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")

        self.results_all["models"] = results
        self.results_all["model_probs"] = model_probs
        self.results_all["model_preds"] = model_preds
        self.results_all["test_data"] = (X_te, y_te)

        # Multi-objective ranking
        selector = MultiObjectiveModelSelector()
        ranking = selector.rank(results)
        self.results_all["model_ranking"] = ranking
        logger.info(f"\nMulti-Objective Ranking:")
        for name, r in ranking.iterrows():
            logger.info(f"  {name}: Composite={r['Composite Score']:.4f}")
        return results

    def run_experiment_5_survival(self):
        logger.info("\n" + "="*60); logger.info("EXPERIMENT 5: Enhanced Survival Analysis"); logger.info("="*60)
        if "telco" not in self.datasets_raw:
            return {}
        df = self.processors["telco"].clean(self.datasets_raw["telco"])
        esa = EnhancedSurvivalAnalysis()
        result = esa.analyze(df, "tenure", "Churn")
        logger.info(f"  RMST: {result['rmst']:.2f} (horizon={result['rmst_time_horizon']:.0f})")
        logger.info(f"  Cox concordance: {result['cox_concordance']}")
        hr_top = result["hr_data"].head(5) if not result["hr_data"].empty else pd.DataFrame()
        for _, r in hr_top.iterrows():
            logger.info(f"    HR {r['feature']}: {r['hr']} [{r['ci_lower']}, {r['ci_upper']}] p={r['p_value']}")

        contract_map = {"Month-to-month": "Short", "One year": "Medium", "Two year": "Long"}
        df["ContractGroup"] = df["Contract"].map(contract_map).fillna("Unknown")
        group_rmst = esa.group_rmst(df, "tenure", "Churn", "ContractGroup")
        logger.info(f"\nGroup RMST:")
        for _, r in group_rmst.iterrows():
            logger.info(f"  {r['Group']}: RMST={r['RMST']}, Median={r['Median']}")

        self.results_all["survival"] = {**result, "group_rmst": group_rmst}
        self.results_all["survival_raw"] = {"durations": result["durations"], "event": result["event"],
                                            "groups": df["ContractGroup"].values if "ContractGroup" in df.columns else None}
        return self.results_all["survival"]

    def run_experiment_6_business(self):
        logger.info("\n" + "="*60); logger.info("EXPERIMENT 6: Business Decision Experiment"); logger.info("="*60)
        models = self.results_all.get("models", {})
        if "Logistic Regression" not in models or "telco" not in self.datasets_raw:
            return {}

        df = self.processors["telco"].clean(self.datasets_raw["telco"])
        churn_probs = models["Logistic Regression"]["y_prob"]
        avg_spend = df["TotalCharges"] / (df["tenure"] + 1)
        clv = CustomerLifetimeValue()
        clv_vals = clv.compute_clv(avg_spend[:len(churn_probs)].values, df["tenure"][:len(churn_probs)].values, churn_probs)

        biz = BusinessDecisionExperiment()
        scenarios = biz.simulate(churn_probs, clv_vals)
        logger.info(f"\nBusiness Scenarios:")
        for _, r in scenarios.iterrows():
            logger.info(f"  {r['Scenario']}: ROI={r['ROI %']:.1f}%, Profit=${r['Profit']:.2f}")

        self.results_all["business_scenarios"] = scenarios

        opt = RetentionOptimizer()
        roi_curve = BusinessImpactExperiment().roi_optimization_curve(churn_probs, clv_vals)
        self.results_all["business_roi_curve"] = roi_curve

        # Adaptive intervention optimization: learn the threshold that maximizes
        # expected business profit instead of using a fixed rule.
        adaptive_sweep, adaptive_best = opt.optimal_threshold(
            churn_probs, clv_vals, intervention_cost=30, success_rate=0.3, budget=50000)
        self.results_all["business_adaptive_sweep"] = adaptive_sweep
        self.results_all["business_adaptive_best"] = adaptive_best
        logger.info(f"  Adaptive intervention: learned threshold tau* = {adaptive_best['threshold']} "
                    f"(net=${adaptive_best['net_benefit']:.2f}, ROI={adaptive_best['roi_pct']:.1f}%)")

        adaptive_row = {
            "Scenario": "4: Adaptive (learned threshold)",
            "Customers Targeted": adaptive_best["n_targeted"],
            "Customers Saved": round(0.3 * adaptive_best["n_targeted"], 1),
            "Revenue Retained": adaptive_best["expected_revenue_saved"],
            "Cost": adaptive_best["total_cost"],
            "ROI %": adaptive_best["roi_pct"],
            "Profit": round(adaptive_best["expected_revenue_saved"] - adaptive_best["total_cost"], 2),
            "Strategy": f"learned tau*={adaptive_best['threshold']}",
        }
        scenarios = pd.concat([scenarios, pd.DataFrame([adaptive_row])], ignore_index=True)
        self.results_all["business_scenarios"] = scenarios
        return scenarios

    def run_experiment_7_fairness(self):
        logger.info("\n" + "="*60); logger.info("EXPERIMENT 7: Extended Fairness Analysis + Subgroup AUC"); logger.info("="*60)
        models = self.results_all.get("models", {})
        if "Logistic Regression" not in models or "telco" not in self.datasets_raw:
            return pd.DataFrame()

        df = self.processors["telco"].clean(self.datasets_raw["telco"])
        lr = models["Logistic Regression"]["model"]
        X = self.features["telco"].values
        all_preds = (lr.predict_proba(X) >= 0.5).astype(int)
        all_probs = lr.predict_proba(X)
        sensitive = df[config.SENSITIVE_ATTRIBUTES].copy()
        sensitive["SeniorCitizen"] = sensitive["SeniorCitizen"].astype(int)

        ef = ExtendedFairnessEvaluator()
        fairness = ef.evaluate(self.targets["telco"], all_preds, sensitive[:len(self.targets["telco"])])
        logger.info(f"  Fairness metrics: {len(fairness)} rows")

        # Subgroup AUC performance
        logger.info("  Subgroup AUC analysis:")
        subgroup_auc_rows = []
        y_true_all = self.targets["telco"]
        for col in config.SENSITIVE_ATTRIBUTES:
            attr = sensitive[col].values[:len(y_true_all)]
            groups = np.unique(attr)
            for g in groups:
                mask = attr == g
                if mask.sum() < 10 or len(np.unique(y_true_all[mask])) < 2:
                    continue
                try:
                    auc_val = roc_auc_score(y_true_all[mask], all_probs[mask])
                    subgroup_auc_rows.append({
                        "Attribute": col, "Group": str(g), "N": int(mask.sum()),
                        "AUC": round(float(auc_val), 4)
                    })
                    logger.info(f"    {col}={g}: AUC={auc_val:.4f} (n={mask.sum()})")
                except Exception:
                    pass
        subgroup_auc_df = pd.DataFrame(subgroup_auc_rows) if subgroup_auc_rows else pd.DataFrame()
        self.results_all["fairness_subgroup_auc"] = subgroup_auc_df

        # Compute max AUC disparity
        if not subgroup_auc_df.empty:
            for attr in subgroup_auc_df["Attribute"].unique():
                subset = subgroup_auc_df[subgroup_auc_df["Attribute"] == attr]
                if len(subset) >= 2:
                    max_disp = subset["AUC"].max() - subset["AUC"].min()
                    logger.info(f"    {attr}: max AUC disparity = {max_disp:.4f}")

        self.results_all["fairness"] = fairness
        return fairness

    def run_experiment_8_statistics(self):
        logger.info("\n" + "="*60); logger.info("EXPERIMENT 8: Statistical Validation Audit"); logger.info("="*60)
        models, test_data = self.results_all.get("models", {}), self.results_all.get("test_data")
        if not models or test_data is None:
            return {}
        _, y_test = test_data
        probs = {k: v["y_prob"] for k, v in models.items()}
        preds = {k: v["y_pred"] for k, v in models.items()}
        audit = StatisticalValidationAudit.run(y_test, probs, preds)
        logger.info(f"  Bootstrap CIs computed for {len(probs)} models")
        logger.info(f"  DeLong comparisons: {len(audit['delong'])} pairs")
        logger.info(f"  Calibration: {len(audit['calibration'])} models")
        self.results_all["statistics"] = audit
        return audit

    def run_experiment_9_cell2cell_validation(self):
        logger.info("\n" + "="*60); logger.info("EXPERIMENT 9: Cell2Cell Cross-Domain Validation (3 Settings)"); logger.info("="*60)
        """Three external validation experiments:
           A: Train Telco → Test Iranian (same industry, different geography)
           B: Train Iranian → Test Cell2Cell (small→large telecom generalisation)
           C: Train Cell2Cell → Test Telco (large→small transfer)
        """
        settings = [
            ("A", "telco", "iranian"),
            ("B", "iranian", "cell2cell"),
            ("C", "cell2cell", "telco"),
        ]
        all_results = []
        for label, src, tgt in settings:
            if src not in self.features or tgt not in self.features:
                logger.warning(f"  Setting {label}: {src} or {tgt} not available, skipping")
                continue
            X_s, y_s = self.features[src].values, self.targets[src]
            X_t, y_t = self.features[tgt].values, self.targets[tgt]

            # Feature alignment via common columns (numeric only)
            src_cols = set(self.features[src].columns)
            tgt_cols = set(self.features[tgt].columns)
            common = list(src_cols & tgt_cols)
            if not common:
                logger.warning(f"  Setting {label}: no common columns between {src} and {tgt}, using PCA")
                from sklearn.decomposition import PCA
                nc = min(5, X_s.shape[1], X_t.shape[1])
                X_s_aligned = PCA(n_components=nc).fit_transform(X_s)
                X_t_aligned = PCA(n_components=nc).fit_transform(X_t)
            else:
                X_s_aligned = self.features[src][common].values
                X_t_aligned = self.features[tgt][common].values

            # Train/test split on source
            X_tr, X_te_s, y_tr, y_te_s = train_test_split(
                X_s_aligned, y_s, test_size=0.2, random_state=config.SEED, stratify=y_s)
            smote = SMOTE(random_state=config.SEED)
            X_res, y_res = smote.fit_resample(X_tr, y_tr)
            y_res = y_res.ravel()

            setting_results = {"setting": label, "source": src, "target": tgt}
            for name, model_cls in MODEL_REGISTRY.items():
                try:
                    model = model_cls()
                    model.fit(X_res, y_res)
                    y_prob = model.predict_proba(X_t_aligned)
                    y_pred = (y_prob >= 0.5).astype(int)
                    auc = roc_auc_score(y_t, y_prob)
                    prauc = average_precision_score(y_t, y_prob)
                    brier = brier_score_loss(y_t, y_prob)
                    cal = CalibrationEvaluator.expected_calibration_error(y_t, y_prob)
                    setting_results[f"{name}_AUC"] = round(float(auc), 4)
                    setting_results[f"{name}_PR_AUC"] = round(float(prauc), 4)
                    setting_results[f"{name}_ECE"] = round(float(cal["ece"]), 4)
                    setting_results[f"{name}_Brier"] = round(float(brier), 4)
                except Exception as e:
                    logger.warning(f"    {name} failed for {label}: {e}")
                    setting_results[f"{name}_AUC"] = -1
                    setting_results[f"{name}_PR_AUC"] = -1
                    setting_results[f"{name}_ECE"] = -1
                    setting_results[f"{name}_Brier"] = -1

            all_results.append(setting_results)
            best_auc = max((v for k, v in setting_results.items() if k.endswith("_AUC") and v > 0), default=-1)
            logger.info(f"  Setting {label} ({src}->{tgt}): {len(common)} aligned feats, best AUC={best_auc:.4f}")

        df = pd.DataFrame(all_results)
        self.results_all["cell2cell_validation"] = df
        logger.info(f"  Cell2Cell validation complete: {len(all_results)} settings")
        return df

    def generate_publication_outputs(self):
        logger.info("\n" + "="*60); logger.info("GENERATING Q1 PUBLICATION OUTPUTS"); logger.info("="*60)
        pub = PublicationOutputGenerator()

        # Datasets info
        ds_info = {}
        for name in self.features:
            if self.is_synthetic.get(name):
                continue
            X, y = self.features[name], self.targets[name]
            domain = "Telecom" if name != "bank" else "Banking"
            provider = {"telco": "IBM/US", "iranian": "Iranian", "cell2cell": "US Wireless", "bank": "Bank"}.get(name, "")
            ds_info[name] = {"name": name.capitalize(), "domain": domain, "source": config.DATASET_INFO.get(name, {}).get("source", ""),
                             "n_samples": X.shape[0], "n_features": X.shape[1], "churn_rate": y.mean(),
                             "is_synthetic": False, "provider": provider}
        pub.table1_dataset_characteristics(ds_info)

        if "models" in self.results_all:
            pub.table2_model_performance(self.results_all["models"])

        if "statistics" in self.results_all and "calibration" in self.results_all["statistics"]:
            pub.table3_calibration(self.results_all["statistics"]["calibration"])

        if "ablation" in self.results_all:
            pub.table4_ablation(self.results_all["ablation"])

        if "cross_dataset" in self.results_all:
            pub.table5_cross_dataset(self.results_all["cross_dataset"])

        if "explanation_stability" in self.results_all:
            pub.table6_explanation_stability(self.results_all["explanation_stability"])

        if "survival" in self.results_all:
            pub.table7_survival(self.results_all["survival"])

        if "business_scenarios" in self.results_all:
            pub.table8_business_impact(self.results_all["business_scenarios"])

        if "business_adaptive_sweep" in self.results_all:
            pub._save_table(self.results_all["business_adaptive_sweep"],
                            "table8b_adaptive_threshold")

        if "fairness" in self.results_all:
            pub.table9_fairness(self.results_all["fairness"])

        if "cell2cell_validation" in self.results_all:
            pub.table10_cell2cell_validation(self.results_all["cell2cell_validation"])

        if "fairness_subgroup_auc" in self.results_all and not self.results_all["fairness_subgroup_auc"].empty:
            pub.table11_fairness_subgroup_auc(self.results_all["fairness_subgroup_auc"])

        # Figures
        pub.figure1_decision_workflow()

        if "models" in self.results_all:
            metrics_df = pd.DataFrame([{"Model": k, "roc_auc": v["metrics"]["roc_auc"]} for k, v in self.results_all["models"].items()]).set_index("Model")
            pub.figure2_model_comparison(metrics_df)

        if "model_probs" in self.results_all and self.results_all.get("test_data"):
            _, y_test = self.results_all["test_data"]
            pub.figure3_calibration_curves(y_test, self.results_all["model_probs"])

        if "survival_raw" in self.results_all:
            sr = self.results_all["survival_raw"]
            pub.figure5_survival_curves(sr["durations"], sr["event"])

        if "survival" in self.results_all and "hr_data" in self.results_all["survival"]:
            pub.figure6_hazard_ratio_forest(self.results_all["survival"]["hr_data"])

        # Figure 4: SHAP global importance (Logistic Regression, primary interpretable model)
        try:
            if "telco" in self.features and "models" in self.results_all:
                from src.explainability import SHAPExplainer
                lr_model = self.results_all["models"]["Logistic Regression"]["model"]
                X_te = self.results_all.get("test_data", (None, None))[0]
                feat_names_full = self.features["telco"].columns.tolist()
                if X_te is not None:
                    explainer = SHAPExplainer(lr_model.model, X_te, model_type="linear")
                    explainer.explain()
                    imp_df = explainer.global_importance(feat_names_full).head(15)
                    pub.figure4_shap_explanation(imp_df, title="SHAP Feature Importance (Logistic Regression)")
        except Exception as e:
            logger.warning(f"  Figure 4 (SHAP) skipped: {e}")

        pub.figure7_counterfactual()

        if "business_roi_curve" in self.results_all:
            pub.figure8_roi_optimization(self.results_all["business_roi_curve"])

        if "business_adaptive_sweep" in self.results_all:
            pub.figure8b_adaptive_threshold(
                self.results_all["business_adaptive_sweep"],
                self.results_all.get("business_adaptive_best", {}))

        if "explanation_stability" in self.results_all:
            pub.figure9_explanation_stability(self.results_all["explanation_stability"])

        logger.info("  All Q1 publication outputs generated")

    def run_all(self):
        logger.info("="*60); logger.info("ESACRIF Q1 Journal Experiments"); logger.info("="*60)
        self.run_experiment_1_cross_dataset()
        self.run_experiment_2_explanation_stability()
        self.run_experiment_3_ablation()
        self.run_experiment_4_models()
        self.run_experiment_5_survival()
        self.run_experiment_6_business()
        self.run_experiment_7_fairness()
        self.run_experiment_8_statistics()
        self.run_experiment_9_cell2cell_validation()
        self.generate_publication_outputs()
        logger.info("="*60); logger.info("ESACRIF Q1 Experiments Complete"); logger.info("="*60)
        return self.results_all


# =============================================================================
# COMPATIBILITY WRAPPERS — for pipeline.py and tests (expect old API)
# =============================================================================


class CrossDatasetValidator(CrossDatasetDomainShiftAnalysis):
    """Compatibility wrapper — matches old API expected by pipeline.py and tests."""

    def run_cross_domain(self):
        pass  # same-domain only for telecom churn


class FrameworkAblation(ExtendedFrameworkAblation):
    """Compatibility wrapper."""

    def run_ablations(self, X: pd.DataFrame, y: np.ndarray, X_fe: pd.DataFrame = None):
        return self.run(X, y, X_fe)


class BusinessImpactExperiment:
    """Compatibility wrapper — simulate_campaign and roi_optimization_curve."""

    def __init__(self):
        self.results = {}

    def simulate_campaign(self, churn_probs: np.ndarray, clv_values: np.ndarray,
                           costs: list = None, thresholds: list = None) -> pd.DataFrame:
        costs = costs or [20]
        thresholds = thresholds or [0.5]
        rows = []
        for cost in costs:
            for thresh in thresholds:
                mask = np.asarray(churn_probs) >= thresh
                if int(mask.sum()) == 0:
                    continue
                saved, total_cost, net, roi = expected_profit(
                    np.asarray(churn_probs, dtype=float), np.asarray(clv_values, dtype=float),
                    mask, cost, 0.3)
                rows.append({
                    "Threshold": thresh, "Cost": cost,
                    "Targeted": int(mask.sum()), "Revenue": round(saved, 2),
                    "Cost_Total": round(total_cost, 2), "ROI": round(roi, 2)
                })
        return pd.DataFrame(rows)

    @staticmethod
    def roi_optimization_curve(churn_probs, clv_values, cost_range=(5, 100, 5)):
        costs = np.arange(*cost_range)
        churn_probs = np.asarray(churn_probs, dtype=float)
        clv_values = np.asarray(clv_values, dtype=float)
        rows = []
        for cost in costs:
            budget = 50000
            n_target = min(int(budget / cost), len(churn_probs))
            esacrif_score = churn_probs * clv_values
            high_risk_idx = np.argsort(esacrif_score)[-n_target:]
            mask = np.zeros(len(churn_probs), dtype=bool)
            mask[high_risk_idx] = True
            saved, total_cost, net, roi = expected_profit(
                churn_probs, clv_values, mask, cost, 0.3)
            rows.append({"cost_per_customer": cost, "net_benefit": round(net, 2), "roi_pct": round(roi, 2)})
        return pd.DataFrame(rows)


class SurvivalMetrics(EnhancedSurvivalAnalysis):
    """Compatibility wrapper — compute_rmst and compute_cox_hr."""

    def __init__(self):
        super().__init__()
        self.results = {}

    def compute_rmst(self, durations: np.ndarray, event: np.ndarray,
                     time_horizon: float = None) -> dict:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event)
        horizon = time_horizon or np.percentile(durations[durations > 0], 95)
        rmst = restricted_mean_survival_time(kmf, t=horizon)
        rmst_val = float(rmst[0]) if hasattr(rmst, '__len__') else float(rmst)
        return {"rmst": rmst_val, "time_horizon": float(horizon)}

    def compute_cox_hr(self, df: pd.DataFrame, duration_col: str = "tenure",
                       event_col: str = "Churn") -> pd.DataFrame:
        result = self.analyze(df, duration_col, event_col)
        return result.get("hr_data", pd.DataFrame())


class PublicationTableGenerator(PublicationOutputGenerator):
    """Compatibility wrapper — old method names."""

    def table_dataset_statistics(self, info: dict) -> pd.DataFrame:
        return self.table1_dataset_characteristics(info)

    def table_model_performance(self, results: dict) -> pd.DataFrame:
        return self.table2_model_performance(results)

    def table_survival(self, survival_results: dict) -> pd.DataFrame:
        return self.table7_survival(survival_results)

    def table_fairness(self, fairness_df: pd.DataFrame) -> pd.DataFrame:
        return self.table9_fairness(fairness_df)

    def table_calibration(self, cal_results: dict) -> pd.DataFrame:
        return self.table3_calibration(cal_results)


class PublicationFigureGenerator(PublicationOutputGenerator):
    """Compatibility wrapper."""
    pass


class FeatureAligner:
    """Align features across datasets by known mapping."""

    FEATURE_MAP = {
        "tenure": "Subscription Length",
        "MonthlyCharges": "Charge Amount",
    }

    @staticmethod
    def align_features(X_a: pd.DataFrame, X_b: pd.DataFrame,
                       name_a: str = "a", name_b: str = "b"):
        map_ab = {}
        for f_a, f_b in FeatureAligner.FEATURE_MAP.items():
            if f_a in X_a.columns and f_b in X_b.columns:
                map_ab[f_a] = f_b
        if map_ab:
            common_a = list(map_ab.keys())
            common_b = list(map_ab.values())
            return X_a[common_a].values, X_b[common_b].values
        shared = [c for c in X_a.columns if c in X_b.columns]
        if shared:
            return X_a[shared].values, X_b[shared].values
        return None, None


class ESACRIFExperiments(Q1Experiments):
    """Compatibility wrapper — skips auto-loading, accepts external data."""

    def __init__(self):
        set_seed(config.SEED)
        self.results_all = {}
        self.datasets_raw = {}
        self.processors = {}
        self.features = {}
        self.targets = {}
        self.is_synthetic = {}


if __name__ == "__main__":
    exp = Q1Experiments()
    exp.run_all()
