import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

import config
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed
from src.utils.metrics import calculate_metrics
from src.preprocessing import TelcoProcessor, IranianProcessor, BankProcessor
from src.preprocessing.cell2cell_processor import Cell2CellProcessor
from src.feature_engineering import TimeAwareFeatureEngineer, FeatureAblation
from src.prediction import MODEL_REGISTRY, EnsembleModel
from src.explainability import SHAPExplainer, PermutationImportance, ExplanationEvaluation
from src.survival import KaplanMeierAnalysis, CoxPHAnalysis
from src.counterfactual import DiCECounterfactual
from src.business import CustomerLifetimeValue, RetentionOptimizer
from src.fairness import FairnessEvaluator
from src.robustness import NoiseInjector, ShiftSimulator
from src.statistics import BootstrapInference, delong_roc_test, mcnemar_test, CalibrationEvaluator
from src.cross_dataset import CrossDatasetExperiment
from experiments import (
    CrossDatasetValidator, FrameworkAblation, BusinessImpactExperiment,
    SurvivalMetrics, PublicationTableGenerator, PublicationFigureGenerator,
    FeatureAligner, ESACRIFExperiments
)

logger = setup_logger("esacrif", log_file=str(config.OUTPUTS_DIR / "run.log"))


class ESACRIFPipeline:
    def __init__(self):
        set_seed(config.SEED)
        self.datasets = {}
        self.processors = {}
        self.features = {}
        self.targets = {}
        self.models = {}
        self.results = {}
        self.experiments_runner = None

    def load_all_datasets(self):
        logger.info("=== Loading Datasets ===")
        for name, (proc_cls, path) in {
            "telco": (TelcoProcessor, config.TELCO_FILE),
            "iranian": (IranianProcessor, config.IRANIAN_FILE),
            "bank": (BankProcessor, config.BANK_FILE),
            "cell2cell": (Cell2CellProcessor, config.CELL2CELL_FILE),
        }.items():
            if not path.exists():
                logger.warning(f"{path} not found, skipping {name}")
                continue
            is_synth = config.SYNTHETIC_DATASETS.get(name, False)
            proc = proc_cls()
            X, y = proc.fit_transform(proc.load_data(str(path)))
            self.datasets[name] = (X, y)
            self.processors[name] = proc
            self.features[name] = X
            self.targets[name] = y
            tag = " [SYNTHETIC]" if is_synth else ""
            logger.info(f"  {name}: {X.shape}, churn={y.mean():.3f}{tag}")

    def engineer_features_telco(self):
        if "telco" not in self.features:
            return
        logger.info("=== Feature Engineering (Telco) ===")
        df = self.processors["telco"].load_data(str(config.TELCO_FILE))
        df = self.processors["telco"].clean(df)
        engineer = TimeAwareFeatureEngineer()
        df_fe = engineer.fit_transform(df)
        fe_cols = engineer.get_feature_names()
        logger.info(f"  Created features: {fe_cols}")
        new_features = pd.get_dummies(df_fe, drop_first=True)
        y = df_fe["Churn"].values
        self.features["telco_fe"] = new_features.drop(columns=["Churn"], errors="ignore")
        self.targets["telco_fe"] = y
        self.feature_engineer = engineer

        X_base = self.features["telco"]
        X_eng = self.features["telco_fe"]
        y = self.targets["telco"]
        ablation = FeatureAblation()
        auc_with, auc_without = ablation.evaluate_engineered_features(
            X_eng.values, X_base.values, y
        )
        logger.info(f"  Ablation: AUC without eng features={auc_without:.4f}, with={auc_with:.4f}, delta={auc_with - auc_without:.4f}")

    def train_models(self, dataset_key: str = "telco"):
        if dataset_key not in self.features:
            logger.error(f"Dataset {dataset_key} not available")
            return
        is_synth = config.SYNTHETIC_DATASETS.get(dataset_key, False)
        tag = " [SYNTHETIC - excluded from primary results]" if is_synth else ""
        logger.info(f"=== Training Models on {dataset_key}{tag} ===")
        X = self.features[dataset_key].values
        y = self.targets[dataset_key]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
        )
        smote = SMOTE(random_state=config.SEED)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        y_train_res = y_train_res.ravel()

        self.test_data = (X_test, y_test)
        trained_models = {}
        for name, model_cls in MODEL_REGISTRY.items():
            try:
                model = model_cls()
                model.fit(X_train_res, y_train_res)
                y_prob = model.predict_proba(X_test)
                y_pred = model.predict(X_test)
                metrics = calculate_metrics(y_test, y_pred, y_prob)
                logger.info(f"  {name}: AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
                trained_models[name] = {"model": model, "metrics": metrics, "y_prob": y_prob, "y_pred": y_pred}
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")
        self.models[dataset_key] = trained_models
        return trained_models

    def survival_analysis(self, duration_col: str = "tenure", event_col: str = "Churn"):
        logger.info("=== Survival Analysis ===")
        if "telco" not in self.processors:
            return
        df = self.processors["telco"].load_data(str(config.TELCO_FILE))
        df = self.processors["telco"].clean(df)
        durations = df[duration_col].values
        event = df[event_col].values
        km = KaplanMeierAnalysis()
        km.fit(durations, event)
        median = km.kmf.median_survival_time_
        logger.info(f"  KM median survival: {median}")

        sm = SurvivalMetrics()
        rmst = sm.compute_rmst(durations, event)
        logger.info(f"  RMST: {rmst}")

        hr_df = sm.compute_cox_hr(df, duration_col, event_col)
        if isinstance(hr_df, pd.DataFrame) and "error" not in hr_df.columns:
            logger.info(f"  Cox concordance: {sm.results.get('cox_concordance', 'N/A')}")
            cols_to_log = hr_df.columns.tolist() if hasattr(hr_df, 'columns') else []
            for col_idx in range(min(5, len(hr_df))):
                r = hr_df.iloc[col_idx]
                hr_col = "hr" if "hr" in cols_to_log else "hazard_ratio"
                logger.info(f"    {r.get('feature', col_idx)}: HR={r.get(hr_col, 'N/A')} [{r.get('ci_lower', 'N/A')}, {r.get('ci_upper', 'N/A')}] p={r.get('p_value', 'N/A')}")

        self.survival_results = {"km": km, "durations": durations, "event": event, "rmst": rmst, "cox": hr_df}

    def cross_dataset_experiments(self):
        logger.info("=== Cross-Dataset Generalization ===")
        validator = CrossDatasetValidator()
        for name in self.features:
            if name == "telco_fe":
                continue
            validator.add_dataset(
                name, self.features[name], self.targets[name],
                is_synthetic=config.SYNTHETIC_DATASETS.get(name, False),
                domain="telecom" if name in ["telco", "iranian"] else "banking"
            )
        validator.run_same_domain(n_components=3)
        summary_data = validator.summary()
        transfers = summary_data.get("transfers", pd.DataFrame()) if isinstance(summary_data, dict) else summary_data
        logger.info(f"Cross-dataset results ({len(transfers)} transfers):")
        for _, r in transfers.iterrows():
            note = r.get("interpretation", r.get("note", ""))
            logger.info(f"  {r['source']} -> {r['target']}: AUC={r['roc_auc']}, aligned_features={r['aligned_features']}, {note}")
        self.cross_dataset_results = transfers

    def fairness_analysis(self):
        logger.info("=== Fairness Analysis ===")
        if "telco" not in self.features or "telco" not in self.models:
            return
        df = self.processors["telco"].load_data(str(config.TELCO_FILE))
        df = self.processors["telco"].clean(df)
        sensitive = df[config.SENSITIVE_ATTRIBUTES].copy()
        sensitive["SeniorCitizen"] = sensitive["SeniorCitizen"].astype(int)
        y_pred = self.models["telco"]["Logistic Regression"]["model"].predict(self.features["telco"].values)
        y_true = self.targets["telco"]
        evaluator = FairnessEvaluator()
        fairness_df = evaluator.evaluate_all(y_true, y_pred, sensitive)
        self.fairness_results = fairness_df
        logger.info(f"  Fairness metrics computed: {len(fairness_df)} rows")

    def robustness_analysis(self):
        logger.info("=== Robustness Analysis ===")
        if "telco" not in self.models:
            return
        X = self.features["telco"].values
        y = self.targets["telco"]
        model = self.models["telco"]["Random Forest"]["model"].model
        injector = NoiseInjector(model)
        gauss_df = injector.gaussian_noise(X, y)
        label_df = injector.label_noise(X, y)
        missing_df = injector.missing_data(X, y)
        self.robustness_results = {
            "gaussian": gauss_df, "label": label_df, "missing": missing_df
        }
        logger.info(f"  Robustness: Gaussian delta={gauss_df['delta'].iloc[-1]:.4f}")

    def run_publication_experiments(self):
        logger.info("=== Publication-Ready Experiments ===")
        self.experiments_runner = ESACRIFExperiments()
        self.experiments_runner.features = self.features
        self.experiments_runner.targets = self.targets
        self.experiments_runner.datasets_raw = {
            k: self.processors[k].load_data(str(getattr(config, f"{k.upper()}_FILE")))
            for k in self.processors if getattr(config, f"{k.upper()}_FILE", None)
        }
        self.experiments_runner.processors = self.processors
        self.experiments_runner.is_synthetic = config.SYNTHETIC_DATASETS

        self.experiments_runner.run_experiment_1_cross_dataset()
        self.experiments_runner.run_experiment_2_explanation_stability()
        self.experiments_runner.run_experiment_3_ablation()
        self.experiments_runner.run_experiment_4_models()
        self.experiments_runner.run_experiment_5_survival()
        self.experiments_runner.run_experiment_6_business()
        self.experiments_runner.run_experiment_7_fairness()
        self.experiments_runner.run_experiment_8_statistics()
        self.experiments_runner.run_experiment_9_cell2cell_validation()
        self.experiments_runner.generate_publication_outputs()

        self.results["publication"] = self.experiments_runner.results_all
        logger.info("  Publication experiments complete")

    def run_all(self):
        logger.info("=" * 60)
        logger.info("ESACRIF - Full Pipeline Execution")
        logger.info("=" * 60)
        self.load_all_datasets()
        self.engineer_features_telco()
        self.train_models("telco")
        self.survival_analysis()
        self.cross_dataset_experiments()
        self.fairness_analysis()
        self.robustness_analysis()
        self.run_publication_experiments()
        logger.info("=" * 60)
        logger.info("ESACRIF Pipeline Complete")
        logger.info("=" * 60)
        return self


if __name__ == "__main__":
    pipeline = ESACRIFPipeline()
    pipeline.run_all()
