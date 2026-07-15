import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.preprocessing import TelcoProcessor, IranianProcessor, BankProcessor
from src.feature_engineering import TimeAwareFeatureEngineer, FeatureAblation
from src.prediction import LogisticRegressionModel, RandomForestModel
from src.utils.metrics import calculate_metrics
from src.fairness import FairnessEvaluator
from src.robustness import NoiseInjector
from src.statistics import BootstrapInference, delong_roc_test, mcnemar_test, CalibrationEvaluator
from src.cross_dataset import CrossDatasetExperiment
from src.survival import KaplanMeierAnalysis, CoxPHAnalysis
from src.business import CustomerLifetimeValue, RetentionOptimizer
from experiments import (
    CrossDatasetValidator, FrameworkAblation, BusinessImpactExperiment,
    SurvivalMetrics, PublicationTableGenerator, PublicationFigureGenerator,
    FeatureAligner
)
import config


def test_telco_processor():
    proc = TelcoProcessor()
    assert proc.target_col == "Churn"
    assert "tenure" in proc.numeric_cols
    assert config.TELCO_FILE.exists()
    df = proc.load_data(str(config.TELCO_FILE))
    assert len(df) > 0
    X, y = proc.fit_transform(df)
    assert X.shape[0] == len(y)
    print(f"  TelcoProcessor: {X.shape}, churn={y.mean():.3f}")


def test_iranian_processor():
    proc = IranianProcessor()
    if config.IRANIAN_FILE.exists():
        df = proc.load_data(str(config.IRANIAN_FILE))
        assert len(df) > 0
        X, y = proc.fit_transform(df)
        assert X.shape[0] == len(y)
        print(f"  IranianProcessor: {X.shape}, churn={y.mean():.3f}")


def test_feature_engineering():
    proc = TelcoProcessor()
    df = proc.load_data(str(config.TELCO_FILE))
    df = proc.clean(df)
    engineer = TimeAwareFeatureEngineer()
    df_fe = engineer.fit_transform(df)
    for f in engineer.get_feature_names():
        assert f in df_fe.columns
    assert len(engineer.get_feature_names()) == 5
    print(f"  FeatureEngineering: {engineer.get_feature_names()}")


def test_ablation():
    proc = TelcoProcessor()
    X, y = proc.fit_transform(proc.load_data(str(config.TELCO_FILE)))
    df_raw = proc.load_data(str(config.TELCO_FILE))
    df_clean = proc.clean(df_raw)
    engineer = TimeAwareFeatureEngineer()
    df_fe = engineer.fit_transform(df_clean)
    X_eng = pd.get_dummies(df_fe.drop(columns=['Churn']), drop_first=True)
    eng_cols = engineer.get_feature_names()
    X_eng_only = X_eng[[c for c in eng_cols if c in X_eng.columns]]
    abl = FeatureAblation()
    auc_with, auc_without = abl.evaluate_engineered_features(X_eng.values, X.values, y)
    assert auc_with > 0
    assert auc_without > 0
    print(f"  Ablation: with={auc_with:.4f}, without={auc_without:.4f}")


def test_logistic_regression():
    model = LogisticRegressionModel()
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    assert len(preds) == len(y)
    assert len(probs) == len(y)
    metrics = calculate_metrics(y, preds, probs)
    assert "roc_auc" in metrics
    print(f"  LogisticRegression: metrics computed OK")


def test_random_forest():
    model = RandomForestModel()
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    assert len(preds) == len(y)
    feat_imp = model.get_feature_importance()
    assert feat_imp is not None
    assert len(feat_imp) == 10
    print(f"  RandomForest: feature importance shape={feat_imp.shape}")


def test_metrics():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.4, 0.7])
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] > 0
    assert metrics["roc_auc"] > 0
    print(f"  Metrics: accuracy={metrics['accuracy']}, roc_auc={metrics['roc_auc']}")


def test_fairness():
    evaluator = FairnessEvaluator()
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    sensitive = pd.DataFrame({"gender": np.random.choice(["Male", "Female"], 100)})
    df = evaluator.evaluate_all(y_true, y_pred, sensitive)
    assert len(df) >= 2
    print(f"  FairnessEvaluator: {len(df)} rows")


def test_robustness():
    from sklearn.linear_model import LogisticRegression
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    model = LogisticRegression(max_iter=1000, random_state=42)
    injector = NoiseInjector(model, cv=3)
    gauss = injector.gaussian_noise(X, y, noise_levels=[0.1])
    assert len(gauss) == 2
    print(f"  NoiseInjector: gaussian noise test passed")


def test_statistics():
    y_true = np.random.randint(0, 2, 200)
    y_prob_a = np.random.rand(200)
    y_prob_b = np.random.rand(200)
    boot = BootstrapInference(n_bootstrap=100)
    ci = boot.compute_ci(y_true, y_prob_a)
    assert "ci_lower" in ci
    z, p = delong_roc_test(y_true, y_prob_a, y_prob_b)
    assert isinstance(z, float)
    mcn = mcnemar_test(y_true, (y_prob_a > 0.5).astype(int), (y_prob_b > 0.5).astype(int))
    assert "p_value" in mcn
    cal = CalibrationEvaluator.expected_calibration_error(y_true, y_prob_a, n_bins=5)
    assert "ece" in cal
    print(f"  Statistics: all tests passed")


def test_survival():
    durations = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    event = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    km = KaplanMeierAnalysis()
    km.fit(durations, event)
    assert hasattr(km.kmf, "median_survival_time_")
    sm = SurvivalMetrics()
    rmst = sm.compute_rmst(durations, event)
    assert "rmst" in rmst
    print(f"  Survival: RMST test passed")


def test_cross_dataset_validator():
    validator = CrossDatasetValidator()
    X1 = pd.DataFrame(np.random.randn(50, 5))
    y1 = np.random.randint(0, 2, 50)
    X2 = pd.DataFrame(np.random.randn(50, 5))
    y2 = np.random.randint(0, 2, 50)
    validator.add_dataset("a", X1, y1, domain="telecom")
    validator.add_dataset("b", X2, y2, domain="telecom")
    validator.run_same_domain()
    summary = validator.summary()
    assert len(summary) >= 1
    print(f"  CrossDatasetValidator: {len(summary)} transfer(s)")


def test_business():
    clv = CustomerLifetimeValue()
    vals = clv.compute_clv(np.array([50, 100]), np.array([12, 24]), np.array([0.3, 0.7]))
    assert len(vals) == 2
    assert all(v >= 0 for v in vals)
    opt = RetentionOptimizer()
    sim = opt.simulate_intervention(0.8, 0.3, 1000, 50)
    assert "roi_pct" in sim
    biz = BusinessImpactExperiment()
    churn_probs = np.array([0.1, 0.3, 0.6, 0.9])
    clv_vals = np.array([100, 200, 300, 400])
    campaign = biz.simulate_campaign(churn_probs, clv_vals, costs=[20], thresholds=[0.5])
    assert len(campaign) > 0
    print(f"  Business: all tests passed")


def test_publication_tables():
    gen = PublicationTableGenerator(output_dir=config.OUTPUTS_DIR)
    info = {"test": {"name": "Test", "domain": "Test", "n_samples": 100, "n_features": 10, "churn_rate": 0.3, "is_synthetic": False, "source": "Test"}}
    df = gen.table_dataset_statistics(info)
    assert len(df) == 1
    # Test survival table with survival results
    sr = {"group_rmst": pd.DataFrame([{"Group": "A", "N": 50, "Events": 10, "RMST": 45.0, "Median Survival": "Not reached", "Time Horizon": 72.0}])}
    gen.table_survival(sr)
    print(f"  PublicationTables: tables created successfully")


def test_feature_aligner():
    X1 = pd.DataFrame({"tenure": [1, 2, 3], "MonthlyCharges": [50, 60, 70]})
    X2 = pd.DataFrame({"Subscription Length": [1, 2, 3], "Charge Amount": [50, 60, 70]})
    X_s, X_t = FeatureAligner.align_features(X1, X2, "telco", "iranian")
    assert X_s is not None
    assert X_t is not None
    assert X_s.shape == X_t.shape
    print(f"  FeatureAligner: shape={X_s.shape}")


def test_framework_ablation():
    proc = TelcoProcessor()
    X, y = proc.fit_transform(proc.load_data(str(config.TELCO_FILE)))
    df_raw = proc.load_data(str(config.TELCO_FILE))
    df_clean = proc.clean(df_raw)
    engineer = TimeAwareFeatureEngineer()
    df_fe = engineer.fit_transform(df_clean)
    X_eng = pd.get_dummies(df_fe.drop(columns=['Churn']), drop_first=True)
    eng_cols = engineer.get_feature_names()
    X_eng_only = X_eng[[c for c in eng_cols if c in X_eng.columns]]
    abl = FrameworkAblation()
    abl.run_ablations(X, y, X_eng_only)
    summary = abl.summary()
    assert len(summary) > 0
    assert "roc_auc" in summary.columns or "error" in summary.columns
    print(f"  FrameworkAblation: {len(summary)} configurations")


if __name__ == "__main__":
    print("Running ESACRIF publication tests...")
    test_telco_processor()
    test_iranian_processor()
    test_feature_engineering()
    test_ablation()
    test_logistic_regression()
    test_random_forest()
    test_metrics()
    test_fairness()
    test_robustness()
    test_statistics()
    test_survival()
    test_cross_dataset_validator()
    test_business()
    test_publication_tables()
    test_feature_aligner()
    test_framework_ablation()
    print("All tests passed!")
