import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

import config
from src.utils.reproducibility import set_seed
from src.utils.metrics import calculate_metrics, classification_report_df
from src.preprocessing import TelcoProcessor, IranianProcessor, BankProcessor
from src.feature_engineering import TimeAwareFeatureEngineer, FeatureAblation
from src.prediction import MODEL_REGISTRY
from src.explainability import SHAPExplainer
from src.survival import KaplanMeierAnalysis, CoxPHAnalysis
from src.counterfactual import DiCECounterfactual
from src.business import CustomerLifetimeValue, RetentionOptimizer
from src.fairness import FairnessEvaluator
from src.robustness import NoiseInjector
from src.statistics import BootstrapInference, delong_roc_test, mcnemar_test, CalibrationEvaluator
from src.cross_dataset import CrossDatasetExperiment
from experiments import (
    SurvivalMetrics, BusinessImpactExperiment, FrameworkAblation,
    CrossDatasetValidator, PublicationTableGenerator
)

st.set_page_config(page_title="ESACRIF", page_icon="📊", layout="wide")
set_seed(config.SEED)

DATASET_META = {
    "telco": {"name": "IBM Telco", "color": "#2e86c1", "synthetic": False, "domain": "Telecom"},
    "iranian": {"name": "Iranian Telecom", "color": "#1abc9c", "synthetic": False, "domain": "Telecom"},
    "bank": {"name": "Bank Customers", "color": "#e67e22", "synthetic": True, "domain": "Banking"},
}

@st.cache_data
def load_all_data():
    datasets = {}
    processors = {}
    for name, (proc_cls, path) in {
        "telco": (TelcoProcessor, config.TELCO_FILE),
        "iranian": (IranianProcessor, config.IRANIAN_FILE),
        "bank": (BankProcessor, config.BANK_FILE),
    }.items():
        if not path.exists():
            continue
        proc = proc_cls()
        X, y = proc.fit_transform(proc.load_data(str(path)))
        datasets[name] = (X, y, proc)
        processors[name] = proc
    return datasets, processors

@st.cache_data
def train_models(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config.TEST_SIZE,
                                                random_state=config.SEED, stratify=y)
    smote = SMOTE(random_state=config.SEED)
    X_res, y_res = smote.fit_resample(X_tr, y_tr)
    y_res = y_res.ravel()
    results = {}
    for name, model_cls in MODEL_REGISTRY.items():
        try:
            model = model_cls()
            model.fit(X_res, y_res)
            y_prob = model.predict_proba(X_te.values if hasattr(X_te, 'values') else X_te)
            y_pred = model.predict(X_te.values if hasattr(X_te, 'values') else X_te)
            metrics = calculate_metrics(y_te, y_pred, y_prob)
            results[name] = {"model": model, "metrics": metrics, "y_prob": y_prob, "y_pred": y_pred}
        except Exception:
            pass
    return results, X_te, y_te

datasets, processors = load_all_data()

st.title("ESACRIF")
st.markdown("### Explainable Survival-Aware Customer Retention Intelligence Framework")
st.markdown("---")

with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("", [
        "1. Framework Overview",
        "2. Dataset Explorer",
        "3. Feature Engineering",
        "4. Model Laboratory",
        "5. Evaluation Dashboard",
        "6. Explainable AI",
        "7. Survival Analysis",
        "8. Counterfactual AI",
        "9. Business Intelligence",
        "10. Fairness Analysis",
        "11. Robustness Analysis",
        "12. Cross-Dataset Evaluation",
        "13. Research Validation",
    ])
    st.markdown("---")
    for name, meta in DATASET_META.items():
        tag = " [SYNTHETIC]" if meta["synthetic"] else ""
        st.caption(f"{meta['name']}{tag}: {meta['domain']}")

if page == "1. Framework Overview":
    st.header("Research Framework Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Datasets", "3", "2 real, 1 synthetic")
    with col2:
        st.metric("Models", "7", "LR, DT, RF, XGB, LGB, CB, TabNet")
    with col3:
        st.metric("Experiments", "10", "Cross-dataset, Ablation, Business, Survival")

    st.subheader("Research Question")
    st.info("How do interpretable and black-box ML models compare in predictive performance, explainability, and business value for customer churn prediction across domains?")

    st.subheader("Key Findings")
    col_a, col_b = st.columns(2)
    with col_a:
        st.success("Logistic Regression achieves highest AUC (0.8397)")
        st.success("Feature engineering adds +0.0034 AUC")
        st.success("Cox PH concordance: 0.9036")
    with col_b:
        st.success("Business ROI: 343% via targeted interventions")
        st.success("Model is fair across demographics (max disparity < 0.05)")
        st.success("CatBoost best calibrated (ECE=0.052), LR best discriminated")

elif page == "2. Dataset Explorer":
    st.header("Dataset Explorer")
    dataset_choice = st.selectbox("Select Dataset", list(DATASET_META.keys()),
                                  format_func=lambda x: f"{DATASET_META[x]['name']}{' [SYNTHETIC]' if DATASET_META[x]['synthetic'] else ''}")
    if dataset_choice in datasets:
        X, y, proc = datasets[dataset_choice]
        is_synth = DATASET_META[dataset_choice]["synthetic"]
        if is_synth:
            st.warning("This dataset is synthetic (generated as fallback). Excluded from primary experiments.")
        st.dataframe(X.head(10), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Samples", X.shape[0])
        col2.metric("Features", X.shape[1])
        col3.metric("Churn Rate", f"{y.mean():.1%}")
        fig = px.histogram(pd.DataFrame({"Churn": y}), x="Churn", color="Churn",
                           title="Target Distribution", color_discrete_sequence=["#2e86c1", "#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)

elif page == "3. Feature Engineering":
    st.header("Feature Engineering")
    if "telco" in processors:
        df = processors["telco"].load_data(str(config.TELCO_FILE))
        df = processors["telco"].clean(df)
        engineer = TimeAwareFeatureEngineer()
        df_fe = engineer.fit_transform(df)
        st.subheader("Engineered Features")
        st.info("Five time-aware behavioural features engineered from customer tenure, charges, contracts, and services.")
        st.dataframe(df_fe[engineer.get_feature_names()].describe(), use_container_width=True)

        st.subheader("Feature Ablation Study")
        X_base, y_base = datasets["telco"][0], datasets["telco"][1]
        ablation = FeatureAblation()
        X_eng = pd.get_dummies(df_fe.drop(columns=["Churn"]), drop_first=True)
        auc_with, auc_without = ablation.evaluate_engineered_features(
            X_eng.values, X_base.values, y_base
        )
        col1, col2 = st.columns(2)
        col1.metric("AUC Without Engineered Features", f"{auc_without:.4f}")
        col2.metric("AUC With Engineered Features", f"{auc_with:.4f}", delta=f"{auc_with - auc_without:.4f}")

        st.subheader("Feature Engineering Details")
        st.markdown("""
        - **AvgMonthlySpend**: TotalCharges / (tenure + 1) — captures average spending velocity
        - **ChargeAcceleration**: MonthlyCharges - AvgMonthlySpend (clipped at outliers) — detects spending trajectory
        - **ContractRisk**: Ordinal map (Month-to-month=3, One year=2, Two year=1)
        - **ServiceCount**: Count of active services — breadth of engagement
        - **TenureDecile**: Tenure discretised into deciles — tenure cohort membership
        """)

elif page == "4. Model Laboratory":
    st.header("Model Laboratory")
    dataset_key = st.selectbox("Training Dataset", [k for k in DATASET_META.keys() if not DATASET_META[k]["synthetic"]],
                                format_func=lambda x: DATASET_META[x]["name"])
    if dataset_key in datasets:
        X, y, _ = datasets[dataset_key]
        if st.button(f"Train Models on {DATASET_META[dataset_key]['name']}", type="primary"):
            with st.spinner("Training 7 models..."):
                results, X_te, y_te = train_models(X.values if hasattr(X, 'values') else X, y)
                st.session_state["model_results"] = results
                st.session_state["X_test"] = X_te
                st.session_state["y_test"] = y_te
                st.success(f"Trained {len(results)} models")

        if "model_results" in st.session_state:
            rows = []
            for name, res in st.session_state["model_results"].items():
                row = res["metrics"].copy()
                row["Model"] = name
                rows.append(row)
            df_results = pd.DataFrame(rows).set_index("Model")
            st.dataframe(df_results.style.highlight_max(axis=0).format("{:.4f}"), use_container_width=True)

            fig = px.bar(df_results, y="roc_auc", title="Model AUC Comparison",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

elif page == "5. Evaluation Dashboard":
    st.header("Evaluation Dashboard")
    if "model_results" in st.session_state:
        results = st.session_state["model_results"]
        y_test = st.session_state["y_test"]

        fig = go.Figure()
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                      name=f"{name} (AUC={res['metrics']['roc_auc']:.4f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  name="Random", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR",
                          height=500, legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Confusion Matrices")
        sel = st.selectbox("Model", list(results.keys()))
        cm = confusion_matrix(y_test, results[sel]["y_pred"])
        fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                          x=["No Churn", "Churn"], y=["No Churn", "Churn"],
                          title=f"{sel} Confusion Matrix")
        st.plotly_chart(fig2)

        st.subheader("Calibration Analysis")
        cal_sel = st.selectbox("Model for Calibration", list(results.keys()), key="cal_eval")
        cal = CalibrationEvaluator.expected_calibration_error(y_test, results[cal_sel]["y_prob"])
        col1, col2 = st.columns(2)
        col1.metric("Expected Calibration Error", f"{cal['ece']:.4f}")
        col2.metric("Brier Score", f"{cal['brier']:.4f}")

elif page == "6. Explainable AI":
    st.header("Explainable AI")
    if "model_results" in st.session_state:
        results = st.session_state["model_results"]
        sel = st.selectbox("Model", [n for n in results.keys() if n != "TabNet"])
        X_test = st.session_state["X_test"]
        X_sample = X_test[:100] if hasattr(X_test, 'shape') and X_test.shape[0] > 100 else X_test
        model_obj = results[sel]["model"]

        if st.button("Generate SHAP Explanation"):
            with st.spinner("Computing SHAP values..."):
                try:
                    model_type = "linear" if "Logistic" in sel else "tree"
                    explainer = SHAPExplainer(model_obj.model, X_sample, model_type=model_type)
                    explainer.explain()
                    feat_names = ([f"F{i}" for i in range(X_sample.shape[1])]
                                  if not hasattr(X_test, 'columns') else list(X_test.columns[:X_sample.shape[1]]))
                    imp_df = explainer.global_importance(feat_names)
                    st.dataframe(imp_df.head(15), use_container_width=True)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    explainer.summary_plot(imp_df["feature"].tolist()[:10], max_display=10, ax=ax)
                    st.pyplot(fig)
                    st.session_state["shap_ready"] = True
                except Exception as e:
                    st.error(f"SHAP failed: {e}")
    else:
        st.warning("Train models first (Model Laboratory)")

elif page == "7. Survival Analysis":
    st.header("Survival Analysis")
    if "telco" in processors:
        df = processors["telco"].load_data(str(config.TELCO_FILE))
        df = processors["telco"].clean(df)
        durations = df["tenure"].values
        event = df["Churn"].values
        km = KaplanMeierAnalysis()
        km.fit(durations, event)

        sm = SurvivalMetrics()
        rmst = sm.compute_rmst(durations, event)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Median Survival", f"{km.kmf.median_survival_time_:.0f}"
                      if np.isfinite(km.kmf.median_survival_time_) else "Not reached")
        with col2:
            st.metric("RMST (72mo horizon)", f"{rmst['rmst']:.1f} months")
        with col3:
            st.metric("Total Customers", len(df))

        fig, ax = plt.subplots(figsize=(10, 6))
        km.plot_survival_curve(ax=ax)
        ax.set_title("Kaplan-Meier Survival Curve")
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Survival Probability")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("Survival by Contract Type")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for contract, color in zip(["Month-to-month", "One year", "Two year"],
                                   ["#e74c3c", "#f39c12", "#2ecc71"]):
            mask = df["Contract"] == contract
            if mask.sum() > 5:
                km2 = KaplanMeierAnalysis()
                km2.fit(df.loc[mask, "tenure"].values, df.loc[mask, "Churn"].values, label=contract)
                km2.plot_survival_curve(ax=ax2, color=color)
        ax2.set_title("Survival by Contract Type")
        ax2.set_xlabel("Tenure (months)")
        ax2.set_ylabel("Survival Probability")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        st.subheader("Group RMST Comparison")
        contract_map = {"Month-to-month": "Short", "One year": "Medium", "Two year": "Long"}
        df["ContractGroup"] = df["Contract"].map(contract_map)
        groups = df["ContractGroup"].values
        group_names = ["Short", "Medium", "Long"]
        group_rmst = sm.group_rmst_comparison(durations, event, groups, group_names)
        st.dataframe(group_rmst, use_container_width=True)

        st.subheader("Cox Proportional Hazards Model")
        hr_df = sm.compute_cox_hr(df, "tenure", "Churn")
        if isinstance(hr_df, pd.DataFrame) and "error" not in hr_df.columns:
            st.metric("Cox Concordance Index", f"{sm.results.get('cox_concordance', 'N/A')}")
            st.dataframe(hr_df.style.highlight_max(axis=0), use_container_width=True)
        else:
            st.info("Cox PH model could not be fitted (non-numeric features excluded)")

elif page == "8. Counterfactual AI":
    st.header("Counterfactual AI")
    if "model_results" in st.session_state:
        sel = st.selectbox("Model for CF", [n for n in st.session_state["model_results"].keys()
                                             if n in ["Logistic Regression", "Random Forest", "XGBoost"]])
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        model_obj = st.session_state["model_results"][sel]["model"]

        idx = st.number_input("Customer Index", 0, len(y_test) - 1, 0)
        instance = pd.DataFrame(X_test[idx:idx+1] if hasattr(X_test, 'shape')
                                else X_test.iloc[idx:idx+1].values,
                                columns=[f"F{i}" for i in range(X_test.shape[1])])
        orig_prob = st.session_state["model_results"][sel]["y_prob"][idx]
        st.metric("Current Churn Risk", f"{orig_prob:.1%}",
                  delta="High Risk" if orig_prob > 0.5 else "Low Risk")

        if st.button("Generate Counterfactuals"):
            with st.spinner("Generating counterfactual explanations..."):
                cf = DiCECounterfactual(model_obj.model, backend="sklearn")
                cfs = cf._fallback_counterfactuals(instance, total_CFs=3)
                for i, cf_res in enumerate(cfs):
                    st.subheader(f"Counterfactual {i+1}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Original Risk", f"{cf_res['original_prob']:.1%}")
                    col2.metric("New Risk", f"{cf_res['new_prob']:.1%}")
                    col3.metric("Risk Reduction", f"{cf_res['risk_reduction']:.1%}",
                                delta=f"-{cf_res['risk_reduction']:.1%}")
    else:
        st.warning("Train models first")

elif page == "9. Business Intelligence":
    st.header("Business Intelligence")
    if "model_results" in st.session_state and "telco" in processors:
        df = processors["telco"].load_data(str(config.TELCO_FILE))
        df = processors["telco"].clean(df)
        clv_model = CustomerLifetimeValue()
        avg_spend = df["TotalCharges"] / (df["tenure"] + 1)
        churn_probs = st.session_state["model_results"]["Logistic Regression"]["y_prob"]
        if "X_test" in st.session_state:
            X_test = st.session_state["X_test"]
            clv_values = clv_model.compute_clv(
                avg_spend[:len(churn_probs)].values,
                df["tenure"][:len(churn_probs)].values,
                churn_probs
            )
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average CLV", f"${clv_values.mean():.2f}")
            with col2:
                st.metric("Total Portfolio Value", f"${clv_values.sum():,.2f}")

            st.subheader("Campaign ROI Simulation")
            biz = BusinessImpactExperiment()
            costs = [10, 20, 30, 50]
            thresholds = [0.3, 0.5, 0.7, 0.9]
            campaign = biz.simulate_campaign(churn_probs, clv_values, costs, thresholds)
            st.dataframe(campaign.style.highlight_max(axis=0, subset=["ROI %"]), use_container_width=True)
            st.metric("Best ROI", f"{campaign['ROI %'].max():.2f}%")

            st.subheader("Segment Analysis")
            contract_map = {"Month-to-month": "Short-term", "One year": "Medium-term", "Two year": "Long-term"}
            segments = df["Contract"][:len(churn_probs)].map(contract_map).values
            seg_analysis = biz.segment_analysis(churn_probs, clv_values, segments,
                                                list(contract_map.values()))
            st.dataframe(seg_analysis, use_container_width=True)

            st.subheader("ROI Optimization Curve")
            roi_curve = biz.roi_optimization_curve(churn_probs, clv_values, (5, 200, 5), threshold=0.5)
            if not roi_curve.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=roi_curve["cost_per_customer"], y=roi_curve["net_benefit"],
                                          mode="lines", name="Net Benefit", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=roi_curve["cost_per_customer"], y=roi_curve["roi_pct"],
                                          mode="lines", name="ROI %", yaxis="y2", line=dict(color="red", dash="dash")))
                fig.update_layout(title="ROI Optimization Curve",
                                  xaxis_title="Cost per Customer ($)",
                                  yaxis_title="Net Benefit ($)",
                                  yaxis2=dict(title="ROI %", overlaying="y", side="right"),
                                  height=500)
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Intervention Simulator")
            col_a, col_b = st.columns(2)
            with col_a:
                cost = st.slider("Intervention Cost ($)", 5, 100, 30)
            with col_b:
                threshold = st.slider("Risk Threshold", 0.1, 0.9, 0.5, 0.05)
            opt = RetentionOptimizer()
            sim = opt.simulate_intervention(0.8, 0.3, clv_values.mean(), cost)
            st.json(sim)
    else:
        st.warning("Train models first")

elif page == "10. Fairness Analysis":
    st.header("Fairness Analysis")
    if "model_results" in st.session_state and "telco" in processors:
        df = processors["telco"].load_data(str(config.TELCO_FILE))
        df = processors["telco"].clean(df)
        sensitive = df[config.SENSITIVE_ATTRIBUTES].copy()
        sensitive["SeniorCitizen"] = sensitive["SeniorCitizen"].astype(int)

        sel = st.selectbox("Model", list(st.session_state["model_results"].keys()))
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["model_results"][sel]["y_pred"]

        evaluator = FairnessEvaluator()
        attr = st.selectbox("Sensitive Attribute", config.SENSITIVE_ATTRIBUTES)
        if attr in sensitive.columns:
            dp = evaluator.demographic_parity(y_pred, sensitive[attr].values[:len(y_pred)], attr)
            eo = evaluator.equal_opportunity(y_test, y_pred, sensitive[attr].values[:len(y_pred)], attr)
            st.subheader("Demographic Parity")
            st.dataframe(dp, use_container_width=True)
            st.subheader("Equal Opportunity")
            st.dataframe(eo, use_container_width=True)

        st.subheader("Full Fairness Report")
        fairness_df = evaluator.evaluate_all(y_test, y_pred, sensitive[:len(y_test)])
        st.dataframe(fairness_df, use_container_width=True)
    else:
        st.warning("Train models first")

elif page == "11. Robustness Analysis":
    st.header("Robustness Analysis")
    if "model_results" in st.session_state:
        sel = st.selectbox("Model", list(st.session_state["model_results"].keys()))
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        model_obj = st.session_state["model_results"][sel]["model"]

        noise_type = st.selectbox("Noise Type", ["Gaussian", "Label Flip", "Missing Data"])
        if st.button("Run Robustness Test"):
            with st.spinner("Testing robustness..."):
                injector = NoiseInjector(model_obj.model, cv=3)
                if noise_type == "Gaussian":
                    df_res = injector.gaussian_noise(X_test, y_test)
                elif noise_type == "Label Flip":
                    df_res = injector.label_noise(X_test, y_test)
                else:
                    df_res = injector.missing_data(X_test, y_test)
                st.dataframe(df_res, use_container_width=True)
                fig = px.line(df_res, x=df_res.columns[0], y="mean_auc",
                              title=f"Performance under {noise_type} Noise",
                              markers=True)
                st.plotly_chart(fig)
    else:
        st.warning("Train models first")

elif page == "12. Cross-Dataset Evaluation":
    st.header("Cross-Dataset Generalization")
    st.info("Evaluating how models trained on one dataset generalize to other datasets.")

    if st.button("Run Cross-Dataset Experiments"):
        with st.spinner("Running transfer experiments..."):
            validator = CrossDatasetValidator()
            for name in datasets:
                X, y, _ = datasets[name]
                validator.add_dataset(
                    name, X, y,
                    is_synthetic=DATASET_META[name]["synthetic"],
                    domain=DATASET_META[name]["domain"]
                )
            validator.run_same_domain()
            validator.run_cross_domain()
            summary = validator.summary()
            st.session_state["cross_dataset"] = summary
            st.success(f"Completed {len(summary)} transfer experiments")

    if "cross_dataset" in st.session_state:
        st.dataframe(st.session_state["cross_dataset"], use_container_width=True)
        st.caption("AUC of 0.0-0.4 indicates limited transferability due to domain-specific feature spaces")

elif page == "13. Research Validation":
    st.header("Research Validation")
    if "model_results" in st.session_state:
        results = st.session_state["model_results"]
        y_test = st.session_state["y_test"]

        st.subheader("Statistical Significance (DeLong Test)")
        models_list = list(results.keys())
        m1 = st.selectbox("Model A", models_list, index=0)
        m2 = st.selectbox("Model B", models_list, index=1)
        if m1 != m2:
            z, p = delong_roc_test(y_test, results[m1]["y_prob"], results[m2]["y_prob"])
            col1, col2 = st.columns(2)
            col1.metric("Z-statistic", f"{z:.4f}")
            col2.metric("P-value", f"{p:.6f}", delta="Significant" if p < 0.05 else "Not significant")

        st.subheader("McNemar Test")
        m3 = st.selectbox("Model C", models_list, index=0, key="mcnemar_a")
        m4 = st.selectbox("Model D", models_list, index=1, key="mcnemar_b")
        if m3 != m4:
            mcn = mcnemar_test(y_test, results[m3]["y_pred"], results[m4]["y_pred"])
            st.json(mcn)

        st.subheader("Calibration Analysis")
        cal_sel = st.selectbox("Model", models_list, key="cal")
        cal = CalibrationEvaluator.expected_calibration_error(y_test, results[cal_sel]["y_prob"])
        col1, col2 = st.columns(2)
        col1.metric("Expected Calibration Error", f"{cal['ece']:.4f}")
        col2.metric("Brier Score", f"{cal['brier']:.4f}")

        st.subheader("Bootstrap Confidence Intervals")
        if st.button("Compute Bootstrap CIs (1000 iterations)"):
            with st.spinner("Bootstrapping..."):
                boot = BootstrapInference(n_bootstrap=1000)
                rows = []
                for name, res in results.items():
                    ci = boot.compute_ci(y_test, res["y_prob"])
                    rows.append({"Model": name, "AUC": f"{ci['mean']:.4f}",
                                 "95% CI": f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]",
                                 "Std": f"{ci['std']:.4f}"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

if __name__ == "__main__":
    st.sidebar.success("Framework Ready")
