from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
TEST_SIZE = 0.2
DROP_COLUMNS = [
    "customer_id",
    "as_of_date",
    "churn_type",
    "churn_probability_true",
]


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_data_path() -> Path:
    return get_project_root() / "data" / "insurance_policyholder_churn_synthetic.csv"


def get_model_path() -> Path:
    return get_project_root() / "model" / "churn_model_new.pkl"


def get_threshold_path() -> Path:
    return get_project_root() / "model" / "threshold_new.pkl"


def get_threshold_plot_path() -> Path:
    return get_project_root() / "model" / "threshold_analysis_new.png"


def get_corr_plot_path() -> Path:
    return get_project_root() / "model" / "target_corr_scatter.png"


def get_scored_output_dir() -> Path:
    return get_project_root() / "data" / "scored"


def get_scored_output_path() -> Path:
    return get_scored_output_dir() / "scored_customers_new_model.parquet"


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(get_data_path())


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns], errors="ignore")
    if "churn_flag" in feature_df.columns:
        feature_df = feature_df.drop(columns=["churn_flag"])
    return add_engineered_features(feature_df)


def add_engineered_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()
    df["premium_increase_shock"] = df["premium_change_pct"].clip(lower=0) * (
        1 + df["num_price_increases_last_3y"]
    )
    df["premium_jump_flag"] = (df["premium_change_pct"] >= 0.12).astype(int)
    df["payment_risk_score"] = df["late_payment_count_12m"] + df["missed_payment_flag"] * 2
    df["service_risk_score"] = (
        df["complaint_flag"] + df["quote_requested_flag"] + df["coverage_downgrade_flag"]
    )
    df["engagement_risk_score"] = df["payment_risk_score"] + df["service_risk_score"]
    df["tenure_inverse"] = 1 / (df["customer_tenure_months"] + 1)
    df["single_policy_short_tenure"] = (
        (df["multi_policy_flag"] == 0) & (df["customer_tenure_months"] <= 24)
    ).astype(int)
    df["premium_x_complaint"] = df["premium_jump_flag"] * df["complaint_flag"]
    df["premium_x_quote"] = df["premium_jump_flag"] * df["quote_requested_flag"]
    df["late_x_quote"] = (
        (df["late_payment_count_12m"] >= 2).astype(int) * df["quote_requested_flag"]
    )
    df["downgrade_x_quote"] = df["coverage_downgrade_flag"] * df["quote_requested_flag"]
    df["monthly_payment_flag"] = (df["payment_frequency"] == "Monthly").astype(int)
    df["monthly_x_premium_jump"] = df["monthly_payment_flag"] * df["premium_jump_flag"]
    df["auto_or_health"] = df["policy_type"].isin(["Auto", "Health"]).astype(int)
    return df


@st.cache_resource
def load_model_bundle():
    model = joblib.load(get_model_path())
    patch_loaded_pipeline(model)
    threshold = float(joblib.load(get_threshold_path()))
    return model, threshold


def patch_loaded_pipeline(model) -> None:
    """Patch sklearn 1.7.x persisted imputers for 1.8.x runtime compatibility."""
    preprocess = getattr(model, "named_steps", {}).get("preprocess")
    if preprocess is None or not hasattr(preprocess, "named_transformers_"):
        return

    for transformer in preprocess.named_transformers_.values():
        if isinstance(transformer, Pipeline):
            imputer = transformer.named_steps.get("imputer")
            if imputer is not None and not hasattr(imputer, "_fill_dtype"):
                imputer._fill_dtype = imputer.statistics_.dtype
        elif isinstance(transformer, ColumnTransformer):
            for nested in transformer.named_transformers_.values():
                if isinstance(nested, Pipeline):
                    imputer = nested.named_steps.get("imputer")
                    if imputer is not None and not hasattr(imputer, "_fill_dtype"):
                        imputer._fill_dtype = imputer.statistics_.dtype


def get_train_test_split():
    raw_df = load_raw_data()
    X = build_feature_frame(raw_df)
    y = raw_df["churn_flag"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return raw_df, X_train, X_test, y_train, y_test


@st.cache_data
def evaluate_saved_model() -> dict:
    _, _, X_test, _, y_test = get_train_test_split()
    model, threshold = load_model_bundle()

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": cm.tolist(),
        "test_size": int(len(y_test)),
        "predicted_positive": int(y_pred.sum()),
        "actual_positive": int(y_test.sum()),
    }


def add_risk_tier(scored_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = scored_df.copy()
    df["risk_tier"] = "stable"
    df.loc[df["churn_probability"] >= threshold, "risk_tier"] = "watch"
    df.loc[df["churn_probability"] >= max(0.30, threshold * 1.5), "risk_tier"] = "high"
    df.loc[df["churn_probability"] >= max(0.50, threshold * 2.5), "risk_tier"] = "critical"
    return df


def risk_tier_to_korean(value: str) -> str:
    mapping = {
        "critical": "즉시 대응",
        "high": "고위험",
        "watch": "관찰 필요",
        "stable": "안정",
    }
    return mapping.get(value, value)


def build_reason_text(row: pd.Series) -> str:
    reasons = []

    if row.get("premium_change_pct", 0) >= 0.12:
        reasons.append("보험료 인상폭이 큼")
    elif row.get("num_price_increases_last_3y", 0) >= 2:
        reasons.append("최근 보험료 인상 횟수가 많음")

    if row.get("late_payment_count_12m", 0) >= 2:
        reasons.append("최근 연체 횟수가 많음")
    elif row.get("missed_payment_flag", 0) == 1:
        reasons.append("납부 누락 이력이 있음")

    if row.get("complaint_flag", 0) == 1:
        reasons.append("민원 이력이 있음")

    if row.get("quote_requested_flag", 0) == 1:
        reasons.append("타사 비교 견적 요청 이력이 있음")

    if row.get("coverage_downgrade_flag", 0) == 1:
        reasons.append("보장 축소 이력이 있음")

    if row.get("customer_tenure_months", 9999) <= 12:
        reasons.append("가입 기간이 짧음")

    if row.get("multi_policy_flag", 1) == 0:
        reasons.append("복수 계약이 아님")

    if not reasons:
        reasons.append("여러 위험 신호가 복합적으로 감지됨")

    return ", ".join(reasons[:3])


@st.cache_data
def score_all_customers() -> pd.DataFrame:
    raw_df = load_raw_data()
    model, threshold = load_model_bundle()
    feature_df = build_feature_frame(raw_df)

    scored_df = raw_df.copy()
    scored_df["churn_probability"] = model.predict_proba(feature_df)[:, 1]
    scored_df["predicted_churn"] = (scored_df["churn_probability"] >= threshold).astype(int)
    scored_df = add_risk_tier(scored_df, threshold)
    scored_df["risk_tier_ko"] = scored_df["risk_tier"].map(risk_tier_to_korean)
    scored_df["prediction_reason"] = scored_df.apply(build_reason_text, axis=1)
    scored_df["coupon_priority"] = (
        scored_df["churn_probability"].rank(method="first", ascending=False).astype(int)
    )
    return scored_df.sort_values(
        by=["predicted_churn", "churn_probability"],
        ascending=[False, False],
    ).reset_index(drop=True)


def refresh_scored_customers_file() -> pd.DataFrame:
    scored_df = score_all_customers()
    output_dir = get_scored_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_df.to_parquet(get_scored_output_path(), index=False)
    load_scored_customers_file.clear()
    return scored_df


@st.cache_data
def load_scored_customers_file() -> pd.DataFrame:
    scored_path = get_scored_output_path()
    if not scored_path.exists():
        raise FileNotFoundError(f"Scored customer file not found: {scored_path}")
    return pd.read_parquet(scored_path)


def summarize_scored_customers(scored_df: pd.DataFrame) -> dict:
    return {
        "total_customers": int(len(scored_df)),
        "predicted_churn_customers": int(scored_df["predicted_churn"].sum()),
        "critical_customers": int((scored_df["risk_tier"] == "critical").sum()),
        "high_customers": int((scored_df["risk_tier"] == "high").sum()),
        "watch_customers": int((scored_df["risk_tier"] == "watch").sum()),
    }
