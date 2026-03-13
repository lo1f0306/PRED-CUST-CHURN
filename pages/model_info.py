from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from src.preprocess import load_data
from src.model_service import (
    evaluate_saved_model,
    get_corr_plot_path,
    get_threshold_plot_path,
    load_model_bundle,
    score_all_customers,
    build_feature_frame,
)


MODEL_PATH = Path("./model/churn_model_new.pkl")
THRESHOLD = 0.35
RANDOM_STATE = 42
TEST_SIZE = 0.2


st.set_page_config(
    page_title="모델 정보",
    page_icon="⚪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1.05rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}
.model-info-header {
    background: linear-gradient(90deg, #4f46e5 0%, #7e22ce 100%);
    border-radius: 18px;
    padding: 25px 30px;
    color: white;
    margin-bottom: 20px;
}
.model-info-grid {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.card {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    min-height: 120px;
}
.card-title {
    font-size: 1rem;
    color: #475569;
    margin-bottom: 0.6rem;
}
.card-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
}
/* section-card 역할을 하는 컨테이너 스타일 */
.stColumn > div > div > [data-testid="stVerticalBlock"]{
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 18px 20px;
    margin-top: 12px;
}
.section-card {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 18px 20px;
    margin-top: 12px;
}
.section-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.25rem;
}
div[data-testid="stSidebarNav"]::before {
    content: "보험 이탈 예측\\A고객 관리 시스템";
    white-space: pre-line;
    display: block;
    font-size: 2rem;
    line-height: 1.5;
    font-weight: 800;
    color: #2563eb;
    margin-bottom: 1.2rem;
    padding-left: 0.2rem;
}
</style>
""",
    unsafe_allow_html=True,
)
@st.cache_data
def load_metrics():
    return evaluate_saved_model()


@st.cache_data
def load_scored_data():
    return score_all_customers()


metrics = load_metrics()
scored_df = load_scored_data()
_, threshold = load_model_bundle()

@st.cache_resource
def load_ml_resources():
    if not MODEL_PATH.exists():
        return None, None, None

    model = joblib.load(MODEL_PATH)

    df = load_data()

    y = df["churn_flag"].astype(int)
    X = build_feature_frame(df)
    # X = X[list(model.feature_names_in_)]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return model, X_test, y_test


model, X_test, y_test = load_ml_resources()

if model is None:
    st.warning("모델 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= THRESHOLD).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.markdown('<div class="main-title">모델 성능</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">jyhong RandomForest 모델 평가 화면</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="model-info-header">
            <div class="model-info-grid">
                <div>
                    <div style="font-size: 1.5rem; font-weight: 700;">churn_model_jyhong.pkl</div>
                    <div style="opacity: 0.9; font-size: 0.95rem; margin-top: 5px;">
                        평가 데이터 {len(X_test):,}건 / threshold {THRESHOLD:.2f}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="opacity: 0.8; font-size: 0.8rem;">모델 계열</div>
                    <div style="font-size: 1.6rem; font-weight: 800;">RandomForest</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="card"><div class="card-title">Accuracy</div><div class="card-value">{report["accuracy"]:.2%}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="card"><div class="card-title">F1 Score</div><div class="card-value">{report["1"]["f1-score"]:.2%}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="card"><div class="card-title">ROC AUC</div><div class="card-value">{roc_auc:.2%}</div></div>',
            unsafe_allow_html=True,
        )

    # left_col1, right_col1 = st.columns(2)
    #
    # with left_col1:
    #     st.markdown('<div class="section-card">', unsafe_allow_html=True)
    #     st.markdown('<div class="section-title">평가요약</div>', unsafe_allow_html=True)
    #     summary_df = pd.DataFrame(
    #         [
    #             ("테스트 데이터 수", metrics["test_size"]),
    #             ("이탈로 예측한 고객 수", metrics["predicted_positive"]),
    #             ("실제 이탈 고객 수", metrics["actual_positive"]),
    #             ("PR AUC", round(metrics["pr_auc"], 4)),
    #             ("재현율 우선 기준값", round(threshold, 4)),
    #         ],
    #         columns=["지표", "값"],
    #     )
    #     st.dataframe(summary_df, width="stretch", hide_index=True)
    #     st.markdown('</div>', unsafe_allow_html=True)
    # with right_col1:
    #     pass

    left_col, right_col = st.columns(2)

    with left_col:
        with st.container(border=True):
        # st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
            fig_roc = px.line(x=fpr, y=tpr, labels={"x": "False Positive Rate", "y": "True Positive Rate"})
            fig_roc.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350)
            st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        with st.container(border=True):
            # st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["Negative", "Positive"],
                y=["Negative", "Positive"],
                color_continuous_scale="Blues",
            )
            fig_cm.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # with st.container(border=True):
    # st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)

    # 파이프라인 객체인지 확인 후 마지막 모델 단계에서 중요도 추출
    if hasattr(model, 'steps'):
        # 파이프라인일 경우 마지막 step (보통 classifier나 model)에서 가져옴
        final_estimator = model.steps[-1][1]

        # 내장 중요도 속성 존재(feature_importances_)
        if hasattr(final_estimator, 'feature_importances_'):
            importances = final_estimator.feature_importances_
        else:   # 내장 중요도 속성 지원하지 않는 경우 permutation Importance 계산
            X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
            y_sample = y_test.loc[X_sample.index]
            result = permutation_importance(estimator=model,       # 학습된 모델 (Pipeline 객체도 가능)
                                            X=X_sample,            # 평가용 피처 데이터
                                            y=y_sample,            # 평가용 타겟 데이터
                                            scoring='roc_auc',     # 중요도 측정 기준 (이탈 예측이므로 roc_auc 또는 f1 권장)
                                            n_repeats=5,           # 각 피처를 섞는 횟수 (보통 5~10회)
                                            n_jobs=-1,             # 모든 CPU 코어 사용 (계산 속도 향상)
                                            random_state=42        # 결과 재현을 위한 고정값)
                                            )
            importances = result.importances_mean
    else:
        importances = model.feature_importances_

    feat_imp = pd.Series(importances, index=X_test.columns).sort_values(ascending=True).tail(7)
    fig_imp = px.bar(
        x=feat_imp.values,
        y=feat_imp.index,
        orientation="h",
        color_discrete_sequence=["#6366f1"],
    )

    fig_imp.update_layout(xaxis_title="Importance", yaxis_title="Features", height=400)
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
