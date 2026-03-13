import pandas as pd
import streamlit as st

from src.model_service import (
    evaluate_saved_model,
    get_corr_plot_path,
    get_threshold_plot_path,
    load_model_bundle,
    load_scored_customers_file,
)


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
.card {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
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
    margin-bottom: 0.5rem;
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


@st.cache_data
def load_metrics():
    return evaluate_saved_model()


@st.cache_data
def load_scored_data():
    return load_scored_customers_file()


metrics = load_metrics()
scored_df = load_scored_data()
model, threshold = load_model_bundle()

st.markdown('<div class="main-title">모델 정보</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">현재 앱에서 사용하는 new 모델의 요약 정보와 핵심 지표입니다.</div>',
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].markdown(
    f'<div class="card"><div class="card-title">모델 파일</div><div class="card-value" style="font-size:1.3rem;">churn_model_new.pkl</div></div>',
    unsafe_allow_html=True,
)
metric_cols[1].markdown(
    f'<div class="card"><div class="card-title">모델 계열</div><div class="card-value" style="font-size:1.4rem;">HistGradientBoosting</div></div>',
    unsafe_allow_html=True,
)
metric_cols[2].markdown(
    f'<div class="card"><div class="card-title">Threshold</div><div class="card-value">{threshold:.3f}</div></div>',
    unsafe_allow_html=True,
)
metric_cols[3].markdown(
    f'<div class="card"><div class="card-title">평가 데이터 수</div><div class="card-value">{metrics["test_size"]:,}</div></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.1, 1])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">성능 요약</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame(
        [
            ("Accuracy", round(metrics["accuracy"], 4)),
            ("Precision", round(metrics["precision"], 4)),
            ("Recall", round(metrics["recall"], 4)),
            ("F1", round(metrics["f1"], 4)),
            ("ROC AUC", round(metrics["roc_auc"], 4)),
            ("PR AUC", round(metrics["pr_auc"], 4)),
        ],
        columns=["지표", "값"],
    )
    st.dataframe(summary_df, width="stretch", hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">운영 기준 해석</div>', unsafe_allow_html=True)
    predicted_cnt = int(scored_df["predicted_churn"].sum())
    total_cnt = int(len(scored_df))
    st.markdown(
        f"""
- 현재 앱에서 사용하는 모델은 `new` 모델입니다.
- 전체 고객 `{total_cnt:,}`명 중 `{predicted_cnt:,}`명이 threshold `{threshold:.4f}` 기준으로 이탈 예상 고객으로 분류됩니다.
- 현재 설정은 재현율을 높게 가져가는 방향이라 예측 이탈 고객 수가 상대적으로 많게 나옵니다.
"""
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">분류 결과 요약</div>', unsafe_allow_html=True)
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["실제 유지", "실제 이탈"],
        columns=["예측 유지", "예측 이탈"],
    )
    st.dataframe(cm_df, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">운영 산출물</div>', unsafe_allow_html=True)
    output_df = pd.DataFrame(
        [
            ("위험 등급 컬럼", "risk_tier / risk_tier_ko"),
            ("우선순위 컬럼", "coupon_priority"),
            ("설명 컬럼", "prediction_reason"),
            ("확률 컬럼", "churn_probability"),
        ],
        columns=["항목", "내용"],
    )
    st.dataframe(output_df, width="stretch", hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

plot_cols = st.columns(2)
threshold_plot = get_threshold_plot_path()
corr_plot = get_corr_plot_path()

if threshold_plot.exists():
    plot_cols[0].image(str(threshold_plot), caption="Threshold 분석", width="stretch")
if corr_plot.exists():
    plot_cols[1].image(str(corr_plot), caption="상관관계 시각화", width="stretch")
