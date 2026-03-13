from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.model_service import load_scored_customers_file


st.set_page_config(
    page_title="보험 이탈 예측",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


DESCRIPTION_KO = {
    "Unique customer identifier": "고객 고유 식별자",
    "Reference date for features/label": "피처와 타깃 산출 기준일",
    "Customer region": "고객 지역",
    "Customer age (years)": "고객 나이(연 단위)",
    "Age band bucket": "연령대 구간",
    "Marital status": "혼인 상태",
    "Tenure with insurer in months": "보험사 거래 기간(개월)",
    "1 if customer holds multiple policies, else 0": "여러 보험을 보유하면 1, 아니면 0",
    "Number of policies held by customer (synthetic)": "고객 보유 보험 수(합성 데이터)",
    "Primary policy type": "주요 보험 상품 유형",
    "Policy renewal month (1-12)": "보험 갱신 월(1-12)",
    "Current annualised premium amount (NZD)": "현재 연간 환산 보험료(NZD)",
    "Prior year annualised premium amount (NZD)": "전년도 연간 환산 보험료(NZD)",
    "Premium change percentage vs last year": "전년 대비 보험료 변동률",
    "Count of premium increases in last 3 years (synthetic)": "최근 3년간 보험료 인상 횟수(합성 데이터)",
    "Coverage amount / sum insured (NZD)": "보장 금액 / 가입 금액(NZD)",
    "Premium divided by coverage amount": "보장 금액 대비 보험료 비율",
    "Payment frequency (Monthly/Annual)": "납입 주기(월납/연납)",
    "1 if autopay enabled, else 0": "자동이체 사용 시 1, 아니면 0",
    "Count of late payments in last 12 months": "최근 12개월 연체 횟수",
    "1 if missed payments flag (>=4 late payments), else 0": "미납 위험 플래그(연체 4회 이상이면 1, 아니면 0)",
    "1 if payment method changed recently, else 0": "최근 결제수단 변경 시 1, 아니면 0",
    "Number of claims in last 12 months": "최근 12개월 청구 건수",
    "Number of approved claims in last 12 months": "최근 12개월 승인된 청구 건수",
    "Number of rejected claims in last 12 months": "최근 12개월 거절된 청구 건수",
    "Number of pending claims in last 12 months": "최근 12개월 처리 대기 청구 건수",
    "Average claim amount (NZD)": "평균 청구 금액(NZD)",
    "Total claim amount in last 12 months (NZD)": "최근 12개월 총 청구 금액(NZD)",
    "Total payout amount in last 12 months (NZD)": "최근 12개월 총 지급 금액(NZD)",
    "Total payout divided by total claim amount (last 12 months)": "총 지급 금액 / 총 청구 금액 비율(최근 12개월)",
    "Average settlement time for claims (days)": "청구 평균 처리 기간(일)",
    "Days since last claim (proxy recency)": "마지막 청구 이후 경과 일수",
    "Number of customer contacts in last 12 months": "최근 12개월 고객 접촉 횟수",
    "1 if complaint lodged, else 0": "민원 접수 시 1, 아니면 0",
    "Complaint resolution time in days (0 if no complaint)": "민원 처리 소요 일수(민원 없으면 0)",
    "1 if customer requested a quote (proxy shopping), else 0": "견적 요청 이력이 있으면 1, 아니면 0",
    "1 if customer downgraded coverage, else 0": "보장 축소 이력이 있으면 1, 아니면 0",
    "Target label: 1 = churned at renewal, 0 = retained": "타깃 라벨: 갱신 시 이탈이면 1, 유지면 0",
    "Heuristic churn reason (synthetic)": "휴리스틱 기반 이탈 사유(합성 데이터)",
    "Underlying churn probability used to generate labels (drop before upload if desired)": "라벨 생성용 내부 이탈 확률",
}


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
    color: #25343F;
    margin-bottom: 1.2rem;
    padding-left: 0.2rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_scored_data() -> pd.DataFrame:
    return load_scored_customers_file()


@st.cache_data
def load_data_dictionary() -> pd.DataFrame | None:
    dict_path = Path(__file__).resolve().parent.parent / "data" / "insurance_policyholder_churn_data_dictionary.csv"
    if not dict_path.exists():
        return None

    data_dict = pd.read_csv(dict_path)
    if "description" in data_dict.columns:
        data_dict["description"] = data_dict["description"].map(lambda value: DESCRIPTION_KO.get(value, value))
    return data_dict


df = load_scored_data()
data_dict = load_data_dictionary()

total_customers = int(len(df))
predicted_churn_count = int(df["predicted_churn"].sum())
predicted_churn_rate = (predicted_churn_count / total_customers) * 100 if total_customers else 0
high_risk_count = int(df["risk_tier"].isin(["high", "critical"]).sum())

st.markdown('<div class="main-title">대시보드</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">new 모델 예측 결과 기준 고객 이탈 모니터링 현황</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">전체 고객 수</div>
            <div class="card-value">{total_customers:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">예측 이탈률</div>
            <div class="card-value">{predicted_churn_rate:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">고위험 고객 수</div>
            <div class="card-value">{high_risk_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

left, right = st.columns(2)

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">연령대별 예측 이탈률</div>', unsafe_allow_html=True)
    age_churn = (
        df.groupby("age_band")["predicted_churn"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={"age_band": "연령대", "predicted_churn": "예측 이탈률"})
    )
    st.bar_chart(age_churn.set_index("연령대"))
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">상품별 고객 분포</div>', unsafe_allow_html=True)
    product_counts = df["policy_type"].value_counts().reset_index()
    product_counts.columns = ["상품", "고객 수"]
    fig = px.pie(product_counts, names="상품", values="고객 수", hole=0.35)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### 예측 데이터 미리보기")
preview_cols = [
    "customer_id",
    "region_name",
    "age",
    "age_band",
    "policy_type",
    "current_premium",
    "churn_probability",
    "predicted_churn",
    "risk_tier_ko",
    "prediction_reason",
]
preview_df = df[preview_cols].head(20).copy()
preview_df["churn_probability"] = (preview_df["churn_probability"] * 100).round(1).astype(str) + "%"
preview_df["predicted_churn"] = preview_df["predicted_churn"].map({1: "이탈 예상", 0: "유지 예상"})
preview_df = preview_df.rename(
    columns={
        "customer_id": "고객 ID",
        "region_name": "지역",
        "age": "나이",
        "age_band": "연령대",
        "policy_type": "상품 유형",
        "current_premium": "현재 보험료",
        "churn_probability": "예측 이탈확률",
        "predicted_churn": "예측 결과",
        "risk_tier_ko": "위험 등급",
        "prediction_reason": "예측 사유",
    }
)
st.dataframe(preview_df, width="stretch", hide_index=True)

if data_dict is not None:
    with st.expander("컬럼 설명 보기"):
        st.dataframe(data_dict, width="stretch", hide_index=True)
