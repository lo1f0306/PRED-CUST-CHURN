from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.model_service import load_model_bundle

st.set_page_config(
    page_title="고객 이탈 예측",
    page_icon="📉",
    layout="wide"
)

# ===============================
# 스타일
# ===============================
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1400px;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: #f8fafc;
}

.main-hero {
    background: linear-gradient(135deg, #97bfb4 0%, #97bfb4 100%);
    padding: 26px 30px;
    border-radius: 22px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
}

.main-title {
    font-size: 2.4rem;
    font-weight: 900;
    color: white;
    margin-bottom: 0.25rem;
}

.sub-title {
    font-size: 1rem;
    color: #cbd5e1;
    margin-bottom: 0;
}

.section-title {
    font-size: 1.65rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 1rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 22px;
    border: 1px solid #e2e8f0 !important;
    background: white;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    padding: 14px;
}

.metric-card-high {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    color: #dd4a48;
    border-radius: 18px;
    padding: 18px 18px;
    margin-bottom: 14px;
    border: 1px solid #fecaca;
}

.metric-card-mid {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    color: #92400e;
    border-radius: 18px;
    padding: 18px 18px;
    margin-bottom: 14px;
    border: 1px solid #fde68a;
}

.metric-card-low {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    color: #166534;
    border-radius: 18px;
    padding: 18px 18px;
    margin-bottom: 14px;
    border: 1px solid #bbf7d0;
}

.metric-label {
    font-size: 0.92rem;
    font-weight: 700;
    opacity: 0.9;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 900;
    line-height: 1.2;
}

.metric-sub {
    font-size: 1rem;
    font-weight: 700;
    margin-top: 0.35rem;
}

.info-chip-wrap {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 10px;
}

.info-chip {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 12px 14px;
}

.info-chip-label {
    font-size: 0.82rem;
    color: #64748b;
    margin-bottom: 0.15rem;
    font-weight: 600;
}

.info-chip-value {
    font-size: 1rem;
    color: #0f172a;
    font-weight: 800;
}

.reason-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 16px 18px;
    margin-top: 14px;
}

.reason-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.6rem;
}

.footer-note {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 14px;
}

/* 입력창 둥글게 */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 14px !important;
}

/* 버튼 */
.stButton > button,
.stFormSubmitButton > button {
    border-radius: 14px !important;
    height: 46px;
    font-weight: 800;
    font-size: 1rem;
    background: #97bfb4;
    color: white;
    border: none;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #1e293b;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 경로
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "insurance_policyholder_churn_synthetic.csv"
MODEL_PATH = BASE_DIR / "model" / "churn_model_new.pkl"
THRESHOLD_PATH = BASE_DIR / "model" / "threshold_new.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return load_model_bundle()

df = load_data()
model, threshold = load_model()

# ===============================
# 파생변수
# ===============================
def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    result = X.copy()
    result['premium_increase_shock'] = np.clip(result['premium_change_pct'], 0, None) * (1 + result['num_price_increases_last_3y'])
    result['premium_jump_flag'] = (result['premium_change_pct'] >= 0.12).astype(int)
    result['payment_risk_score'] = result['late_payment_count_12m'] + result['missed_payment_flag'] * 2
    result['service_risk_score'] = result['complaint_flag'] + result['quote_requested_flag'] + result['coverage_downgrade_flag']
    result['engagement_risk_score'] = result['payment_risk_score'] + result['service_risk_score']
    result['tenure_inverse'] = 1 / (result['customer_tenure_months'] + 1)
    result['single_policy_short_tenure'] = ((result['multi_policy_flag'] == 0) & (result['customer_tenure_months'] <= 24)).astype(int)
    result['premium_x_complaint'] = result['premium_jump_flag'] * result['complaint_flag']
    result['premium_x_quote'] = result['premium_jump_flag'] * result['quote_requested_flag']
    result['late_x_quote'] = (result['late_payment_count_12m'] >= 2).astype(int) * result['quote_requested_flag']
    result['downgrade_x_quote'] = result['coverage_downgrade_flag'] * result['quote_requested_flag']
    result['monthly_payment_flag'] = (result['payment_frequency'] == 'Monthly').astype(int)
    result['monthly_x_premium_jump'] = result['monthly_payment_flag'] * result['premium_jump_flag']
    result['auto_or_health'] = result['policy_type'].isin(['Auto', 'Health']).astype(int)
    return result

# ===============================
# 입력 1행 생성
# ===============================
def build_input_row(
    raw_df: pd.DataFrame,
    age: int,
    policy_type: str,
    tenure: int,
    premium: float,
    late_payment_count: int,
    quote_requested_flag: int,
    num_claims_12m: int
) -> pd.DataFrame:
    drop_cols = [
        'customer_id',
        'as_of_date',
        'churn_flag',
        'churn_type',
        'churn_probability_true'
    ]

    feature_df = raw_df.drop(columns=[c for c in drop_cols if c in raw_df.columns]).copy()

    base_row = {}
    for col in feature_df.columns:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            base_row[col] = float(feature_df[col].median())
        else:
            mode_val = feature_df[col].mode(dropna=True)
            base_row[col] = mode_val.iloc[0] if not mode_val.empty else ""

    base_row["age"] = age
    base_row["policy_type"] = policy_type
    base_row["customer_tenure_months"] = tenure
    base_row["current_premium"] = premium
    base_row["late_payment_count_12m"] = late_payment_count
    base_row["quote_requested_flag"] = quote_requested_flag
    base_row["num_claims_12m"] = num_claims_12m

    if "missed_payment_flag" in base_row:
        base_row["missed_payment_flag"] = 1 if late_payment_count >= 4 else 0

    if "premium_last_year" in base_row:
        base_row["premium_last_year"] = premium / 1.05

    if "premium_change_pct" in base_row:
        prev = base_row["premium_last_year"]
        base_row["premium_change_pct"] = (premium - prev) / (prev + 1e-6)

    if "num_price_increases_last_3y" in base_row:
        base_row["num_price_increases_last_3y"] = 2 if quote_requested_flag == 1 else 1

    if "num_pending_claims_12m" in base_row:
        base_row["num_pending_claims_12m"] = 0
    if "num_rejected_claims_12m" in base_row:
        base_row["num_rejected_claims_12m"] = 0
    if "num_approved_claims_12m" in base_row:
        base_row["num_approved_claims_12m"] = num_claims_12m

    if "avg_claim_amount" in base_row and num_claims_12m == 0:
        base_row["avg_claim_amount"] = 0

    if "total_claim_amount_12m" in base_row:
        avg_claim = base_row.get("avg_claim_amount", 0)
        base_row["total_claim_amount_12m"] = avg_claim * num_claims_12m

    if "total_payout_amount_12m" in base_row:
        payout_ratio = base_row.get("payout_ratio_12m", 0.8)
        total_claim = base_row.get("total_claim_amount_12m", 0)
        base_row["total_payout_amount_12m"] = total_claim * payout_ratio

    if "payment_frequency" in base_row and (base_row["payment_frequency"] == "" or pd.isna(base_row["payment_frequency"])):
        base_row["payment_frequency"] = "Monthly"

    if "age_band" in base_row:
        if age < 25:
            base_row["age_band"] = "18-24"
        elif age < 35:
            base_row["age_band"] = "25-34"
        elif age < 45:
            base_row["age_band"] = "35-44"
        elif age < 55:
            base_row["age_band"] = "45-54"
        elif age < 65:
            base_row["age_band"] = "55-64"
        elif age < 75:
            base_row["age_band"] = "65-74"
        else:
            base_row["age_band"] = "75+"

    input_df = pd.DataFrame([base_row])
    input_df = add_engineered_features(input_df)
    return input_df

policy_options = sorted(df["policy_type"].dropna().unique().tolist())

# ===============================
# 헤더
# ===============================
st.markdown("""
<div class="main-hero">
    <div class="main-title">고객 이탈 예측</div>
    <div class="sub-title">AI 기반 고객 이탈 예측 분석 · 주요 고객 위험 신호를 빠르게 확인할 수 있습니다.</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="section-title">고객 정보 입력</div>', unsafe_allow_html=True)

        with st.form("predict_form"):
            age = st.number_input("나이", min_value=18, max_value=100, value=35)
            policy_type = st.selectbox("보험 상품", policy_options)
            premium = st.number_input("현재 보험료 (원)", min_value=0, value=150000, step=10000)
            tenure = st.number_input("가입 기간 (개월)", min_value=1, max_value=600, value=24)
            late_payment_count = st.number_input("최근 1년 연체 횟수", min_value=0, max_value=20, value=0)
            quote_requested_flag = st.selectbox("견적 요청 여부", [0, 1], format_func=lambda x: "아니오" if x == 0 else "예")
            num_claims_12m = st.number_input("최근 1년 청구 횟수", min_value=0, max_value=20, value=0)

            submitted = st.form_submit_button("예측하기", use_container_width=True)

with right:
    with st.container(border=True):
        st.markdown('<div class="section-title">예측 결과</div>', unsafe_allow_html=True)

        if submitted:
            input_df = build_input_row(
                raw_df=df,
                age=age,
                policy_type=policy_type,
                tenure=tenure,
                premium=premium,
                late_payment_count=late_payment_count,
                quote_requested_flag=quote_requested_flag,
                num_claims_12m=num_claims_12m
            )

            prob = model.predict_proba(input_df)[:, 1][0]
            pred = 1 if prob >= threshold else 0

            if prob >= 0.70:
                risk = "고위험"
                card_class = "metric-card-high"
            elif prob >= 0.40:
                risk = "중위험"
                card_class = "metric-card-mid"
            else:
                risk = "저위험"
                card_class = "metric-card-low"

            st.markdown(f"""
            <div class="{card_class}">
                <div class="metric-label">예측 결과</div>
                <div class="metric-value">이탈 확률 {prob*100:.1f}%</div>
                <div class="metric-sub">위험도: {risk}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-chip-wrap">
                <div class="info-chip">
                    <div class="info-chip-label">나이</div>
                    <div class="info-chip-value">{age}세</div>
                </div>
                <div class="info-chip">
                    <div class="info-chip-label">보험 상품</div>
                    <div class="info-chip-value">{policy_type}</div>
                </div>
                <div class="info-chip">
                    <div class="info-chip-label">현재 보험료</div>
                    <div class="info-chip-value">{premium:,}원</div>
                </div>
                <div class="info-chip">
                    <div class="info-chip-label">가입 기간</div>
                    <div class="info-chip-value">{tenure}개월</div>
                </div>
                <div class="info-chip">
                    <div class="info-chip-label">연체 횟수</div>
                    <div class="info-chip-value">{late_payment_count}회</div>
                </div>
                <div class="info-chip">
                    <div class="info-chip-label">견적 요청</div>
                    <div class="info-chip-value">{"예" if quote_requested_flag == 1 else "아니오"}</div>
                </div>
                <div class="info-chip">
                    <div class="info-chip-label">청구 횟수</div>
                    <div class="info-chip-value">{num_claims_12m}회</div>
                </div>
                <div class="info-chip">
                    <div class="info-chip-label">예측 클래스</div>
                    <div class="info-chip-value">{"이탈 위험 고객" if pred == 1 else "유지 가능 고객"}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            reasons = []
            if late_payment_count >= 2:
                reasons.append("연체 횟수가 높아 납부 불안정 신호가 있습니다.")
            if quote_requested_flag == 1:
                reasons.append("비교 견적 요청 이력이 있어 이탈 가능성이 높아질 수 있습니다.")
            if tenure <= 24:
                reasons.append("가입 기간이 짧아 계약 결속도가 낮을 수 있습니다.")
            if premium >= 200000:
                reasons.append("보험료 수준이 높아 가격 부담이 작용할 수 있습니다.")
            if num_claims_12m >= 3:
                reasons.append("최근 청구 이력이 많아 서비스 경험이 이탈에 영향을 줄 수 있습니다.")

            st.markdown('<div class="reason-box">', unsafe_allow_html=True)
            st.markdown('<div class="reason-title">해석 포인트</div>', unsafe_allow_html=True)

            if reasons:
                for reason in reasons[:3]:
                    st.write(f"- {reason}")
            else:
                st.write("- 입력 정보 기준 뚜렷한 고위험 신호는 상대적으로 적습니다.")

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="footer-note">모델 기준 threshold: {threshold:.4f}</div>',
                unsafe_allow_html=True
            )

        else:
            st.info("좌측 양식을 작성하고 예측 버튼을 클릭하세요.")
