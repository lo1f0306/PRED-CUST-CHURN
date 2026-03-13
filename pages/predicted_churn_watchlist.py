import streamlit as st

from src.model_service import (
    load_model_bundle,
    load_scored_customers_file,
    summarize_scored_customers,
)


st.set_page_config(page_title="이탈 예상 고객 목록", page_icon="🎯", layout="wide")


@st.cache_data
def load_scored_data():
    return load_scored_customers_file()


scored_df = load_scored_data()
summary = summarize_scored_customers(scored_df)
_, threshold = load_model_bundle()

st.title("이탈 예상 고객 목록")
st.caption("저장된 모델과 기준값으로 전체 고객을 다시 점수화한 결과입니다.")

card_cols = st.columns(5)
card_cols[0].metric("전체 고객 수", f"{summary['total_customers']:,}")
card_cols[1].metric("이탈 예상 고객", f"{summary['predicted_churn_customers']:,}")
card_cols[2].metric("즉시 대응", f"{summary['critical_customers']:,}")
card_cols[3].metric("고위험", f"{summary['high_customers']:,}")
card_cols[4].metric("관찰 필요", f"{summary['watch_customers']:,}")

st.info(f"현재 이탈 분류 기준값: {threshold:.4f}")

left, right, third, fourth = st.columns(4)
with left:
    show_predicted_only = st.checkbox("이탈 예상 고객만 보기", value=True)
with right:
    tier_filter = st.selectbox("위험 등급", ["전체", "즉시 대응", "고위험", "관찰 필요", "안정"])
with third:
    policy_filter = st.selectbox(
        "상품 유형",
        ["전체"] + sorted(scored_df["policy_type"].dropna().unique().tolist()),
    )
with fourth:
    region_filter = st.selectbox(
        "지역",
        ["전체"] + sorted(scored_df["region_name"].dropna().unique().tolist()),
    )

keyword = st.text_input("고객 ID 또는 지역 검색")
top_n = st.slider("표시할 행 수", min_value=20, max_value=500, value=100, step=20)

filtered = scored_df.copy()
if show_predicted_only:
    filtered = filtered[filtered["predicted_churn"] == 1]
if tier_filter != "전체":
    filtered = filtered[filtered["risk_tier_ko"] == tier_filter]
if policy_filter != "전체":
    filtered = filtered[filtered["policy_type"] == policy_filter]
if region_filter != "전체":
    filtered = filtered[filtered["region_name"] == region_filter]
if keyword:
    filtered = filtered[
        filtered["customer_id"].astype(str).str.contains(keyword, case=False, na=False)
        | filtered["region_name"].astype(str).str.contains(keyword, case=False, na=False)
    ]

display_df = filtered[
    [
        "coupon_priority",
        "customer_id",
        "region_name",
        "policy_type",
        "current_premium",
        "churn_probability",
        "risk_tier_ko",
        "prediction_reason",
    ]
].head(top_n).copy()

display_df["current_premium"] = display_df["current_premium"].round(0).astype(int)
display_df["churn_probability"] = (display_df["churn_probability"] * 100).round(2)
display_df = display_df.rename(
    columns={
        "coupon_priority": "우선순위",
        "customer_id": "고객 ID",
        "region_name": "지역",
        "policy_type": "상품 유형",
        "current_premium": "현재 보험료",
        "churn_probability": "이탈 예상 확률(%)",
        "risk_tier_ko": "위험 등급",
        "prediction_reason": "이탈 예상 이유",
    }
)

st.dataframe(display_df, width="stretch", hide_index=True)

csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "현재 목록 CSV 다운로드",
    data=csv_data,
    file_name="predicted_churn_watchlist.csv",
    mime="text/csv",
)
