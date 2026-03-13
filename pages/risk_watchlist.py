import pandas as pd
import streamlit as st

from src.model_service import (
    load_model_bundle,
    load_scored_customers_file,
    summarize_scored_customers,
)


pd.set_option("styler.render.max_elements", 500000)

st.set_page_config(
    page_title="위험 고객 관리",
    page_icon="🚨",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
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
    margin-bottom: 2rem;
}
.card-danger { background-color: #fff5f5; border: 1px solid #fecaca; border-radius: 18px; padding: 20px; }
.card-warning { background-color: #fffbeb; border: 1px solid #fed7aa; border-radius: 18px; padding: 20px; }
.card-success { background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 18px; padding: 20px; }
.card-label { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; }
.card-number { font-size: 2.2rem; font-weight: 800; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_scored_data() -> pd.DataFrame:
    return load_scored_customers_file()


df = load_scored_data()
summary = summarize_scored_customers(df)
_, threshold = load_model_bundle()

st.markdown('<div class="main-title">위험 고객 관리</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">new 모델 예측 기준 위험 고객 모니터링 및 필터링</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<div class="card-danger"><div class="card-label" style="color:#dc2626;">고위험 고객</div><div class="card-number" style="color:#b91c1c;">{summary["critical_customers"] + summary["high_customers"]:,}</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="card-warning"><div class="card-label" style="color:#d97706;">관찰 필요 고객</div><div class="card-number" style="color:#c2410c;">{summary["watch_customers"]:,}</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    stable_count = summary["total_customers"] - summary["predicted_churn_customers"]
    st.markdown(
        f'<div class="card-success"><div class="card-label" style="color:#16a34a;">유지 예상 고객</div><div class="card-number" style="color:#15803d;">{stable_count:,}</div></div>',
        unsafe_allow_html=True,
    )

st.caption(f"현재 이탈 분류 threshold: {threshold:.4f}")
st.write("")

with st.container(border=True):
    st.subheader("고객 상세 리스트")

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 2, 1, 1])

    with ctrl_col1:
        search_category = st.selectbox("검색 조건", ["전체", "고객 ID", "지역"], label_visibility="collapsed")
    with ctrl_col2:
        search_keyword = st.text_input(
            "검색어 입력",
            placeholder=f"{search_category} 기준으로 검색합니다",
            label_visibility="collapsed",
        )
    with ctrl_col3:
        risk_filter = st.selectbox("위험등급 필터", ["전체", "즉시 대응", "고위험", "관찰 필요", "안정"], label_visibility="collapsed")
    with ctrl_col4:
        predicted_only = st.checkbox("예측 이탈만", value=True)

    filtered = df.copy()

    if predicted_only:
        filtered = filtered[filtered["predicted_churn"] == 1]

    if search_keyword:
        keyword = search_keyword.strip()
        if search_category == "전체":
            filtered = filtered[
                filtered["customer_id"].astype(str).str.contains(keyword, case=False, na=False)
                | filtered["region_name"].astype(str).str.contains(keyword, case=False, na=False)
            ]
        elif search_category == "고객 ID":
            filtered = filtered[filtered["customer_id"].astype(str).str.contains(keyword, case=False, na=False)]
        else:
            filtered = filtered[filtered["region_name"].astype(str).str.contains(keyword, case=False, na=False)]

    if risk_filter != "전체":
        filtered = filtered[filtered["risk_tier_ko"] == risk_filter]

    show_cols = [
        "coupon_priority",
        "customer_id",
        "region_name",
        "age",
        "policy_type",
        "customer_tenure_months",
        "current_premium",
        "churn_probability",
        "risk_tier_ko",
        "prediction_reason",
    ]
    result = filtered[show_cols].copy()
    result["customer_tenure_months"] = (result["customer_tenure_months"] / 12).map("{:.1f}".format)
    result["churn_probability"] = (result["churn_probability"] * 100).round(1)

    result.columns = [
        "우선순위",
        "고객 ID",
        "지역",
        "나이",
        "상품 유형",
        "가입기간",
        "현재 보험료",
        "예측 이탈확률",
        "위험등급",
        "예측 사유",
    ]
    result["가입기간"] = result["가입기간"].apply(lambda x: f"{x}년")
    result["현재 보험료"] = result["현재 보험료"].apply(lambda x: f"{int(x):,}원")

    def highlight_risk(row):
        style = [""] * len(row)
        risk = row["위험등급"]
        if risk == "즉시 대응":
            color = "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
        elif risk == "고위험":
            color = "background-color: #ffedd5; color: #9a3412; font-weight: bold;"
        elif risk == "관찰 필요":
            color = "background-color: #fef3c7; color: #92400e; font-weight: bold;"
        else:
            color = "background-color: #f0fdf4; color: #166534; font-weight: bold;"
        style[row.index.get_loc("위험등급")] = color
        return style

    styled_df = result.style.apply(highlight_risk, axis=1)

    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        column_config={
            "위험등급": st.column_config.TextColumn("위험등급", help="new 모델 기준 위험 분류"),
            "예측 이탈확률": st.column_config.ProgressColumn(
                "예측 이탈확률",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

st.caption(f"조회 결과: {len(result):,}명")
