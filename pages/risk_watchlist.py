from pathlib import Path
import pandas as pd
import streamlit as st

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="위험 고객 관리",
    page_icon="⚠️",
    layout="wide"
)

# 2. 스타일 정의 (CSS)
st.markdown("""
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

/* 상단 대시보드 카드 스타일 */
.card-danger { background-color: #fff5f5; border: 1px solid #fecaca; border-radius: 18px; padding: 20px; }
.card-warning { background-color: #fffbeb; border: 1px solid #fed7aa; border-radius: 18px; padding: 20px; }
.card-success { background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 18px; padding: 20px; }

.card-label { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; }
.card-number { font-size: 2.2rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)


# 3. 데이터 로드 함수
@st.cache_data
def load_data():
    DATA_PATH = "./data/insurance_policyholder_churn_synthetic.csv"
    try:
        df = pd.read_csv(DATA_PATH)
        df["risk_level"] = pd.cut(
            df["churn_probability_true"],
            bins=[-1, 0.4, 0.7, 1.0],
            labels=["저위험", "중위험", "고위험"]
        )
        return df
    except FileNotFoundError:
        # 데이터가 없을 경우를 대비한 샘플 데이터 생성 (테스트용)
        data = {
            "customer_id": [101, 102, 103],
            "age": [30, 45, 28],
            "policy_type": ["Premium", "Basic", "Gold"],
            "current_premium": [50000, 30000, 80000],
            "churn_probability_true": [0.85, 0.55, 0.2],
            "region_name": ["서울", "경기", "부산"]
        }
        df = pd.DataFrame(data)
        df["risk_level"] = pd.cut(df["churn_probability_true"], bins=[-1, 0.4, 0.7, 1.0], labels=["저위험", "중위험", "고위험"])
        return df


df = load_data()

# 4. 헤더 섹션
st.markdown('<div class="main-title">위험 고객 관리</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">실시간 이탈 위험 고객 모니터링 및 분석</div>', unsafe_allow_html=True)

# 5. 상단 요약 카드 (Metric Cards)
col1, col2, col3 = st.columns(3)

high_cnt = int((df["risk_level"] == "고위험").sum())
mid_cnt = int((df["risk_level"] == "중위험").sum())
low_cnt = int((df["risk_level"] == "저위험").sum())

with col1:
    st.markdown(
        f'<div class="card-danger"><div class="card-label" style="color:#dc2626;">고위험 고객</div><div class="card-number" style="color:#b91c1c;">{high_cnt:,}</div></div>',
        unsafe_allow_html=True)
with col2:
    st.markdown(
        f'<div class="card-warning"><div class="card-label" style="color:#d97706;">중위험 고객</div><div class="card-number" style="color:#c2410c;">{mid_cnt:,}</div></div>',
        unsafe_allow_html=True)
with col3:
    st.markdown(
        f'<div class="card-success"><div class="card-label" style="color:#16a34a;">저위험 고객</div><div class="card-number" style="color:#15803d;">{low_cnt:,}</div></div>',
        unsafe_allow_html=True)

st.write("")  # 간격 조절

# 6. 상세 관리 섹션 (검색 및 데이터프레임)
# 기존 section-card 영역과 글자가 겹쳐서 subheader로 수정했습니다.
with st.container(border=True):
    st.subheader("📋 고객 상세 리스트")

    # 검색 및 필터링 컨트롤러 영역
    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        search = st.text_input("고객 검색", placeholder="고객 ID 또는 지역명을 입력하세요", label_visibility="collapsed")
    with ctrl_col2:
        risk_filter = st.selectbox("위험도 필터", ["모든 위험도", "고위험", "중위험", "저위험"], label_visibility="collapsed")

    # 데이터 필터링 로직
    filtered = df.copy()
    if search:
        keyword = search.strip()
        filtered = filtered[
            filtered["customer_id"].astype(str).str.contains(keyword, case=False, na=False) |
            filtered["region_name"].astype(str).str.contains(keyword, case=False, na=False)
            ]
    if risk_filter != "모든 위험도":
        filtered = filtered[filtered["risk_level"] == risk_filter]

    # 데이터 출력용 포맷팅
    show_cols = ["customer_id", "age", "region_name", "policy_type", "current_premium", "churn_probability_true", "risk_level"]
    result = filtered[show_cols].copy()
    result.columns = ["고객 ID", "나이", "지역", "보험 상품", "월 보험료", "이탈 확률", "위험도"]

    # 가독성을 위한 변환
    result["이탈 확률"] = (result["이탈 확률"] * 100).astype(int)
    result["월 보험료"] = result["월 보험료"].apply(lambda x: f"{int(x):,}원")

    # 최종 테이블 출력
    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True,
        column_config={
            "위험도": st.column_config.TextColumn("위험도", help="이탈 확률에 따른 분류"),
            "이탈 확률": st.column_config.ProgressColumn("이탈 확률", format="%d%%", min_value=0, max_value=100)
        }
    )

# 7. 하단 안내문
st.caption(f"최근 업데이트: {len(result)}명의 고객 데이터가 조회되었습니다.")