import streamlit as st
import pandas as pd

st.title("🎯 집중 관리 대상 고객 리스트")

# 임시 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv("././data/insurance_policyholder_churn_synthetic.csv")
    df['churn_prob'] = df['churn_probability_true'] # 실제로는 모델 예측값을 매핑
    return df

df = load_data()

# 필터링 사이드바 UI
st.sidebar.header("필터 설정")
min_prob = st.sidebar.slider("최소 이탈 확률", 0.0, 1.0, 0.7)
selected_policy = st.sidebar.multiselect("보험 종류", df['policy_type'].unique(), default=df['policy_type'].unique())

# 데이터 필터링 적용
filtered_df = df[(df['churn_prob'] >= min_prob) & (df['policy_type'].isin(selected_policy))]

st.write(f"총 **{len(filtered_df)}명**의 집중 관리 대상이 검색되었습니다.")

# 데이터프레임 출력 (그리드 형태)
st.dataframe(filtered_df[['customer_id', 'age', 'marital_status', 'policy_type', 'churn_prob']].sort_values(by='churn_prob', ascending=False))

# CSV 다운로드 기능
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 결과 다운로드 (CSV)",
    data=csv,
    file_name='high_risk_customers.csv',
    mime='text/csv',
)