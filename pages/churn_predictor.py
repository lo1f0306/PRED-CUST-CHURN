import streamlit as st
import pandas as pd
import joblib

st.title("🔍 실시간 이탈 위험 예측")

# 1. 고객 데이터 입력 폼 (UI 컴포넌트 활용)
with st.form("prediction_form"):
    st.subheader("고객 정보 입력")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("나이", min_value=18, max_value=100, value=35)
        marital_status = st.selectbox("결혼 여부", ["Single", "Married", "Divorced"])
    with col2:
        policy_type = st.selectbox("보험 종류", ["Auto", "Home", "Health", "Life"])
        premium = st.number_input("월 납입금", value=100.0)
    with col3:
        claims = st.number_input("최근 1년 청구 횟수", min_value=0, value=0)

    submit_button = st.form_submit_button(label="이탈 위험 예측하기")

# 2. 예측 및 결과 출력
if submit_button:
    # 실제로는 src 모듈의 전처리 로직을 불러와서 처리해야 합니다.
    # 여기서는 UI 흐름 예시만 보여줍니다.
    st.markdown("---")
    st.subheader("예측 결과")

    # 임시 확률값
    mock_prob = 0.75

    if mock_prob > 0.5:
        st.error(f"⚠️ 이탈 위험도: 높음 ({mock_prob * 100}%)")
        # 해당 부분은 특성중요도로 불가능하고 SHAP를 통해 이 데이터를 CHURN=1로 도출하는데 영향을 미친 속성을 출력해줌.
        # SHAP으로 도출된 속성값이랑 특정 워딩을 매핑해놔서 그걸 출력할 수 있어보임.
        RESULT_WORD_DICT = {
            ''
        }
        st.write("**주요 이탈 예상 이유:**")
        st.write("1. 최근 1년 청구 횟수가 낮아 효용성을 느끼지 못할 가능성")
        st.write("2. 'Single' 가구로 다른 보험사로의 이동이 자유로움")
    else:
        st.success(f"✅ 이탈 위험도: 낮음 ({mock_prob * 100}%)")