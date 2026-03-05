# streamlit main page
import streamlit as st

# 페이지 정의
entry_p = st.Page("pages/entry.py", title="홈", icon="🏠", default=True)
churn_predictor_p = st.Page("pages/churn_predictor.py", title="고객이탈예측", icon="🔮")
risk_watchlist_p = st.Page("pages/risk_watchlist.py", title="위험리스트", icon="🚨")

# 내비게이션 실행
pg = st.navigation({
    "Project": [entry_p],
    "Analysis Tools": [churn_predictor_p, risk_watchlist_p]
})

# 이전 페이지와 비교
if "prev_page" not in st.session_state:
    st.session_state.prev_page = pg.title

if st.session_state.prev_page != pg.title:
    st.session_state.prev_page = pg.title

    # session_state 상태 확인 코드
    # st.write(st.session_state)

    # 1. session_state에 유지해야하는 key
    keep_keys = ['prev_page']

    # 2. session_state key 중에 keep_keys에 없는 것만 삭제
    for key in list(st.session_state.keys()):
        if key not in keep_keys:
            del st.session_state[key]

st.markdown("""
    <style>
        div.st-emotion-cache-zy6yx3 {
            padding: 3rem 1rem 10rem !important;
        }
        div.st-emotion-cache-1frkdi4 {
            margin-bottom: -1.5rem !important;
        }
        div.stButton > button p {
            white-space: nowrap !important;
            font-size: 14px !important;
        }
        div.stButton > button {
            min-width: 35px !important;
            width: 100% !important;
            padding: 0px !important;
            margin: 0px 2px !important;
        }
        [data-testid="column"] {
            padding-left: 1px !important;
            padding-right: 1px !important;
        }

        /* st_folium 컨테이너 내부 여백 제거 */
        .element-container:has(iframe) {
            margin-bottom: -10px !important;
        }
    </style>
    """
            , unsafe_allow_html=True)

pg.run()

