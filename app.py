import streamlit as st

from src.model_service import refresh_scored_customers_file


st.set_page_config(
    page_title="Dashboard",
    layout="wide",
)

entry_p = st.Page("pages/entry.py", title="홈", icon="🏠", default=True)
churn_predictor_p = st.Page("pages/churn_predictor.py", title="고객이탈예측", icon="🔮")
risk_watchlist_p = st.Page("pages/risk_watchlist.py", title="위험리스트", icon="🚨")
simulation_p = st.Page("pages/simulation_kys.py", title="시뮬레이션", icon="📈")
model_info_p = st.Page("pages/model_info.py", title="모델 정보", icon="⚪")
model_info_2_p = st.Page("pages/model_monitor.py", title="모델 정보 2", icon="🧪")

pg = st.navigation(
    {
        "Project": [entry_p, model_info_p, model_info_2_p],
        "Analysis Tools": [churn_predictor_p, risk_watchlist_p, simulation_p],
    }
)

if "prev_page" not in st.session_state:
    st.session_state.prev_page = pg.title

if st.session_state.prev_page != pg.title:
    st.session_state.prev_page = pg.title

    keep_keys = ["prev_page", "shared_model_cache_warmed"]
    for key in list(st.session_state.keys()):
        if key not in keep_keys:
            del st.session_state[key]

if "shared_model_cache_warmed" not in st.session_state:
    with st.spinner("모델 예측 결과를 준비하는 중입니다..."):
        refresh_scored_customers_file()
    st.session_state.shared_model_cache_warmed = True

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
        .element-container:has(iframe) {
            margin-bottom: -10px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

pg.run()
