import streamlit as st

from src.model_service import refresh_scored_customers_file


st.set_page_config(
    page_title="Dashboard",
    layout="wide",
)

entry_p = st.Page("pages/entry.py", title="홈", icon="🏠", default=True)
churn_predictor_p = st.Page("pages/churn_predictor.py", title="고객이탈예측", icon="🔮")
risk_watchlist_p = st.Page("pages/risk_watchlist.py", title="위험고객관리", icon="🚨")
simulation_p = st.Page("pages/simulation_kys.py", title="시뮬레이션", icon="📈")
model_info_p = st.Page("pages/model_info.py", title="모델성능 대시보드", icon="⚪")
model_info_2_p = st.Page("pages/model_monitor.py", title="자동분석 레포트", icon="🧪")

pg = st.navigation(
    {
        "": [entry_p, model_info_p],
        "예측/시뮬레이션": [model_info_2_p,churn_predictor_p, risk_watchlist_p, simulation_p],
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
    # 앱 시작 시 전체 고객 예측 결과 파일을 먼저 생성합니다.
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
    .stButton > button {
        min-width: 35px !important;
        width: 100% !important;
        padding: 0px !important;
        margin: 0px 2px !important;
        background-color: #ff4b4b; !important;
    }
    .stFormSubmitButton > button {
        background: #ff4b4b; !important;
    }
    div.stButton > button p {
        white-space: nowrap !important;
        font-size: 14px !important;
    }
    
    [data-testid="column"] {
        padding-left: 1px !important;
        padding-right: 1px !important;
    }
    .element-container:has(iframe) {
        margin-bottom: -10px !important;
    }
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
        # border: 1px solid #e5e7eb;
        # border-radius: 18px;
        padding: 18px 20px;
        margin-top: 12px;
        height: 100%;
    }
    .section-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.5rem;
        border-left: 5px solid #034EA2;
        padding-left: 10px;
    }
    
    div[data-testid="stSidebarNav"]::before {
        content: "보험 이탈 예측\\A고객 관리 시스템";
        white-space: pre-line;
        display: block;
        font-size: 2rem;
        line-height: 1.5;
        font-weight: 800;
        color: #034EA2;/*#30364F;#2563eb;*/
        margin-bottom: 1.2rem;
        padding-left: 0.2rem;
    }
    .stSidebar {
        min-width: 320px;
        /*background-color: #ACBAC4;*/
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0 !important;
        width: 0 !important;
        flex: 0 0 0 !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stSidebarNav"]::before {
        display: none !important;
        content: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

pg.run()
