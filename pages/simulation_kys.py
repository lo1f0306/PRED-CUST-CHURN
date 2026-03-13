import sklearn
import streamlit
import pandas




from pathlib import Path
import platform

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from src.model_service import load_model_bundle


# ==============================
# 페이지 설정
# ==============================
st.set_page_config(
    page_title="What-If 비즈니스 시뮬레이션",
    page_icon="📈",
    layout="wide"
)

# ==============================
# 경로 설정
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "insurance_policyholder_churn_synthetic.csv"
MODEL_PATH = BASE_DIR / "model" / "churn_model_new.pkl"
THRESHOLD_PATH = BASE_DIR / "model" / "threshold_new.pkl"

TARGET_COL = "churn_flag"
DROP_COLS = ["customer_id", "as_of_date", "churn_type", "churn_probability_true"]

# ==============================
# 한글 폰트 설정
# ==============================
def set_korean_font():
    system_name = platform.system()

    if system_name == "Windows":
        plt.rc("font", family="Malgun Gothic")
    elif system_name == "Darwin":
        plt.rc("font", family="AppleGothic")
    else:
        font_candidates = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for font_path in font_candidates:
            if Path(font_path).exists():
                fontprop = fm.FontProperties(fname=font_path)
                plt.rc("font", family=fontprop.get_name())
                break

    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()

# ==============================
# 스타일
# ==============================
st.markdown("""
<style>
.block-container {
    padding-top: 1.05rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
.big-title {
    font-size: 2.35rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.15rem;
}
.sub-text {
    font-size: 1.02rem;
    color: #475569;
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 0.35rem;
    margin-bottom: 0.9rem;
}
.info-card {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
    margin-bottom: 0.8rem;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 14px 16px;
    border-radius: 18px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}
div[data-testid="stMetricLabel"] {
    color: #64748b;
}
div[data-testid="stMetricValue"] {
    color: #0f172a;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 데이터 / 모델 로드
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return load_model_bundle()

# ==============================
# 노트북 파생변수 그대로 사용
# ==============================
def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    result = X.copy()
    result["premium_increase_shock"] = np.clip(result["premium_change_pct"], 0, None) * (
        1 + result["num_price_increases_last_3y"]
    )
    result["premium_jump_flag"] = (result["premium_change_pct"] >= 0.12).astype(int)
    result["payment_risk_score"] = result["late_payment_count_12m"] + result["missed_payment_flag"] * 2
    result["service_risk_score"] = (
        result["complaint_flag"] + result["quote_requested_flag"] + result["coverage_downgrade_flag"]
    )
    result["engagement_risk_score"] = result["payment_risk_score"] + result["service_risk_score"]
    result["tenure_inverse"] = 1 / (result["customer_tenure_months"] + 1)
    result["single_policy_short_tenure"] = (
        (result["multi_policy_flag"] == 0) & (result["customer_tenure_months"] <= 24)
    ).astype(int)
    result["premium_x_complaint"] = result["premium_jump_flag"] * result["complaint_flag"]
    result["premium_x_quote"] = result["premium_jump_flag"] * result["quote_requested_flag"]
    result["late_x_quote"] = (result["late_payment_count_12m"] >= 2).astype(int) * result["quote_requested_flag"]
    result["downgrade_x_quote"] = result["coverage_downgrade_flag"] * result["quote_requested_flag"]
    result["monthly_payment_flag"] = (result["payment_frequency"] == "Monthly").astype(int)
    result["monthly_x_premium_jump"] = result["monthly_payment_flag"] * result["premium_jump_flag"]
    result["auto_or_health"] = result["policy_type"].isin(["Auto", "Health"]).astype(int)
    return result

# ==============================
# 모델 입력 생성
# ==============================
def make_model_input(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    if TARGET_COL in X.columns:
        X = X.drop(columns=[TARGET_COL])

    usable_drop_cols = [c for c in DROP_COLS if c in X.columns]
    X = X.drop(columns=usable_drop_cols, errors="ignore")

    X = add_engineered_features(X)
    return X

# ==============================
# 위험 점수 계산
# ==============================
def compute_priority_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)

    if "premium_change_pct" in df.columns:
        score += df["premium_change_pct"].fillna(0) * 12

    if "num_price_increases_last_3y" in df.columns:
        score += df["num_price_increases_last_3y"].fillna(0) * 2.5

    if "late_payment_count_12m" in df.columns:
        score += df["late_payment_count_12m"].fillna(0) * 3.5

    if "missed_payment_flag" in df.columns:
        score += df["missed_payment_flag"].fillna(0) * 4.0

    if "complaint_flag" in df.columns:
        score += df["complaint_flag"].fillna(0) * 4.5

    if "quote_requested_flag" in df.columns:
        score += df["quote_requested_flag"].fillna(0) * 3.0

    if "coverage_downgrade_flag" in df.columns:
        score += df["coverage_downgrade_flag"].fillna(0) * 3.0

    if "multi_policy_flag" in df.columns:
        score += (df["multi_policy_flag"].fillna(0) == 0).astype(int) * 1.5

    if "customer_tenure_months" in df.columns:
        score += (df["customer_tenure_months"].fillna(999) <= 24).astype(int) * 2.0

    return score

# ==============================
# 상위 위험 고객 선택
# ==============================
def select_top_risky_indices(base_df: pd.DataFrame, condition_mask: pd.Series, pct: int) -> pd.Index:
    candidate_idx = base_df.index[condition_mask].tolist()
    if len(candidate_idx) == 0 or pct <= 0:
        return pd.Index([])

    priority_score = compute_priority_score(base_df)
    ranked_idx = priority_score.loc[candidate_idx].sort_values(ascending=False).index

    n = int(np.ceil(len(ranked_idx) * pct / 100))
    n = min(max(n, 0), len(ranked_idx))
    return ranked_idx[:n]

# ==============================
# 시뮬레이션 반영
# ==============================
def apply_simulation_scenario(
    df: pd.DataFrame,
    price_relief_pct: int,
    reduce_price_jump_customers_pct: int,
    reduce_late_risk_customers_pct: int,
    reduce_complaint_customers_pct: int,
    reduce_quote_requested_customers_pct: int,
    reduce_downgrade_customers_pct: int,
) -> pd.DataFrame:
    sim_df = df.copy()

    if "premium_change_pct" in sim_df.columns and price_relief_pct > 0:
        sim_df["premium_change_pct"] = sim_df["premium_change_pct"] * (1 - price_relief_pct / 100)

    if "premium_change_pct" in sim_df.columns:
        chosen = select_top_risky_indices(
            base_df=sim_df,
            condition_mask=sim_df["premium_change_pct"] >= 0.12,
            pct=reduce_price_jump_customers_pct
        )
        if len(chosen) > 0:
            sim_df.loc[chosen, "premium_change_pct"] = 0.119

    if "num_price_increases_last_3y" in sim_df.columns:
        sim_df["num_price_increases_last_3y"] = np.maximum(
            sim_df["num_price_increases_last_3y"] - 1, 0
        )

    if "late_payment_count_12m" in sim_df.columns:
        chosen = select_top_risky_indices(
            base_df=sim_df,
            condition_mask=sim_df["late_payment_count_12m"] >= 2,
            pct=reduce_late_risk_customers_pct
        )
        if len(chosen) > 0:
            sim_df.loc[chosen, "late_payment_count_12m"] = 1
            if "missed_payment_flag" in sim_df.columns:
                sim_df.loc[chosen, "missed_payment_flag"] = 0

    if "complaint_flag" in sim_df.columns:
        chosen = select_top_risky_indices(
            base_df=sim_df,
            condition_mask=sim_df["complaint_flag"] == 1,
            pct=reduce_complaint_customers_pct
        )
        if len(chosen) > 0:
            sim_df.loc[chosen, "complaint_flag"] = 0

    if "quote_requested_flag" in sim_df.columns:
        chosen = select_top_risky_indices(
            base_df=sim_df,
            condition_mask=sim_df["quote_requested_flag"] == 1,
            pct=reduce_quote_requested_customers_pct
        )
        if len(chosen) > 0:
            sim_df.loc[chosen, "quote_requested_flag"] = 0

    if "coverage_downgrade_flag" in sim_df.columns:
        chosen = select_top_risky_indices(
            base_df=sim_df,
            condition_mask=sim_df["coverage_downgrade_flag"] == 1,
            pct=reduce_downgrade_customers_pct
        )
        if len(chosen) > 0:
            sim_df.loc[chosen, "coverage_downgrade_flag"] = 0

    return sim_df

# ==============================
# 예측 함수
# ==============================
def predict_churn(df: pd.DataFrame, model, threshold: float):
    X = make_model_input(df)
    probs = model.predict_proba(X)[:, 1]
    pred = (probs >= threshold).astype(int)

    return {
        "X": X,
        "probs": probs,
        "pred": pred,
        "churn_rate": pred.mean() * 100,
        "avg_prob": probs.mean() * 100,
        "churn_count": int(pred.sum()),
    }

# ==============================
# 정책별 단독 효과 비교
# ==============================
def run_single_policy_simulations(
    df: pd.DataFrame,
    model,
    threshold: float,
    price_relief_pct: int,
    reduce_price_jump_customers_pct: int,
    reduce_late_risk_customers_pct: int,
    reduce_complaint_customers_pct: int,
    reduce_quote_requested_customers_pct: int,
    reduce_downgrade_customers_pct: int,
):
    base = predict_churn(df, model, threshold)

    scenarios = [
        {
            "정책": "보험료 인상폭 완화",
            "params": dict(
                price_relief_pct=price_relief_pct,
                reduce_price_jump_customers_pct=0,
                reduce_late_risk_customers_pct=0,
                reduce_complaint_customers_pct=0,
                reduce_quote_requested_customers_pct=0,
                reduce_downgrade_customers_pct=0,
            ),
        },
        {
            "정책": "급격한 보험료 인상 고객 완화",
            "params": dict(
                price_relief_pct=0,
                reduce_price_jump_customers_pct=reduce_price_jump_customers_pct,
                reduce_late_risk_customers_pct=0,
                reduce_complaint_customers_pct=0,
                reduce_quote_requested_customers_pct=0,
                reduce_downgrade_customers_pct=0,
            ),
        },
        {
            "정책": "고연체 고객 정상화",
            "params": dict(
                price_relief_pct=0,
                reduce_price_jump_customers_pct=0,
                reduce_late_risk_customers_pct=reduce_late_risk_customers_pct,
                reduce_complaint_customers_pct=0,
                reduce_quote_requested_customers_pct=0,
                reduce_downgrade_customers_pct=0,
            ),
        },
        {
            "정책": "민원 고객 해소",
            "params": dict(
                price_relief_pct=0,
                reduce_price_jump_customers_pct=0,
                reduce_late_risk_customers_pct=0,
                reduce_complaint_customers_pct=reduce_complaint_customers_pct,
                reduce_quote_requested_customers_pct=0,
                reduce_downgrade_customers_pct=0,
            ),
        },
        {
            "정책": "비교 견적 요청 감소",
            "params": dict(
                price_relief_pct=0,
                reduce_price_jump_customers_pct=0,
                reduce_late_risk_customers_pct=0,
                reduce_complaint_customers_pct=0,
                reduce_quote_requested_customers_pct=reduce_quote_requested_customers_pct,
                reduce_downgrade_customers_pct=0,
            ),
        },
        {
            "정책": "보장 축소 고객 감소",
            "params": dict(
                price_relief_pct=0,
                reduce_price_jump_customers_pct=0,
                reduce_late_risk_customers_pct=0,
                reduce_complaint_customers_pct=0,
                reduce_quote_requested_customers_pct=0,
                reduce_downgrade_customers_pct=reduce_downgrade_customers_pct,
            ),
        },
    ]

    rows = []
    for scenario in scenarios:
        sim_df_single = apply_simulation_scenario(df=df, **scenario["params"])
        sim_result_single = predict_churn(sim_df_single, model, threshold)
        rows.append(
            {
                "정책": scenario["정책"],
                "평균 이탈확률 감소폭": base["avg_prob"] - sim_result_single["avg_prob"],
                "예상 이탈률 감소폭": base["churn_rate"] - sim_result_single["churn_rate"],
                "방어 고객 수": base["churn_count"] - sim_result_single["churn_count"],
            }
        )

    return pd.DataFrame(rows).sort_values("평균 이탈확률 감소폭", ascending=True)

# ==============================
# 막대그래프 함수
# ==============================
def draw_pretty_bar_chart(title, ylabel, before_value, after_value, before_color, after_color):
    fig, ax = plt.subplots(figsize=(6.2, 4.45), facecolor="white")
    ax.set_facecolor("white")

    labels = ["원본", "시뮬레이션"]
    values = [before_value, after_value]
    colors = [before_color, after_color]

    x = np.array([0.10, 0.7])

    bars = ax.bar(
        x,
        values,
        color=colors,
        width=0.15,
        edgecolor="none",
        zorder=3
    )

    # 축 스타일
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # x축 라벨 크게
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15, fontweight="bold", color="#334155")

    # y축 라벨 작게
    ax.tick_params(axis="y", labelsize=9, colors="#64748B")
    ax.set_ylabel(ylabel, fontsize=10, color="#475569", labelpad=8)

    # 제목
    ax.set_title(title, fontsize=18, fontweight="bold", color="#0F172A", pad=12)

    # 그리드
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.16)
    ax.set_axisbelow(True)

    # 여백
    y_max = max(values)
    ax.set_ylim(0, y_max * 1.17 if y_max > 0 else 1)
    ax.set_xlim(-0.22, 1.0)

    # 값 라벨
    value_colors = ["#111827", "#576A8F"]
    for idx, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + y_max * 0.012,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color=value_colors[idx]
        )

    # 감소 배지
    diff = before_value - after_value
    if diff >= 0:
        diff_text = f"▼ {diff:.2f}%p 감소"
        badge_fc = "#ECFDF5"
        badge_tc = "#047857"
    else:
        diff_text = f"▲ {abs(diff):.2f}%p 증가"
        badge_fc = "#FEF2F2"
        badge_tc = "#B91C1C"

    ax.text(
        0.5,
        0.95,
        diff_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color=badge_tc,
        bbox=dict(
            boxstyle="round,pad=0.34",
            facecolor=badge_fc,
            edgecolor="none"
        )
    )

    return fig

# ==============================
# 헤더
# ==============================
st.markdown('<div class="big-title">📈 What-If 비즈니스 시뮬레이션</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text"><b>정책을 바꾸면 이탈률은 어떻게 달라질까?</b> '
    '학습에 사용한 실제 파생변수를 그대로 적용해 정책 변화 효과를 시뮬레이션합니다.</div>',
    unsafe_allow_html=True
)

# ==============================
# 사이드바
# ==============================
st.sidebar.markdown("## 🛠 시뮬레이션 시나리오 설정")

price_relief_pct = st.sidebar.slider("전체 보험료 인상폭 완화 (%)", 0, 80, 40, 10)
reduce_price_jump_customers_pct = st.sidebar.slider("급격한 보험료 인상 고객 완화 (%)", 0, 100, 60, 10)
reduce_late_risk_customers_pct = st.sidebar.slider("고연체 고객 정상화 (%)", 0, 100, 60, 10)
reduce_complaint_customers_pct = st.sidebar.slider("민원 고객 해소 (%)", 0, 100, 60, 10)
reduce_quote_requested_customers_pct = st.sidebar.slider("비교 견적 요청 감소 (%)", 0, 100, 50, 10)
reduce_downgrade_customers_pct = st.sidebar.slider("보장 축소 고객 감소 (%)", 0, 100, 50, 10)

clv = st.sidebar.number_input(
    "고객 1명당 예상 유지 가치(원)",
    min_value=0,
    value=1200000,
    step=100000
)

run_simulation = st.sidebar.button("🚀 시뮬레이션 실행", use_container_width=True)

# ==============================
# 로드
# ==============================
try:
    df = load_data()
    model, threshold = load_model()
except Exception as e:
    st.error(f"데이터 또는 모델 로드 중 오류가 발생했습니다: {e}")
    st.stop()

if not run_simulation:
    st.info("좌측 사이드바에서 조건을 설정한 뒤 시뮬레이션 실행 버튼을 눌러주세요.")
    st.stop()

# ==============================
# 계산
# ==============================
try:
    base_result = predict_churn(df, model, threshold)

    sim_df = apply_simulation_scenario(
        df=df,
        price_relief_pct=price_relief_pct,
        reduce_price_jump_customers_pct=reduce_price_jump_customers_pct,
        reduce_late_risk_customers_pct=reduce_late_risk_customers_pct,
        reduce_complaint_customers_pct=reduce_complaint_customers_pct,
        reduce_quote_requested_customers_pct=reduce_quote_requested_customers_pct,
        reduce_downgrade_customers_pct=reduce_downgrade_customers_pct,
    )

    sim_result = predict_churn(sim_df, model, threshold)

    improvement = base_result["churn_rate"] - sim_result["churn_rate"]
    saved_customers = base_result["churn_count"] - sim_result["churn_count"]
    saved_value = saved_customers * clv

    policy_effect_df = run_single_policy_simulations(
        df=df,
        model=model,
        threshold=threshold,
        price_relief_pct=price_relief_pct,
        reduce_price_jump_customers_pct=reduce_price_jump_customers_pct,
        reduce_late_risk_customers_pct=reduce_late_risk_customers_pct,
        reduce_complaint_customers_pct=reduce_complaint_customers_pct,
        reduce_quote_requested_customers_pct=reduce_quote_requested_customers_pct,
        reduce_downgrade_customers_pct=reduce_downgrade_customers_pct,
    )

except Exception as e:
    st.error(f"시뮬레이션 계산 중 오류가 발생했습니다: {e}")
    with st.expander("디버깅 정보"):
        st.write("원본 데이터 컬럼:", df.columns.tolist())
        st.write("threshold:", threshold)
    st.stop()

# ==============================
# KPI
# ==============================
st.markdown('<div class="section-title">📊 시뮬레이션 결과</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("현재 예상 이탈률", f'{base_result["churn_rate"]:.2f}%')

with col2:
    st.metric("시뮬레이션 후 이탈률", f'{sim_result["churn_rate"]:.2f}%', f"{-improvement:.2f}%p")

with col3:
    avg_prob_delta = sim_result["avg_prob"] - base_result["avg_prob"]
    st.metric("평균 이탈확률", f'{sim_result["avg_prob"]:.2f}%', f"{avg_prob_delta:.2f}%p")

with col4:
    churn_count_delta = sim_result["churn_count"] - base_result["churn_count"]
    st.metric("이탈 고객 수", f'{sim_result["churn_count"]:,}명', f"{churn_count_delta:,}명")

with col5:
    st.metric("예상 방어 매출", f"{saved_value:,.0f}원")

st.divider()

# ==============================
# 그래프 영역
# ==============================
st.markdown('<div class="section-title">💡 기대 효과 분석</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    fig1 = draw_pretty_bar_chart(
        title="평균 이탈확률 비교",
        ylabel="이탈확률 (%)",
        before_value=float(base_result["avg_prob"]),
        after_value=float(sim_result["avg_prob"]),
        before_color="#FF7444",
        after_color="#B7BDF7",
    )
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

with col_right:
    fig2 = draw_pretty_bar_chart(
        title="예상 이탈률 비교",
        ylabel="예상 이탈률 (%)",
        before_value=float(base_result["churn_rate"]),
        after_value=float(sim_result["churn_rate"]),
        before_color="#FF7444",
        after_color="#B7BDF7",
    )
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# # ==============================
# # 정책별 효과 비교
# # ==============================
# st.markdown('<div class="section-title">📌 정책별 효과 비교</div>', unsafe_allow_html=True)
#
# fig3, ax3 = plt.subplots(figsize=(9, 5), facecolor="white")
# ax3.set_facecolor("white")
#
# bars3 = ax3.barh(
#     policy_effect_df["정책"],
#     policy_effect_df["평균 이탈확률 감소폭"],
#     color="#8BB4EA",
#     edgecolor="none",
#     height=0.48
# )
#
# ax3.set_title("정책별 평균 이탈확률 감소폭", fontsize=15, fontweight="bold", color="#0F172A", pad=12)
# ax3.set_xlabel("감소폭 (%p)", fontsize=11, color="#475569")
# ax3.spines["top"].set_visible(False)
# ax3.spines["right"].set_visible(False)
# ax3.spines["left"].set_color("#CBD5E1")
# ax3.spines["bottom"].set_color("#CBD5E1")
# ax3.tick_params(axis="y", labelsize=11, colors="#334155")
# ax3.tick_params(axis="x", labelsize=10, colors="#64748B")
# ax3.xaxis.grid(True, linestyle="--", alpha=0.16)
# ax3.set_axisbelow(True)
#
# for bar, value in zip(bars3, policy_effect_df["평균 이탈확률 감소폭"]):
#     ax3.text(
#         value + 0.04,
#         bar.get_y() + bar.get_height() / 2,
#         f"{value:.2f}",
#         va="center",
#         fontsize=11,
#         color="#1E3A8A",
#         fontweight="bold"
#     )
#
# st.pyplot(fig3, use_container_width=True)
# plt.close(fig3)
#
# st.dataframe(policy_effect_df, use_container_width=True)

# ==============================
# 자동 요약문
# ==============================
st.markdown('<div class="section-title">📝 시뮬레이션 요약</div>', unsafe_allow_html=True)

top_policy = policy_effect_df.sort_values("평균 이탈확률 감소폭", ascending=False).iloc[0]["정책"]

st.success(
    f"현재 시나리오에서는 예상 이탈 고객이 {base_result['churn_count']:,}명에서 "
    f"{sim_result['churn_count']:,}명으로 감소하여 약 {saved_customers:,}명의 고객 이탈을 방어할 수 있습니다. "
    f"예상 방어 매출은 {saved_value:,.0f}원이며, 단일 정책 기준으로는 '{top_policy}'의 효과가 가장 크게 나타났습니다."
)

print("sklearn:", sklearn.__version__)
print("joblib:", joblib.__version__)
print("streamlit:", streamlit.__version__)
print("pandas:", pandas.__version__)

# ==============================
# 변화 확인
# ==============================
with st.expander("시뮬레이션 변화 확인"):
    st.write("원본/시뮬레이션 동일 여부:", df.equals(sim_df))
    st.write("기존 이탈 고객 수:", int(base_result["pred"].sum()))
    st.write("시뮬레이션 이탈 고객 수:", int(sim_result["pred"].sum()))

    raw_check_cols = [
        "premium_change_pct",
        "num_price_increases_last_3y",
        "late_payment_count_12m",
        "complaint_flag",
        "quote_requested_flag",
        "coverage_downgrade_flag",
    ]
    raw_check_cols = [c for c in raw_check_cols if c in df.columns]

    if raw_check_cols:
        raw_compare = pd.DataFrame({
            "원본 평균": df[raw_check_cols].mean(numeric_only=True),
            "시뮬레이션 평균": sim_df[raw_check_cols].mean(numeric_only=True),
        })
        raw_compare["차이"] = raw_compare["시뮬레이션 평균"] - raw_compare["원본 평균"]
        st.markdown("**원본 변수 변화**")
        st.dataframe(raw_compare, use_container_width=True)

        changed_count_df = pd.DataFrame({
            "컬럼": raw_check_cols,
            "변경 행 수": [int((df[c] != sim_df[c]).sum()) for c in raw_check_cols]
        })
        st.markdown("**컬럼별 변경 행 수**")
        st.dataframe(changed_count_df, use_container_width=True)

    base_X_debug = make_model_input(df)
    sim_X_debug = make_model_input(sim_df)

    engineered_check_cols = [
        "premium_increase_shock",
        "premium_jump_flag",
        "payment_risk_score",
        "service_risk_score",
        "engagement_risk_score",
        "premium_x_quote",
        "late_x_quote",
        "downgrade_x_quote",
        "monthly_x_premium_jump",
    ]
    engineered_check_cols = [c for c in engineered_check_cols if c in base_X_debug.columns]

    if engineered_check_cols:
        engineered_compare = pd.DataFrame({
            "원본 평균": base_X_debug[engineered_check_cols].mean(numeric_only=True),
            "시뮬레이션 평균": sim_X_debug[engineered_check_cols].mean(numeric_only=True),
        })
        engineered_compare["차이"] = engineered_compare["시뮬레이션 평균"] - engineered_compare["원본 평균"]
        st.markdown("**파생변수 변화**")
        st.dataframe(engineered_compare, use_container_width=True)

    prob_compare = pd.DataFrame({
        "구분": ["원본", "시뮬레이션"],
        "평균 이탈확률(%)": [base_result["avg_prob"], sim_result["avg_prob"]],
        "예상 이탈률(%)": [base_result["churn_rate"], sim_result["churn_rate"]],
        "이탈 고객 수": [base_result["churn_count"], sim_result["churn_count"]],
    })
    st.markdown("**예측 결과 비교**")
    st.dataframe(prob_compare, use_container_width=True)
