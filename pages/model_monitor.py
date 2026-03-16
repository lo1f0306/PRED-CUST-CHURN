from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# 로컬 모듈 호출 (환경에 맞게 유지)
try:
    from src.model_service import load_model_bundle, load_scored_customers_file
except ImportError:
    st.error("모듈을 찾을 수 없습니다. 경로를 확인해주세요.")

# --- 설정 및 상수 ---
st.set_page_config(
    page_title="자동 분석 리포트",
    page_icon="📊",
    layout="wide",
)

RISK_ORDER = ["critical", "high", "watch", "stable"]
RISK_LABELS = {
    "critical": "즉시 대응",
    "high": "고위험",
    "watch": "관찰 필요",
    "stable": "안정",
}
RISK_COLORS = {
    "critical": "#ef4444",
    "high": "#f97316",
    "watch": "#f59e0b",
    "stable": "#10b981",
}
RISK_CARD_CLASSES = {
    "critical": "kpi-critical",
    "high": "kpi-high",
    "watch": "kpi-watch",
    "stable": "kpi-stable",
}

# --- 스타일링 (CSS) ---
st.markdown(
    """
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', sans-serif; }
    .block-container { padding-top: 1.5rem; max-width: 1380px; }
    .report-title { font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 0.2rem; }
    .report-subtitle { color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; }
    .section-title { font-size: 1.6rem; font-weight: 800; color: #1e293b; margin: 2.5rem 0 1.2rem 0; padding-left: 0.5rem; border-left: 5px solid #3b82f6; }
    .card { background: white; border-radius: 20px; padding: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #f1f5f9; height: 100%; }
    .kpi-card { border-radius: 20px; padding: 1.5rem; min-height: 160px; display: flex; flex-direction: column; justify-content: center; transition: transform 0.2s; }
    .kpi-card:hover { transform: translateY(-5px); }
    .kpi-neutral { background: #ffffff; border: 1px solid #e2e8f0; }
    .kpi-critical { background: #fff1f2; border: 1px solid #fecdd3; }
    .kpi-high { background: #fff7ed; border: 1px solid #fed7aa; }
    .kpi-watch { background: #fffbeb; border: 1px solid #fde68a; }
    .kpi-stable { background: #f0fdf4; border: 1px solid #bbf7d0; }
    .kpi-label { font-size: 1rem; color: #64748b; font-weight: 600; margin-bottom: 0.5rem; }
    .kpi-value { font-size: 2.4rem; font-weight: 800; color: #0f172a; }
    .kpi-sub { font-size: 0.9rem; color: #94a3b8; margin-top: 0.5rem; }
    .insight-pill { display: inline-block; padding: 0.3rem 1rem; border-radius: 8px; background: #f1f5f9; font-weight: 700; color: #334155; margin-bottom: 0.8rem; }
    .insight-body { font-size: 1.05rem; line-height: 1.7; color: #334155; }
    .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }
    .info-item { background: #f8fafc; border-radius: 12px; padding: 1rem; border: 1px solid #f1f5f9; }
</style>
""",
    unsafe_allow_html=True,
)


# --- 헬퍼 함수 ---
def fmt_int(value: int) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def get_group_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0: return {t: {"count": 0, "pct": 0} for t in RISK_ORDER}
    return {tier: {"count": (df["risk_tier"] == tier).sum(), "pct": ((df["risk_tier"] == tier).sum() / total * 100)} for
            tier in RISK_ORDER}


def build_group_insight(df: pd.DataFrame, tier: str) -> str:
    group_df = df[df["risk_tier"] == tier]
    if group_df.empty: return "해당 그룹의 데이터가 없습니다."

    prem_change = group_df["premium_change_pct"].mean() * 100 if "premium_change_pct" in group_df.columns else 0
    comp_rate = group_df["complaint_flag"].mean() * 100 if "complaint_flag" in group_df.columns else 0
    quote_rate = group_df["quote_requested_flag"].mean() * 100 if "quote_requested_flag" in group_df.columns else 0

    insights = {
        "critical": f"**즉시 대응** 고객군은 보험료 변동({prem_change:.1f}%)과 타사 견적 요청({quote_rate:.1f}%)이 결합된 이탈 직전 단계입니다.",
        "high": f"**고위험** 고객군은 민원 발생률({comp_rate:.1f}%)이 상대적으로 높으며 가격 민감도가 상승 중입니다.",
        "watch": f"**관찰 필요** 고객군은 이탈 징후가 낮으나 견적 요청({quote_rate:.1f}%) 등의 간접 신호가 관측됩니다.",
        "stable": f"**안정** 고객군은 변동성이 낮으며 서비스 만족도가 유지되고 있는 층입니다."
    }
    return insights.get(tier, "")


# --- 메인 로직 ---
try:
    scored_df = load_scored_customers_file()
    _, threshold = load_model_bundle()

    total_cnt = len(scored_df)
    summary = get_group_summary(scored_df)

    # 헤더 섹션
    header_l, header_r = st.columns([5, 1])
    with header_l:
        st.markdown('<div class="report-title">데이터 분석 리포트</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="report-subtitle">기준일: {datetime.now().strftime("%Y-%m-%d")} | Threshold: {threshold:.4f}</div>',
            unsafe_allow_html=True)
    with header_r:
        st.markdown(
            '<div style="text-align:right; margin-top:1rem;"><span style="background:#dcfce7; color:#15803d; padding:0.5rem 1rem; border-radius:10px; font-weight:700;">LIVE 분석 완료</span></div>',
            unsafe_allow_html=True)

    # KPI 카드 섹션
    kpi_cols = st.columns(5)
    kpi_cols[0].markdown(
        f'<div class="kpi-card kpi-neutral"><div class="kpi-label">전체 대상</div><div class="kpi-value">{fmt_int(total_cnt)}</div><div class="kpi-sub">모니터링 고객</div></div>',
        unsafe_allow_html=True)
    for i, tier in enumerate(RISK_ORDER):
        kpi_cols[i + 1].markdown(
            f'<div class="kpi-card {RISK_CARD_CLASSES[tier]}">'
            f'<div class="kpi-label">{RISK_LABELS[tier]}</div>'
            f'<div class="kpi-value">{fmt_int(summary[tier]["count"])}</div>'
            f'<div class="kpi-sub">{fmt_pct(summary[tier]["pct"])} 비중</div></div>',
            unsafe_allow_html=True
        )

   
    # 인사이트 섹션 (통합)
    st.markdown('<div class="section-title">핵심 그룹별 인사이트</div>', unsafe_allow_html=True)
    ins_l, ins_r = st.columns(2)

    with ins_l:
        for tier in ["critical", "high"]:
            with st.container():
                st.markdown(
                    f'<div class="card"><div class="insight-pill" style="background:{RISK_COLORS[tier]}22; color:{RISK_COLORS[tier]}">{RISK_LABELS[tier]}</div>'
                    f'<div class="insight-body">{build_group_insight(scored_df, tier)}</div></div>',
                    unsafe_allow_html=True)
                st.write("")  # 간격

    with ins_r:
        for tier in ["watch", "stable"]:
            with st.container():
                st.markdown(
                    f'<div class="card"><div class="insight-pill" style="background:{RISK_COLORS[tier]}22; color:{RISK_COLORS[tier]}">{RISK_LABELS[tier]}</div>'
                    f'<div class="insight-body">{build_group_insight(scored_df, tier)}</div></div>',
                    unsafe_allow_html=True)
                st.write("")

    # 데이터 미리보기 섹션
    st.markdown('<div class="section-title">분석 결과 샘플 (상위 30건)</div>', unsafe_allow_html=True)
    preview_cols = ["customer_id", "churn_probability", "risk_tier_ko", "premium_change_pct", "prediction_reason"]
    # 존재하는 컬럼만 필터링
    valid_cols = [c for c in preview_cols if c in scored_df.columns]
    st.dataframe(
        scored_df[valid_cols].sort_values("churn_probability", ascending=False).head(30).style.format(
            {"churn_probability": "{:.1%}"}),
        use_container_width=True
    )

except Exception as e:
    st.error(f"리포트 생성 중 오류가 발생했습니다: {e}")