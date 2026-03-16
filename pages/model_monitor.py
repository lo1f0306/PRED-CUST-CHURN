from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.model_service import load_model_bundle, load_scored_customers_file


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


st.markdown(
    """
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1380px;
}
.report-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}
.report-subtitle {
    color: #475569;
    font-size: 1.04rem;
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 1.75rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.badge {
    display: inline-block;
    padding: 0.42rem 0.85rem;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 700;
}
.badge-done {
    background: #dcfce7;
    color: #15803d;
}
.card {
    background: white;
    border-radius: 22px;
    padding: 1.35rem 1.45rem;
    box-shadow: 0 6px 22px rgba(15, 23, 42, 0.06);
    border: 1px solid #e2e8f0;
}
.kpi-card {
    border-radius: 24px;
    padding: 1.55rem;
    min-height: 190px;
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
}
.kpi-neutral {
    background: #ffffff;
    border: 1px solid #e2e8f0;
}
.kpi-critical {
    background: #fff1f2;
    border: 1px solid #fecdd3;
}
.kpi-high {
    background: #fff7ed;
    border: 1px solid #fed7aa;
}
.kpi-watch {
    background: #fffbeb;
    border: 1px solid #fde68a;
}
.kpi-stable {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
}
.kpi-label {
    font-size: 1.1rem;
    color: #334155;
    margin-bottom: 0.8rem;
    font-weight: 700;
}
.kpi-value {
    font-size: 2.85rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 1rem;
    color: #475569;
    margin-top: 0.75rem;
}
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
}
.info-item {
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1rem 1.1rem;
}
.info-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #0f172a;
}
.info-value {
    font-size: 1rem;
    color: #334155;
    margin-top: 0.45rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def fmt_int(value: int) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def safe_rate(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or len(df) == 0:
        return 0.0
    return float(df[column].mean()) * 100


def safe_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or len(df) == 0:
        return 0.0
    return float(df[column].mean())


def get_group_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    total = len(df)
    summary = {}
    for tier in RISK_ORDER:
        count = int((df["risk_tier"] == tier).sum())
        summary[tier] = {
            "count": count,
            "pct": (count / total * 100) if total else 0.0,
        }
    return summary


def build_group_insight(df: pd.DataFrame, tier: str) -> str:
    group_df = df[df["risk_tier"] == tier]
    premium_change = safe_mean(group_df, "premium_change_pct") * 100
    complaint_rate = safe_rate(group_df, "complaint_flag")
    quote_rate = safe_rate(group_df, "quote_requested_flag")
    late_payment = safe_mean(group_df, "late_payment_count_12m")

    if tier == "critical":
        return (
            f"즉시 대응 고객군은 평균 보험료 변동률 {premium_change:.1f}%, "
            f"민원 비율 {complaint_rate:.1f}%, 타사 견적 요청 비율 {quote_rate:.1f}%로 "
            "강한 이탈 신호가 겹쳐 나타납니다."
        )
    if tier == "high":
        return (
            f"고위험 고객군은 평균 보험료 변동률 {premium_change:.1f}%, "
            f"최근 연체 횟수 {late_payment:.2f}회 수준으로 누적 불만이나 가격 민감 신호가 큽니다."
        )
    if tier == "watch":
        return (
            f"관찰 필요 고객군은 타사 견적 요청 비율 {quote_rate:.1f}%, "
            f"최근 연체 횟수 {late_payment:.2f}회로 모니터링이 필요한 초기 위험 신호가 보입니다."
        )
    return (
        f"안정 고객군은 평균 보험료 변동률 {premium_change:.1f}%, "
        f"민원 비율 {complaint_rate:.1f}%로 상대적으로 유지 가능성이 높은 편입니다."
    )


try:
    scored_df = load_scored_customers_file()
    _, threshold = load_model_bundle()

    total_cnt = len(scored_df)
    summary = get_group_summary(scored_df)

    left, right = st.columns([6, 1.2])
    with left:
        st.markdown('<div class="report-title">자동 분석 리포트</div>', unsafe_allow_html=True)
        st.markdown(
            (
                f'<div class="report-subtitle">'
                f'공통 스코어 파일 기준 위험 등급 현황 · 업데이트: {datetime.now().strftime("%Y년 %m월 %d일")} '
                f'· model threshold: {threshold:.4f} · risk tiers: 0.25 / 0.40 / 0.60'
                f"</div>"
            ),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
        st.markdown('<span class="badge badge-done">분석 완료</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card kpi-neutral">
                <div class="kpi-label">전체 고객 수</div>
                <div class="kpi-value">{fmt_int(total_cnt)}</div>
                <div class="kpi-sub">100% 전체 고객 기준</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for col, tier in zip([c2, c3, c4, c5], RISK_ORDER):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card {RISK_CARD_CLASSES[tier]}">
                    <div class="kpi-label">{RISK_LABELS[tier]} 고객</div>
                    <div class="kpi-value">{fmt_int(summary[tier]["count"])}</div>
                    <div class="kpi-sub">{fmt_pct(summary[tier]["pct"])} 고객 비중</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">위험군 분포 분석</div>', unsafe_allow_html=True)
    dist_left, dist_right = st.columns([1.05, 1.2])

    with dist_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=130)
        sizes = [summary[tier]["count"] for tier in RISK_ORDER]
        labels = [f'{RISK_LABELS[tier]} {summary[tier]["pct"]:.1f}%' for tier in RISK_ORDER]
        colors = [RISK_COLORS[tier] for tier in RISK_ORDER]
        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            startangle=0,
            wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
            textprops={"fontsize": 11},
        )
        ax.axis("equal")
        ax.set_title("현재 위험군 분포", fontsize=15, pad=20)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with dist_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 위험군별 상세 정보")
        for tier in RISK_ORDER:
            pct = summary[tier]["pct"]
            count = summary[tier]["count"]
            st.markdown(
                f"""
                <div style="border:1px solid #e5e7eb; border-radius:16px; padding:1rem 1.1rem; margin-bottom:1rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-size:1.2rem; font-weight:800; color:#0f172a;">{RISK_LABELS[tier]} 고객</div>
                        <div style="font-size:1rem; color:#0f172a; border:1px solid #e5e7eb; padding:0.25rem 0.6rem; border-radius:999px;">{fmt_int(count)}명</div>
                    </div>
                    <div style="width:100%; height:12px; background:#e5e7eb; border-radius:999px; overflow:hidden; margin-top:1rem; margin-bottom:1rem;">
                        <div style="width:{pct}%; height:100%; background:{RISK_COLORS[tier]}; border-radius:999px;"></div>
                    </div>
                    <div style="text-align:right; font-size:1rem; color:#334155;">{pct:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">핵심 인사이트</div>', unsafe_allow_html=True)
    insight_cols = st.columns(2)
    with insight_cols[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for tier in ["critical", "high"]:
            st.markdown(f"### {RISK_LABELS[tier]}")
            st.write(build_group_insight(scored_df, tier))
        st.markdown('</div>', unsafe_allow_html=True)
    with insight_cols[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for tier in ["watch", "stable"]:
            st.markdown(f"### {RISK_LABELS[tier]}")
            st.write(build_group_insight(scored_df, tier))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">공통 기준 안내</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-title">공통 데이터 소스</div>
                    <div class="info-value">data/scored/scored_customers_new_model.parquet</div>
                </div>
                <div class="info-item">
                    <div class="info-title">이탈 threshold</div>
                    <div class="info-value">0.1669 이상이면 predicted_churn = 1</div>
                </div>
                <div class="info-item">
                    <div class="info-title">위험 등급 구간</div>
                    <div class="info-value">즉시 대응 0.60+, 고위험 0.40+, 관찰 필요 0.25+, 안정 0.25 미만</div>
                </div>
                <div class="info-item">
                    <div class="info-title">페이지 일치 범위</div>
                    <div class="info-value">홈, 위험리스트, 단건예측, 모델 정보 2 모두 동일 기준 사용</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">분석 결과 샘플</div>', unsafe_allow_html=True)
    preview_cols = [
        col
        for col in [
            "customer_id",
            "churn_probability",
            "risk_tier_ko",
            "premium_change_pct",
            "complaint_flag",
            "late_payment_count_12m",
            "quote_requested_flag",
            "coverage_downgrade_flag",
            "prediction_reason",
        ]
        if col in scored_df.columns
    ]
    show_df = scored_df[preview_cols].copy().sort_values("churn_probability", ascending=False)
    if "churn_probability" in show_df.columns:
        show_df["churn_probability"] = (show_df["churn_probability"] * 100).round(1)

    st.dataframe(show_df.head(30), use_container_width=True, hide_index=True)

except Exception as exc:
    st.error(f"자동 분석 리포트 생성 중 오류가 발생했습니다: {exc}")
