import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from src.model_service import (
    evaluate_saved_model,
    load_model_bundle,
    load_scored_customers_file,
)

# 1. 게이지 차트 함수 (크기 최적화)
def draw_gauge_chart(value, title, color="#2563eb"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "#f1f5f9"},
                {'range': [50, 80], 'color': "#e2e8f0"},
                {'range': [80, 100], 'color': "#cbd5e1"}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# 데이터 로드
metrics = evaluate_saved_model()
scored_df = load_scored_customers_file()
_, threshold = load_model_bundle()

st.markdown('<div class="main-title">📊 모델 성능 대시보드</div>', unsafe_allow_html=True)


# --- 섹션 1: 핵심 성과 게이지 (Recall 강조) ---
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">핵심 성능 지표 (Core Metrics)</div>', unsafe_allow_html=True)
g_col1, g_col2, g_col3 = st.columns(3)

with g_col1:
    st.plotly_chart(draw_gauge_chart(metrics["recall"], "Recall (재현율)"), use_container_width=True)
    st.caption("실제 이탈자 중 모델이 찾아낸 비율 (가장 중요)")
with g_col2:
    st.plotly_chart(draw_gauge_chart(metrics["roc_auc"], "ROC-AUC (분별력)", "#10b981"), use_container_width=True)
    st.caption("이탈자와 유지자를 구분하는 종합 성능")
with g_col3:
    st.plotly_chart(draw_gauge_chart(metrics["f1"], "F1-Score (균형)", "#f59e0b"), use_container_width=True)
    st.caption("정밀도와 재현율의 조화 평균")
st.markdown('</div>', unsafe_allow_html=True)

# --- 섹션 2: 분포 및 분류 결과 시각화 ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">예측 등급 분포</div>', unsafe_allow_html=True)
    tier_counts = scored_df["risk_tier_ko"].value_counts().reset_index()
    tier_counts.columns = ["위험 등급", "고객 수"]
    fig_pie = px.pie(tier_counts, names="위험 등급", values="고객 수", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">분류 결과 (Confusion Matrix)</div>', unsafe_allow_html=True)
    cm = metrics["confusion_matrix"]
    # 히트맵으로 시각화
    fig_cm = px.imshow(cm,
                       labels=dict(x="예측값", y="실제값", color="인원 수"),
                       x=['유지 예측', '이탈 예측'],
                       y=['실제 유지', '실제 이탈'],
                       text_auto=True, color_continuous_scale='Blues')
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 섹션 3: 임계값 및 변수 중요도 (이미지 호출) ---
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">모델 판단 근거 분석</div>', unsafe_allow_html=True)
img_col1, img_col2 = st.columns(2)

# 기존에 생성된 분석 그래프 이미지 표시
from src.model_service import get_threshold_plot_path, get_corr_plot_path
t_path = get_threshold_plot_path()
c_path = get_corr_plot_path()

with img_col1:
    if t_path.exists():
        st.image(str(t_path), caption="Threshold 최적화 분석 (Precision-Recall Curve)")
with img_col2:
    if c_path.exists():
        st.image(str(c_path), caption="주요 피처와 타겟 간 상관관계")
st.markdown('</div>', unsafe_allow_html=True)