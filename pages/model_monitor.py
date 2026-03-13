import pandas as pd
import streamlit as st

from src.model_service import (
    evaluate_saved_model,
    get_corr_plot_path,
    get_threshold_plot_path,
    load_model_bundle,
    load_scored_customers_file,
)


st.set_page_config(page_title="이탈 예측 모델 모니터링", page_icon="📈", layout="wide")


@st.cache_data
def load_metrics():
    return evaluate_saved_model()


@st.cache_data
def load_scored_data():
    return load_scored_customers_file()


metrics = load_metrics()
scored_df = load_scored_data()
_, threshold = load_model_bundle()
tn, fp = metrics["confusion_matrix"][0]
fn, tp = metrics["confusion_matrix"][1]
predicted_churn_precision = tp / (tp + fp) if (tp + fp) else 0.0
captured_churn_recall = tp / (tp + fn) if (tp + fn) else 0.0

st.title("이탈 예측 모델 모니터링")
st.caption("저장된 모델의 성능, 분류 기준값, 현재 운영 상태를 확인합니다.")

metric_cols = st.columns(6)
metric_cols[0].metric("정확도", f"{metrics['accuracy']:.3f}")
metric_cols[1].metric("정밀도", f"{metrics['precision']:.3f}")
metric_cols[2].metric("재현율", f"{metrics['recall']:.3f}")
metric_cols[3].metric("F1 점수", f"{metrics['f1']:.3f}")
metric_cols[4].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
metric_cols[5].metric("기준값", f"{metrics['threshold']:.3f}")

left, right = st.columns([1.1, 1])

with left:
    st.subheader("평가 요약")
    summary_df = pd.DataFrame(
        [
            ("테스트 데이터 수", metrics["test_size"]),
            ("이탈로 예측한 고객 수", metrics["predicted_positive"]),
            ("실제 이탈 고객 수", metrics["actual_positive"]),
            ("PR AUC", round(metrics["pr_auc"], 4)),
            ("재현율 우선 기준값", round(threshold, 4)),
        ],
        columns=["지표", "값"],
    )
    st.dataframe(summary_df, width="stretch", hide_index=True)

    st.caption(
        "재현율 우선 기준값은 고객의 이탈 확률이 이 값 이상일 때 이탈 예상으로 분류하는 컷오프입니다. "
        "현재는 이탈 고객을 놓치지 않는 쪽에 맞춰 낮게 설정되어 있습니다."
    )

    st.subheader("혼동행렬")
    cm = metrics["confusion_matrix"]
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["실제 유지", "실제 이탈"],
            columns=["예측 유지", "예측 이탈"],
        ),
        width="stretch",
    )

with right:
    st.subheader("현재 운영 해석")
    st.markdown(
        """
        - 저장된 모델은 이탈 고객을 놓치지 않는 방향으로 맞춰져 있습니다.
        - 오진이 일부 늘어나더라도 쿠폰/프로모션 대상 누락을 줄이는 것이 우선입니다.
        - 재현율을 핵심 지표로 보고, 정확도는 보조적으로 함께 확인합니다.
        """
    )

    predicted_cnt = int(scored_df["predicted_churn"].sum())
    total_cnt = int(len(scored_df))
    st.info(f"현재 전체 고객 {total_cnt:,}명 중 {predicted_cnt:,}명이 이탈 예상 고객으로 분류됩니다.")

    st.subheader("실제 적중 현황")
    hit_df = pd.DataFrame(
        [
            ("이탈 예측 고객 중 실제 이탈", f"{tp:,} / {tp + fp:,}명"),
            ("이탈 예측 적중률", f"{predicted_churn_precision * 100:.1f}%"),
            ("실제 이탈 고객 중 잡아낸 수", f"{tp:,} / {tp + fn:,}명"),
            ("실제 이탈 포착률", f"{captured_churn_recall * 100:.1f}%"),
        ],
        columns=["항목", "값"],
    )
    st.dataframe(hit_df, width="stretch", hide_index=True)

st.subheader("개선 가설")
st.markdown(
    """
    1. 캠페인 반응 변수 추가: 쿠폰 사용 여부, 리텐션 상담 이력, 이전 프로모션 반응 여부
    2. 보험료 충격 변수 확장: 상품별 인상폭, 누적 인상 횟수, 갱신 직전 인상 여부
    3. 행동 변화 변수 추가: 최근 연체 증가 폭, 문의 증가, 민원 추세 변화
    4. 운영형 지표 추가: 상위 5%, 10%, 20% 고객군 기준 정밀도와 리프트 비교
    5. 예산 기준 운영: 고정 기준값 대신 쿠폰 수량 기준으로 타깃 수를 조정
    """
)

plot_cols = st.columns(2)
threshold_plot = get_threshold_plot_path()
corr_plot = get_corr_plot_path()

if threshold_plot.exists():
    plot_cols[0].image(str(threshold_plot), caption="기준값 분석 그래프", width="stretch")
if corr_plot.exists():
    plot_cols[1].image(str(corr_plot), caption="타깃 상관 상위 특성", width="stretch")

st.subheader("상위 이탈 예상 고객 예시")
top_df = scored_df[
    [
        "customer_id",
        "region_name",
        "policy_type",
        "current_premium",
        "churn_probability",
        "risk_tier_ko",
        "coupon_priority",
        "prediction_reason",
    ]
].head(20).copy()
top_df["churn_probability"] = (top_df["churn_probability"] * 100).round(1)
top_df = top_df.rename(
    columns={
        "customer_id": "고객 ID",
        "region_name": "지역",
        "policy_type": "상품 유형",
        "current_premium": "현재 보험료",
        "churn_probability": "이탈 예상 확률(%)",
        "risk_tier_ko": "위험 등급",
        "coupon_priority": "우선순위",
        "prediction_reason": "이탈 예상 이유",
    }
)
st.dataframe(top_df, width="stretch", hide_index=True)
