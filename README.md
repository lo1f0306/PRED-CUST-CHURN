# RETURNS
머신러닝/딥러닝을 통한 가입고객 이탈예측 프로젝트

## 📂 프로젝트 구조 (Project Structure)

```text
PRED-CUST-CHURN/
├── data/                    # 데이터 폴더
├── analysis/                # 개별 가설 분석 폴더
├── src/                     # 공통 모듈 (팀원 공유용)
│   └── preprocess.py        # 데이터 로드 및 전처리 클래스
├── pages/                   # Streamlit 웹 어플리케이션
│   ├── churn_predictor.py   # 이탈 예측 화면
│   ├── entry.py             # 진입 화면
│   └── risk_watchlist.py    # 이탈 확률이 높은 고객 리스트 화면
├── model/                   # 학습된 모델 저장 폴더
├── requirements.txt         # 필요 라이브러리 목록
└── app.py                   # 실행 streamlit
