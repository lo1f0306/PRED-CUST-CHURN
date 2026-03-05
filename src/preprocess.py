import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(type:str = 'train'):
    """
    데이터를 공통으로 로드하는 함수
    :return:
    """
    # 데이터 파일명
    data_file_name = 'insurance_policyholder_churn_synthetic.csv'

    # data 폴더 접근
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # src의 상위인 Project 폴더
    data_path = project_root / "data" / data_file_name

    df = pd.read_csv(data_path)
    # 전처리
    common_preprocess_churn_data(df)

    return df

def common_preprocess_churn_data(df):
    """
    공통적으로 처리할 공통 전처리 함수
    1. 학습에 사용하지 않는 변수 제거
    :param df:
    :return:
    """
    # 1. 학습에 사용하지 않을 변수 제거 (Drop)
    # churn_probability_true는 정답 유출 방지를 위해 반드시 제거
    drop_cols = ['customer_id',             # 사용자 ID
                 'as_of_date',              # 기준일자
                 'churn_type',              # 이탈 원인 분류
                 'churn_probability_true']  # 실제 이탈 확률
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    return df
