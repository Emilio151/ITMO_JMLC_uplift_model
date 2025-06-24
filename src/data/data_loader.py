# Data loading functions will be placed here 

import pandas as pd
import numpy as np

def load_uplift_data(filepath: str = "uplift_ab_test.csv") -> pd.DataFrame:
    """
    Загружает данные для uplift-моделирования из CSV-файла и выполняет базовую обработку пропусков.
    """
    df = pd.read_csv(filepath)
    # Заполнение пропусков в dependents_cnt и children_cnt нулями
    df['dependents_cnt'] = df['dependents_cnt'].fillna(0)
    df['children_cnt'] = df['children_cnt'].fillna(0)
    # Заполнение пропусков в monthly_income_amt средним
    df['monthly_income_amt'] = df['monthly_income_amt'].fillna(np.mean(df['monthly_income_amt']))
    # Заполнение пропусков в marital_status_cd значением 'UNM'
    df['marital_status_cd'] = df['marital_status_cd'].fillna('UNM')
    return df 