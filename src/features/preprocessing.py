# Feature engineering and preprocessing functions will be placed here 

import pandas as pd
import numpy as np

def impute_education(df: pd.DataFrame) -> pd.DataFrame:
    """
    Импутация пропусков в education_level_cd на основе ближайшего по доходу класса.
    """
    income_by_edu = df.dropna(subset=['education_level_cd']) \
        .groupby('education_level_cd')['monthly_income_amt'] \
        .mean()
    def _impute(row):
        if pd.isnull(row['education_level_cd']):
            diffs = abs(income_by_edu - row['monthly_income_amt'])
            return diffs.idxmin()
        else:
            return row['education_level_cd']
    df['education_level_cd'] = df.apply(_impute, axis=1)
    return df

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет неиспользуемые столбцы, как в ноутбуке.
    """
    return df.drop(['pensioner_flg','transactions_max_1m','transactions_avg_1m'], axis=1)

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot кодирование education_level_cd и marital_status_cd.
    """
    return pd.get_dummies(df, columns=['education_level_cd', 'marital_status_cd'], drop_first=True) 