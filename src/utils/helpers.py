# Helper functions (e.g., percent_missing, share_pivot) will be placed here 

import numpy as np
import pandas as pd

def percent_missing(df: pd.DataFrame) -> pd.Series:
    """
    Возвращает Series с процентом пропусков по каждому столбцу (>0).
    """
    percent_nan = 100 * df.isnull().sum() / len(df)
    percent_nan = percent_nan[percent_nan > 0].sort_values()
    return percent_nan

def share_pivot(df: pd.DataFrame, index_cols: list) -> pd.DataFrame:
    """
    Возвращает DataFrame с количеством и долей уникальных комбинаций index_cols.
    """
    pivot = pd.DataFrame(df[index_cols].value_counts()).sort_index()
    pivot.columns = ['#']
    pivot['share'] = np.round(pivot['#'] / len(df), 2)
    return pivot 