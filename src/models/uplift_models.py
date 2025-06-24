# Uplift model training and definition functions will be placed here 

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklift.models import SoloModel, TwoModels
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def make_train_test(df_encoded: pd.DataFrame, df_original: pd.DataFrame, test_size=0.2, random_state=42):
    train, test = train_test_split(df_encoded, test_size=test_size, random_state=random_state, stratify=df_original['treatment'])
    return train, test

def apply_smote(train: pd.DataFrame):
    X_train = train.drop(columns=['target'])
    y_train = train['target']
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote

def prepare_meta_learner_data(X_train_smote, y_train_smote, test):
    X_train = X_train_smote.drop(columns=['treatment'])
    treatment_train = X_train_smote['treatment']
    y_train = y_train_smote
    X_test = test.drop(columns=['treatment', 'target'])
    treatment_test = test['treatment']
    y_test = test['target']
    return X_train, treatment_train, y_train, X_test, treatment_test, y_test

def train_s_learner(model_name, X_train, treatment_train, y_train):
    if model_name == 'catboost':
        estimator = CatBoostClassifier(silent=True)
    elif model_name == 'lgbm':
        estimator = LGBMClassifier(random_state=1, n_jobs=-1)
    elif model_name == 'xgboost':
        estimator = XGBClassifier(random_state=1, n_jobs=-1, eval_metric='logloss')
    else:
        raise ValueError('Unknown model_name')
    model = SoloModel(estimator=estimator)
    model.fit(X_train, treatment_train, y_train)
    return model

def train_t_learner(model_name, X_train, treatment_train, y_train):
    if model_name == 'catboost':
        estimator = CatBoostClassifier(silent=True)
    elif model_name == 'lgbm':
        estimator = LGBMClassifier(random_state=1, n_jobs=-1)
    elif model_name == 'xgboost':
        estimator = XGBClassifier(random_state=1, n_jobs=-1, eval_metric='logloss')
    else:
        raise ValueError('Unknown model_name')
    model = TwoModels(estimator_trmnt=estimator, estimator_ctrl=estimator)
    model.fit(X_train, treatment_train, y_train)
    return model 