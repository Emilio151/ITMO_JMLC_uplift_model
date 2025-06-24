# ITMO JMLC Uplift Model

## Описание
Проект для моделирования uplift и анализа A/B теста.

## Структура
- `notebooks/` — ноутбуки и скрипты для EDA/экспериментов
- `src/` — основной код проекта (data, features, models, evaluation, utils)
- `uplift_ab_test.csv` — исходные данные

## Быстрый старт
1. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```
2. Запустите ноутбук или скрипт из папки `notebooks/`.

## Пример использования
```python
from src.data.data_loader import load_uplift_data
from src.features.preprocessing import impute_education, drop_unused_columns, encode_categorical
from src.models.uplift_models import make_train_test, apply_smote, prepare_meta_learner_data, train_s_learner, train_t_learner
from src.evaluation.metrics import calculate_uplift
from src.utils.helpers import percent_missing, share_pivot

df = load_uplift_data("uplift_ab_test.csv")
df = impute_education(df)
df = drop_unused_columns(df)
df_encoded = encode_categorical(df)
# и т.д.
```

## Контакты
Автор: [Ваше имя]

# ITMO_JMLC_uplift_model
Проект по Junior ML Contest для ITMO AI Talent Hub

В рамках данного проекта показ процесс создания аплифт модели по продукту "Инвесткопилка". В рамках проекта было реализовано следующее:
1) Сбор данных (представлен в прикрепленном csv файле)
2) Обработка данных (работа с выборосами, несбалансированностью классов)
3) Анализ данных (распределение target переменной, анализ корреляций признаков)
4) Feature engineering
5) Выбор метамодели
6) Выбор ML модели
7) Анализ работы модели 

# Project Structure

- `notebooks/`: Jupyter notebooks for experiments and EDA
- `src/`: Source code modules
  - `data/`: Data loading and cleaning
  - `features/`: Feature engineering and preprocessing
  - `models/`: Model definitions and training
  - `evaluation/`: Metrics and evaluation functions
  - `utils/`: Utility/helper functions
- `uplift_ab_test.csv`: Raw data 
