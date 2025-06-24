from src.data.data_loader import load_uplift_data
from src.features.preprocessing import impute_education, drop_unused_columns, encode_categorical
from src.models.uplift_models import make_train_test, apply_smote, prepare_meta_learner_data, train_s_learner
from src.evaluation.metrics import calculate_uplift


def main():
    df = load_uplift_data("uplift_ab_test.csv")

    df = impute_education(df)

    df = drop_unused_columns(df)

    df_encoded = encode_categorical(df)

    train, test = make_train_test(df_encoded, df)

    X_train_smote, y_train_smote = apply_smote(train)

    X_train, treatment_train, y_train, X_test, treatment_test, y_test = prepare_meta_learner_data(X_train_smote, y_train_smote, test)

    model = train_s_learner('catboost', X_train, treatment_train, y_train)

    uplift_effect = model.predict(X_test)

    calculate_uplift(y_test, uplift_effect, treatment_test)


if __name__ == "__main__":
    main() 