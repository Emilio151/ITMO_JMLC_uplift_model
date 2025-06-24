# Uplift and Qini metrics functions will be placed here 

from sklift.metrics import uplift_at_k, uplift_auc_score, qini_auc_score

def calculate_uplift(y_test, uplift_effect, treatment_test):
    """
    Вычисляет uplift at k, AUUC, AUQC и печатает результаты.
    """
    upliftk = uplift_at_k(
        y_true=y_test,
        uplift=uplift_effect,
        treatment=treatment_test,
        strategy='by_group',
        k=0.3
    )
    upliftk_all = uplift_at_k(
        y_true=y_test,
        uplift=uplift_effect,
        treatment=treatment_test,
        strategy='overall',
    )
    qini_coef = qini_auc_score(
        y_true=y_test,
        uplift=uplift_effect,
        treatment=treatment_test
    )
    uplift_auc = uplift_auc_score(
        y_true=y_test,
        uplift=uplift_effect,
        treatment=treatment_test
    )
    print(f'uplift at top 30% by group: {upliftk:.3f} by overall: {upliftk_all:.3f}\n',
          f'AUUC by group: {uplift_auc:.3f}\n',
          f'AUQC by group: {qini_coef:.3f}\n')
    return upliftk 