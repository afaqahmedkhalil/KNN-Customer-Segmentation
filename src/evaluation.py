import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance


def evaluate_model(model, X_test, y_test, labels=['Low', 'Medium', 'High']):
    """Return accuracy, confusion matrix, report, ROC-AUC."""
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    # Binarize labels for multi-class AUC
    y_test_bin = label_binarize(y_test, classes=labels)
    y_pred_bin = label_binarize(preds, classes=labels)
    roc_auc = roc_auc_score(y_test_bin, y_pred_bin, average='macro')

    return acc, cm, report, roc_auc


def get_feature_importance(model, X_test, y_test, feature_names):
    """Permutation importance."""
    result = permutation_importance(model, X_test, y_test, n_repeats=10)

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean
    }).sort_values(by='Importance', ascending=False)

    return importance_df
