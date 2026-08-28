"""Per-class classification metrics, thin wrappers over scikit-learn."""
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def per_class_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> pd.DataFrame:
    """Precision/recall/F1/support per class, as a DataFrame indexed by label."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {"precision": precision, "recall": recall, "f1-score": f1, "support": support}, index=labels
    )


def confusion_matrix_df(y_true: list[str], y_pred: list[str], labels: list[str]) -> pd.DataFrame:
    """Confusion matrix as a DataFrame: rows are true labels, columns are predicted labels."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)
