import pandas as pd

from src.evaluation.classifier_metrics import confusion_matrix_df, per_class_report


def test_per_class_report_has_precision_recall_f1_per_label():
    y_true = ["fox", "fox", "coyote", "coyote", "coyote"]
    y_pred = ["fox", "coyote", "coyote", "coyote", "fox"]

    report = per_class_report(y_true, y_pred, labels=["coyote", "fox"])

    assert set(report.index) == {"coyote", "fox"}
    assert set(report.columns) >= {"precision", "recall", "f1-score", "support"}
    assert report.loc["coyote", "support"] == 3
    assert report.loc["fox", "support"] == 2


def test_confusion_matrix_df_rows_are_true_columns_are_predicted():
    y_true = ["fox", "fox", "coyote"]
    y_pred = ["fox", "coyote", "coyote"]

    matrix = confusion_matrix_df(y_true, y_pred, labels=["coyote", "fox"])

    assert matrix.loc["fox", "fox"] == 1
    assert matrix.loc["fox", "coyote"] == 1
    assert matrix.loc["coyote", "coyote"] == 1
    assert matrix.loc["coyote", "fox"] == 0
