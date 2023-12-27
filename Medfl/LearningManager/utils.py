#!/usr/bin/env python3

import pkg_resources
import torch
import yaml
from sklearn.metrics import *
from yaml.loader import SafeLoader

from scripts.base import *

yaml_path = pkg_resources.resource_filename(__name__, "params.yaml")
with open(yaml_path) as g:
    params = yaml.load(g, Loader=SafeLoader)


def custom_classification_report(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Accuracy
    denominator_acc = tp + tn + fp + fn
    acc = (tp + tn) / denominator_acc if denominator_acc != 0 else 0.0

    # Sensitivity/Recall
    denominator_sen = tp + fn
    sen = tp / denominator_sen if denominator_sen != 0 else 0.0

    # Specificity
    denominator_sp = tn + fp
    sp = tn / denominator_sp if denominator_sp != 0 else 0.0

    # PPV/Precision
    denominator_ppv = tp + fp
    ppv = tp / denominator_ppv if denominator_ppv != 0 else 0.0

#     NPV
    denominator_npv = tn + fn
    npv = tn / denominator_npv if denominator_npv != 0 else 0.0

    # F1 Score
    denominator_f1 = sen + ppv
    f1 = 2 * (sen * ppv) / denominator_f1 if denominator_f1 != 0 else 0.0

    # False Positive Rate
    denominator_fpr = fp + tn
    fpr = fp / denominator_fpr if denominator_fpr != 0 else 0.0

    # True Positive Rate
    denominator_tpr = tp + fn
    tpr = tp / denominator_tpr if denominator_tpr != 0 else 0.0

    return {
        "confusion matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "Accuracy": round(acc, 3),
        "Sensitivity/Recall": round(sen, 3),
        "Specificity": round(sp, 3),
        "PPV/Precision": round(ppv, 3),
        "NPV": round(npv, 3),
        "F1-score": round(f1, 3),
        "False positive rate": round(fpr, 3),
        "True positive rate": round(tpr, 3),
    }


def test(model, test_loader, device=torch.device("cpu")):
    model.eval()
    with torch.no_grad():
        X_test, y_test = test_loader.dataset[:][0], test_loader.dataset[:][1]
        y_hat = torch.squeeze(model(X_test), 1).round()

    return custom_classification_report(y_test, y_hat)


column_map = {"object": "VARCHAR(255)", "int64": "INT", "float64": "FLOAT"}


def empty_db():
    my_eng.execute(text(f"DELETE FROM  {'DataSets'}"))
    my_eng.execute(text(f"DELETE FROM {'Nodes'}"))
    my_eng.execute(text(f"DELETE FROM {'FedDatasets'}"))
    my_eng.execute(text(f"DELETE FROM {'Networks'}"))
    my_eng.execute(text(f"DELETE FROM {'FLsetup'}"))

    my_eng.execute(text(f"DELETE FROM {'FLpipeline'}"))
    my_eng.execute(text(f"ALTER TABLE {'DataSets'} AUTO_INCREMENT = 1"))
    my_eng.execute(text(f"ALTER TABLE {'Nodes'} AUTO_INCREMENT = 1"))
    my_eng.execute(text(f"ALTER TABLE {'Networks'} AUTO_INCREMENT = 1"))
    my_eng.execute(text(f"ALTER TABLE {'FedDatasets'} AUTO_INCREMENT = 1"))
    my_eng.execute(text(f"ALTER TABLE {'FLsetup'} AUTO_INCREMENT = 1"))
    my_eng.execute(text(f"ALTER TABLE {'FLpipeline'} AUTO_INCREMENT = 1"))
    my_eng.execute(text(f"DELETE FROM {'testresults'}"))
