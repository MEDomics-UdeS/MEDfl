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
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = (tp) / (tp + fn)
    sp = (tn) / (tn + fp)
    ppv = (tp) / (tp + fp)
    npv = (tn) / (tn + fn)
    f1 = 2 * (sen * ppv) / (sen + ppv)
    fpr = (fp) / (fp + tn)
    tpr = (tp) / (tp + fn)
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
