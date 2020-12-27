"""
файл с утилитами
"""

import os
from time import perf_counter
import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
    auc,
    cohen_kappa_score,
    classification_report,
    # confusion_matrix,
)
from sklearn.metrics import recall_score, precision_score
import shap
import matplotlib.pyplot as plt
from functools import wraps

def get_metrics(model, x_val, y_val):
    """
    Вычисление простых метрик
    """

    y_pred = model.predict(x_val)
    
    mse = np.mean((y_val - y_pred)**2)
    mask = y_val > 0
    mape =  (np.fabs(y_val - y_pred) / y_val)[mask].mean()

    return y_pred, mse, mape


def shap_analysis(booster, data, name_f):
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(21, 12))
    shap_values = shap.TreeExplainer(booster).shap_values(data)
    fig = plt.figure(figsize=(40, 40))
    shap.summary_plot(shap_values, data, show=False, max_display=len(data.columns))
    fig.savefig(name_f, bbox_inches="tight")

