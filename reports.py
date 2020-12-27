import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)

import matplotlib.pyplot as plt
import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

from plotly.subplots import make_subplots


pd.set_option("display.max_columns", None)

import itertools
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("seaborn-colorblind")
plt.rc("font", size=14)

from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    precision_recall_curve,
    auc,
    cohen_kappa_score,
    matthews_corrcoef,
    brier_score_loss,
    mean_squared_error,
    classification_report,
    confusion_matrix,
)

# ---------------------------------- End of import ----------------------------------


logger = logging.getLogger(__file__)


def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred / G_true


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def print_report(
    y_test,
    y_pred,
    is_multiclass: bool = False,
    thresh: float = 0.1,
    classes: list = ["Non-paid", "Paid"],
):

    if is_multiclass:
        ind = np.array([np.argmax(x) for x in y_pred])
        print("Accuracy is:", accuracy_score(y_test, ind))
    else:
        ind = np.array([1 if x >= thresh else 0 for x in y_pred])

    logger.info(f"Sample percent for sending to event: {len(ind[ind != 0])/len(ind)}")
    logger.info(f"Cohen's kappa score is: {cohen_kappa_score(y_test, ind)}")
    report = classification_report(y_test, ind, target_names=classes)
    logger.info(report)

    cnf_matrix = confusion_matrix(y_test, ind)

    plt.figure()
    plot_confusion_matrix(
        cnf_matrix, classes=classes, normalize=False, title="Not normalized confusion matrix"
    )

    plt.figure()
    plot_confusion_matrix(
        cnf_matrix, classes=classes, normalize=True, title="Normalized confusion matrix"
    )


def plot_roc_curve(y_test, y_pred, yaxis="y", xaxis="x"):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    # ax.set_title("Receiver Operating Characteristic")
    # ax.plot(false_positive_rate, true_positive_rate, "b", label=f"AUC = {roc_auc:.2f}")

    trace1 = go.Scatter(x=false_positive_rate, y=true_positive_rate, yaxis=yaxis, xaxis=xaxis)

    trace2 = go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"))
    # ax.legend(loc="lower right")

    # ax.set_xlim([-0.1, 1.2])
    # ax.set_ylim([-0.1, 1.2])
    # ax.grid()
    # ax.set_ylabel("True Positive Rate")
    # ax.set_xlabel("False Positive Rate")
    return roc_auc, [trace1, trace2]


def plot_pr_curve(y_test, y_pred, yaxis="y", xaxis="x"):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    proportion = sum(y_test == 1) / sum(y_test == 0)

    # if ax is None:
    #     ax = plt.gca()

    trace1 = go.Scatter(x=recall, y=precision, yaxis=yaxis, xaxis=xaxis)

    trace2 = go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"))

    # ax.set_title("Precision-Recall Curve")
    # ax.plot(recall, precision, "b", label=f"PR AUC = {pr_auc:.2f}")
    # ax.legend(loc="upper right")
    # ax.plot([0, 1], [proportion, proportion], "r--")
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])
    # ax.grid()
    # ax.set_ylabel("Precision")
    # ax.set_xlabel("Recall")
    return pr_auc, [trace1, trace2]


def plot_cohen_kappa(y_test, y_pred, ax=None) -> (float, float):
    thresholds = np.linspace(0, 1, 100)
    kappa = []
    for thr in thresholds:
        ind = np.array([1 if x >= thr else 0 for x in y_pred])
        kappa.append(cohen_kappa_score(y_test, ind))

    kappa = np.array(kappa)
    max_k = np.max(kappa)
    max_thr = thresholds[np.argmax(kappa)]

    if ax is None:
        ax = plt.gca()

    ax.set_title("Cohen's kappa score curve")
    ax.plot(thresholds, kappa, "b", label=f"max kappa is {max_k:.2f}")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid()
    ax.set_ylabel("kappa")
    ax.set_xlabel("threshold")
    return (float("{0:.3f}".format(max_k)), float("{0:.2f}".format(max_thr)))


def plot_matthews_corrcoef(y_test, y_pred, ax=None) -> (float, float):
    thresholds = np.linspace(0, 1, 100)
    mcc = []
    for thr in thresholds:
        ind = np.array([1 if x >= thr else 0 for x in y_pred])
        mcc.append(matthews_corrcoef(y_test, ind))

    mcc = np.array(mcc)
    max_mcc = np.max(mcc)
    max_thr = thresholds[np.argmax(mcc)]

    if ax is None:
        ax = plt.gca()

    ax.set_title("Matthews correlation coefficient curve")
    ax.plot(thresholds, mcc, "b", label=f"max MCC is {max_mcc:.2f}")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid()
    ax.set_ylabel("kappa")
    ax.set_xlabel("threshold")
    return (float("{0:.3f}".format(max_mcc)), float("{0:.2f}".format(max_thr)))


def plot_brier_scor(y_test, y_pred, ax=None) -> (float, float):
    thresholds = np.linspace(0, 1, 100)
    brier = []
    for thr in thresholds:
        ind = np.array([1 if x >= thr else 0 for x in y_pred])
        brier.append(brier_score_loss(y_test, ind))

    brier = np.array(brier)
    min_brier = np.min(brier)
    br_thr = thresholds[np.argmin(brier)]

    if ax is None:
        ax = plt.gca()

    ax.set_title("Brier score curve")
    ax.plot(thresholds, brier, "b", label=f"min brier is {min_brier:.2f}")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid()
    ax.set_ylabel("brier")
    ax.set_xlabel("threshold")
    return (float("{0:.3f}".format(min_brier)), float("{0:.2f}".format(br_thr)))


def plot_rel_probs(y_test, y_pred, n: int = 1000, ax=None):
    t_df = pd.DataFrame(data={"scor": y_pred, "real": y_test})
    t_df.sort_values(by=["scor"], inplace=True)

    for k in range(n, 1, -1):
        if t_df.shape[0] % k == 0:
            print(k)
            break

    if k != n:
        print(f"Cannot split without remainder, so change n_bins to {k}")
    n = k

    parts = np.array_split(t_df.values, n)
    parts = np.mean(parts, axis=1, keepdims=True)
    parts = np.reshape(parts, (n, 2))

    if ax is None:
        ax = plt.gca()

    ax.set_title("Concordance of model predictions with prior probabilities")
    ax.plot(
        parts[:, 0],
        parts[:, 1],
        "bo",
        label=f"pred. lims are [{t_df.scor.iloc[0]:.5f}, {t_df.scor.iloc[-1]:.5f}]",
    )
    ax.legend(loc="lower right")
    x = np.linspace(0, 1, 3)
    ax.plot(x, x, "r")
    ax.grid()
    ax.set_xlabel("Ответ алгоритма")
    ax.set_ylabel("Оценка")


def plot_metrics(y_test, y_pred, model_id):  # , n_bins: int = 1000):
    """
    This function plots all metrics.
    """

    _, trace_roc1 = plot_roc_curve(y_test, y_pred, "y1", "x1")
    _, trace_pr1 = plot_pr_curve(y_test, y_pred, "y2", "x2")

    # _, trace_roc2 = plot_roc_curve(y_test, y_pred, "y3", "x3")
    # _, trace_pr2 = plot_pr_curve(y_test, y_pred, "y4", "x4")

    # _, trace_roc3 = plot_roc_curve(y_test, y_pred, "y5", "x5")
    # _, trace_pr3 = plot_pr_curve(y_test, y_pred, "y6", "x6")
    nrows, ncols = 1, 2
    # it1, it2 = itertools.tee(range(nrows * ncols))
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(21, 12))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=(
            "The ROC AUC",
            "The PR AUC",
            # "The ROC AUC",
            # "The PR AUC",
            # "The ROC AUC",
            # "The PR AUC",
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        # column_widths=[0.5, 0.5],
    )

    fig = add_suplot(trace_roc1, fig, "False Positive Rate", "True Positive Rate", 1, 1)
    fig = add_suplot(trace_pr1, fig, "Recall", "Precision", 1, 2)

    # fig = add_suplot(trace_roc2, fig, "False Positive Rate", "True Positive Rate", 1, 3)
    # fig = add_suplot(trace_pr2, fig, "Recall", "Precision", 2, 1)

    # fig = add_suplot(trace_roc3, fig, "False Positive Rate", "True Positive Rate", 2, 2)
    # fig = add_suplot(trace_pr3, fig, "Recall", "Precision", 2, 3)

    fig.update_layout(
        height=600,
        width=800,
        title_text=f"",
        showlegend=False,
        yaxis1=dict(scaleanchor="x1", scaleratio=1, constrain="domain"),
        yaxis2=dict(scaleanchor="x2", scaleratio=1, constrain="domain"),
        # yaxis3=dict(scaleanchor="x3", scaleratio=1),
        # yaxis4=dict(scaleanchor="x4", scaleratio=1),
        # yaxis5=dict(scaleanchor="x5", scaleratio=1),
        # yaxis6=dict(scaleanchor="x6", scaleratio=1),
        xaxis1=dict(range=[0, 1], constrain="domain"),
        xaxis2=dict(range=[0, 1], constrain="domain"),
        # xaxis3=dict(range=[0, 1], constrain="domain"),
        # xaxis4=dict(range=[0, 1], constrain="domain"),
        # xaxis5=dict(range=[0, 1], constrain="domain"),
        # xaxis6=dict(range=[0, 1], constrain="domain"),
    )
    # xaxis=dict(
    #     range=[0, 1],  # sets the range of xaxis
    #     # meanwhile compresses the xaxis by decreasing its "domain"
    # ),
    # xaxis2=dict(
    #     range=[0, 1],  # sets the range of xaxis
    #     constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
    # ),

    plot(fig, auto_open=False, filename="../DataForAntiFraud/images/" + model_id + "_metrics.html")

    # pr_auc = plot_pr_curve(
    #     y_test, y_pred, ax=axes[next(it1) // ncols, next(it2) % ncols]
    # )
    # max_mcc, mcc_thr = plot_matthews_corrcoef(
    #     y_test, y_pred, ax=axes[next(it1) // ncols, next(it2) % ncols]
    # )

    # max_k, kappa_thr = plot_cohen_kappa(
    #     y_test, y_pred, ax=axes[next(it1) // ncols, next(it2) % ncols]
    # )
    # min_brier, br_thr = plot_brier_scor(
    #     y_test, y_pred, ax=axes[next(it1) // ncols, next(it2) % ncols]
    # )
    # plot_rel_probs(
    #     y_test, y_pred, n=n_bins, ax=axes[next(it1) // ncols, next(it2) % ncols]
    # )

    # plt.show()
    # print(
    #     f"The rmse of model's prediction is: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}"
    # )
    # print(f"The Gini of model's prediction is: {Gini(y_test, y_pred):.4f}")
    # print(f"The ROC AUC of model's prediction is: {roc_auc:.4f}")
    # print(f"The PR AUC of model's prediction is: {pr_auc:.4f}")
    # print(f"Max Cohen's kappa is {max_k:.3f} with threshold = {kappa_thr:.2f}")
    # print(
    #     f"Max Matthews correlation coefficient is {max_mcc:.3f} with threshold = {mcc_thr:.2f}"
    # )
    # print(f"Min Brier score is {min_brier:.3f} with threshold = {br_thr:.2f}")
    return fig


def add_suplot(traces, fig, x_label, y_label, row, col):
    """
    microwrapper
    """
    for trace in traces:
        fig.add_trace(trace, row=row, col=col)
    fig.update_xaxes(title_text=x_label, row=row, col=col, range=[0, 1])
    fig.update_yaxes(title_text=y_label, row=row, col=col, range=[0, 1])
    return fig

