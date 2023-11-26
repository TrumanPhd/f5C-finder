# evaluation utils
# Author: Guohao Wang
# Date: 2023/11/19

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, auc, roc_curve
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve
import numpy as np
from scipy.interpolate import interp1d

# calculation
def evaluation_metrics(y_test, y_pred, th=0.5):
    y_predlabel = [(0 if item < th else 1) for item in y_pred]
    y_test = np.array([(0 if item < 1 else 1) for item in y_test])
    y_predlabel = np.array(y_predlabel)
    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    SP = tn * 1.0 / ((tn + fp) * 1.0)
    SN = tp*1.0/((tp+fn)*1.0)
    MCC = matthews_corrcoef(y_test, y_predlabel)
    Recall = recall_score(y_test, y_predlabel)
    Precision = precision_score(y_test, y_predlabel)
    F1 = f1_score(y_test, y_predlabel)
    Acc = accuracy_score(y_test, y_predlabel)
    AUC = roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    AUPR = auc(recall_aupr, precision_aupr)
    y_pred_int = y_predlabel
    return Recall, SN,SP, MCC, Precision, F1, Acc, AUC, AUPR, tp, fn, tn, fp, y_pred_int

# Set font and border parameters
font = {'family': 'Arial', 'weight': 'bold', 'size': 16}
plt.rc('font', **font)

# Plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred_int, classes, save_path, normalize=False, show=False):
    cm = confusion_matrix(y_true, y_pred_int)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')

    # Set borders bold
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right', weight='bold')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, weight='bold')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", weight='bold')

    ax.set_xlabel('Predicted label', weight='bold')
    ax.set_ylabel('True label', weight='bold')
    ax.set_title('Confusion Matrix', weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

# Plot the ROC curve
def plot_ROC_curve(fpr, tpr, auc_value, save_path,show=False):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_value:.3f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Set axes bold
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel('False Positive Rate', weight='bold')
    ax.set_ylabel('True Positive Rate', weight='bold')
    ax.set_title('ROC Curve', weight='bold')
    ax.legend(loc='lower right', prop={'weight': 'bold'})

    # Make tick marks bold and point inwards
    ax.tick_params(axis='both', which='both', length=5, width=2, direction='in')

    # Add light gray dashed grid lines
    ax.grid(color='lightgray', linestyle='--', linewidth=1)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

# Plot the precision-recall curve
def plot_PR_curve(y_trues, y_outputs, save_path,show=False):
    precision, recall, _ = precision_recall_curve(y_trues, y_outputs)
    AUPR = auc(recall, precision)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='darkorange', lw=2, label=f'AUPR = {AUPR:.3f}')
    ax.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')

    # Set axes bold
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel('Recall', weight='bold')
    ax.set_ylabel('Precision', weight='bold')
    ax.set_title('Precision-Recall Curve', weight='bold')
    ax.legend(loc='lower right', prop={'weight': 'bold'})

    # Make tick marks bold and point inwards
    ax.tick_params(axis='both', which='both', length=5, width=2, direction='in')

    # Add light gray dashed grid lines
    ax.grid(color='lightgray', linestyle='--', linewidth=1)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

#multi ROC
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score

def MultiROC(curves, save_path, show=False):
    fig, ax = plt.subplots()

    for curve in curves:
        fpr, tpr, auc_value, model_name = curve
        ax.plot(fpr, tpr, lw=2, label=f'{model_name} - AUC = {auc_value:.3f}')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Set axes bold
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel('False Positive Rate', weight='bold')
    ax.set_ylabel('True Positive Rate', weight='bold')
    ax.set_title('ROC Curve', weight='bold')
    ax.legend(loc='lower right', prop={'weight': 'bold'})

    # Make tick marks bold and point inwards
    ax.tick_params(axis='both', which='both', length=5, width=2, direction='in')

    # Add light gray dashed grid lines
    ax.grid(color='lightgray', linestyle='--', linewidth=1)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

dict_name = {'RF':'Fusion-RF','svm':'Fusion-SVM','ada':'Fusion-ADA'}

def curve_generator(train_label, y_pred_train, model_name):
    fpr, tpr, _ = roc_curve(train_label, y_pred_train)
    AUC = roc_auc_score(train_label, y_pred_train)
    return (fpr, tpr, AUC, dict_name[model_name])

def MultiPR(curves, save_path, show=False):
    fig, ax = plt.subplots()

    for curve in curves:
        y_trues, y_outputs, model_name = curve
        precision, recall, _ = precision_recall_curve(y_trues, y_outputs)
        AUPR = auc(recall, precision)
        ax.plot(recall, precision, lw=2, label=f'{model_name} - AUPR = {AUPR:.3f}')

    ax.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')

    # Set axes bold
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel('Recall', weight='bold')
    ax.set_ylabel('Precision', weight='bold')
    ax.set_title('Precision-Recall Curve', weight='bold')
    ax.legend(loc='lower right', prop={'weight': 'bold'})

    # Make tick marks bold and point inwards
    ax.tick_params(axis='both', which='both', length=5, width=2, direction='in')

    # Add light gray dashed grid lines
    ax.grid(color='lightgray', linestyle='--', linewidth=1)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
"""
# 示例用法：
# 在调用 curve_generator 时传入 model_name
curve1 = curve_generator(train_label, y_pred_train, "Model 1")
curve2 = curve_generator(train_label, y_pred_train2, "Model 2")

# 在调用 MultiROC 和 MultiPR 时传入包含 model_name 的元组
curves = [curve1, curve2]

MultiROC(curves, 'roc_curves.png', show=True)
MultiPR(curves, 'pr_curves.png', show=True)
"""
"""
def Aiming(y_hat, y):
    '''
    the “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y_hat[v])
    return sorce_k / n


def Coverage(y_hat, y):
    '''
    The “Coverage” rate (also called “Recall”) is to reflect the average ratio of the
    correctly predicted labels over the real labels; to measure the percentage of the
    real labels that are covered by the hits of prediction.
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y[v])

    return sorce_k / n


def Accuracy(y_hat, y):
    '''
    The “Accuracy” rate is to reflect the average ratio of correctly predicted labels
    over the total labels including correctly and incorrectly predicted labels as well
    as those real labels but are missed in the prediction
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / union
    return sorce_k / n


def AbsoluteTrue(y_hat, y):
    '''
    same
    '''

    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            sorce_k += 1
    return sorce_k / n


def AbsoluteFalse(y_hat, y):
    '''
    hamming loss
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        sorce_k += (union - intersection) / m
    return sorce_k / n


def evaluate(y_hat, y):
    aiming = Aiming(y_hat, y)
    coverage = Coverage(y_hat, y)
    accuracy = Accuracy(y_hat, y)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_false = AbsoluteFalse(y_hat, y)
    return aiming, coverage, accuracy, absolute_true, absolute_false
"""