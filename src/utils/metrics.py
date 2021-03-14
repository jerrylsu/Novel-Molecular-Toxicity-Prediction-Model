from sklearn import metrics
from typing import List
import pandas as pd


class Metrics(object):
    def __init__(self):
        pass

    def calculate_accuracy(self, y_true: List, y_pred: List):
        return metrics.accuracy_score(y_true, y_pred, normalize=True)

    def calculate_auc(self, y_true: List, y_pred: List):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        return metrics.auc(fpr, tpr)

    def calculate_precision(self, y_true: List, y_pred: List):
        return metrics.precision_score(y_true, y_pred)

    def calculate_recall(self, y_true: List, y_pred: List):
        return metrics.recall_score(y_true, y_pred)

    def calculate_f1(self, y_true: List, y_pred: List):
        return metrics.f1_score(y_true, y_pred)

    def classification_report(self, y_true: List, y_pred: List):
        return metrics.classification_report(y_true, y_pred)

    def calculate_confusion_matrix(self, y_true: List, y_pred: List):
        return metrics.confusion_matrix(y_true, y_pred)
