import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def evaluate_model(model, features, target):

    predictions = model.predict(features)

    probabilities = model.predict_proba(features)[:, 1]

    f1 = f1_score(target, predictions)
    roc = roc_auc_score(target, probabilities)

    return f1, roc


def test_model(model, features_test, target_test):

    probabilities = model.predict_proba(features_test)[:, 1]

    predictions = model.predict(features_test)

    f1 = f1_score(target_test, predictions)

    roc = roc_auc_score(target_test, probabilities)

    print("F1:", f1)
    print("ROC-AUC:", roc)

    return probabilities


def plot_roc_curve(target_test, probabilities):

    fpr, tpr, thresholds = roc_curve(target_test, probabilities)

    plt.figure()

    plt.plot(fpr, tpr, label="Model")

    plt.plot([0, 1], [0, 1], linestyle='--', label="Random model")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.show()