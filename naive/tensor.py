import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from utils import load_dataset
import pandas as pd


def compute_f1(labels, predicts, dim):
    tp, fp, fn = 0, 0, 0
    for ids in range(len(labels)):
        label = labels[ids]
        pred = predicts[ids]
        if label == dim and pred == dim:
            tp += 1
        elif label != dim and pred == dim:
            fp += 1
        elif label == dim and pred != dim:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def average_compute_f1(labels, predicts):
    return (
        compute_f1(labels, predicts, 0)
        + compute_f1(labels, predicts, 1)
        + compute_f1(labels, predicts, 2)
    ) / 3


output = False

X_train = np.squeeze(load_dataset("../data/train_pre_data.h5"), 1)
y_train = np.array(pd.read_csv("../data/train_pre_label.csv")["label"])

if output:
    X_testa = np.squeeze(load_dataset("../data/testa.h5"), 1)
    X_testb = np.squeeze(load_dataset("../data/testb.h5"), 1)
    X_test = np.concatenate((X_testa, X_testb))
    y_test = pd.read_csv("../data/submit.csv")
else:
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

rank = 10
factors = [parafac(tl.tensor(x), rank=rank)[1] for x in X_train]
X_train_transformed = np.array([tl.kruskal_to_tensor((np.ones(rank), f)).flatten() for f in factors])
factors_test = [parafac(tl.tensor(x), rank=rank)[1] for x in X_test]
X_test_transformed = np.array([tl.kruskal_to_tensor((np.ones(rank), f)).flatten() for f in factors_test])

clf = XGBClassifier(objective='multi:softmax', num_class=3)
clf.fit(X_train_transformed, y_train)

y_pred = clf.predict(X_test_transformed)

if output:
    y_test["label"] = y_pred
    y_test.to_csv("../data/submit.csv", index=False)
else:
    print(accuracy_score(y_test, y_pred))
    print(average_compute_f1(y_test, y_pred))
