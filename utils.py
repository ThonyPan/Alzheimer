import h5py
import imageio
import os
import numpy as np
import pandas as pd

def load_dataset(path):
    with h5py.File(path, "r") as hf:
        data = hf["data"][:]
    return data

def load_train():
    X_train = load_dataset("../data/train_pre_data.h5")
    y_train = np.array(pd.read_csv("../data/train_pre_label.csv")["label"])
    return X_train, y_train

def load_test():
    X_val_a = load_dataset("../data/testa.h5")
    X_val_b = load_dataset("../data/testb.h5")
    X_val = np.concatenate((X_val_a, X_val_b))
    y_val = np.array(pd.read_csv("../data/submit.csv")["label"])

    X_val = X_val[y_val != '不相关']
    y_val = y_val[y_val != '不相关'].astype(int)

    return X_val, y_val

def f1_score(labels, predicts, dim):
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


def average_f1_score(labels, predicts):
    return (
        f1_score(labels, predicts, 0)
        + f1_score(labels, predicts, 1)
        + f1_score(labels, predicts, 2)
    ) / 3

def visualize_data(data, path):
    assert data.shape.__len__() == 3
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Convert data to 8-bit
    data = data.astype('uint8')

    for i in range(data.shape[0]):
        imageio.imwrite(f"{path}/image_{i}.png", data[i])

def export_ex2_data(data, path):
    import pyvista
    assert data.shape.__len__() == 3
    grid = pyvista.UniformGrid()
    grid.dimensions = data.shape
    grid.point_data["values"] = data.flatten(order='F')  

    grid.save(path)