import h5py
import imageio
import os

def load_dataset(path):
    with h5py.File(path, "r") as hf:
        data = hf["data"][:]
    return data

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