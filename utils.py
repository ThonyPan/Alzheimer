import h5py
import imageio
import os

def load_dataset(path):
    with h5py.File(path, "r") as hf:
        data = hf["data"][:]
    return data

def visualize_data(data, path):
    assert data.shape.__len__() == 3
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Convert data to 8-bit
    data = data.astype('uint8')

    for i in range(data.shape[0]):
        imageio.imwrite(f"{path}/image_{i}.png", data[i])