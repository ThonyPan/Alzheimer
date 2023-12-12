import h5py

def load_dataset(path):
    with h5py.File(path, "r") as hf:
        data = hf["data"][:]
    return data