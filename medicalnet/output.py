import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MyDataSet import MyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from utils import load_dataset
from medicalnet import generate_model

def load_checkpoint(filepath, device='cuda'):
    checkpoint = torch.load(filepath, map_location=device)
    model, _ = generate_model(phase='test')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_testa = load_dataset("../data/testa.h5")
    X_testb = load_dataset("../data/testb.h5")
    X_test = np.concatenate((np.array(X_testa), np.array(X_testb)))
    test_datasets = MyDataset(datas=X_test, phase='test')
    test_loader = DataLoader(dataset=test_datasets)
    model = load_checkpoint('medicalnet_3d_resnet10_checkpoint.pth', device)
    model.eval()

    result_df = pd.DataFrame(columns=['testa_id','label'])

    with torch.no_grad():
        for ii, image in tqdm(enumerate(test_loader)):
            image = image.to(device)
            output = model(image)
            _, indexs = torch.max(output.data, 1)
            indexs = np.squeeze(indexs.cpu().detach().numpy()).tolist()
            if ii < len(X_testa):
                result_df.loc[result_df.shape[0]] = [('testa_{}'.format(ii)),indexs]
            else:
                result_df.loc[result_df.shape[0]] = [('testb_{}'.format(ii - len(X_testa))),indexs]


    result_df.to_csv('../data/submit.csv', index=False)
