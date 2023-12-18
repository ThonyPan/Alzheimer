import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_dataset
from generate_model import generate_model
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from MyDataSet import MyDataset

def load_checkpoint(model_weights_path, cnn_name, model_depth=None, n_classes=None, in_channels=None, sample_size=None, device='cuda'):
    filepath = os.path.join(model_weights_path, cnn_name + '.pth')
    model = torch.load(filepath, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_testa = load_dataset("../data/testa.h5")
    X_testb = load_dataset("../data/testb.h5")
    X_test = np.concatenate((np.array(X_testa), np.array(X_testb)))
    y_test = np.array(pd.read_csv("../data/test_label.csv")["label"])
    test_datasets = MyDataset(datas=X_test, phase='test')
    test_loader = DataLoader(test_datasets, batch_size=1)

    model = load_checkpoint(args.model_weights_path, args.cnn_name, n_classes=args.n_classes, in_channels=args.in_channels, device=device)

    result_df = pd.DataFrame(columns=['testa_id','label'])

    results = []
    with torch.no_grad():
        for ii, image in tqdm(enumerate(test_loader)):
            image = image.to(device)
            output = model(image)
            _, indexs = torch.max(output.data, 1)
            results.append(indexs.item())
            indexs = np.squeeze(indexs.cpu().detach().numpy()).tolist()
            if ii < len(X_testa):
                result_df.loc[result_df.shape[0]] = [('testa_{}'.format(ii)),indexs]
            else:
                result_df.loc[result_df.shape[0]] = [('testb_{}'.format(ii - len(X_testa))),indexs]

    results = np.array(results)
    results = results[y_test != '不相关']
    y_test = [int(i) for i in y_test if i != '不相关']
    print("F1 score: {}".format(f1_score(y_test, results, average='macro')))
    result_df.to_csv('../data/' + args.cnn_name + '_submit.csv', index=False)
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--cnn_name", type=str, default="cnn")
    args.add_argument("--n_classes", type=int, default=3)
    args.add_argument("--in_channels", type=int, default=1)
    args.add_argument("--model_weights_path", type=str, default="weights")

    args = args.parse_args()
    main(args)
