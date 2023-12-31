import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MyDataSet import MyDataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils import load_dataset, load_train, load_test
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from generate_model import generate_model
from monai.networks.nets import DenseNet264

import argparse


def main(args):
    # 检查CUDA是否可用，并设置device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = load_train()
    X_val, y_val = load_test()
    
    # # 转换为PyTorch张量并移动到GPU（如果可用）
    # X_train_tensor = torch.tensor(X_train).float()
    # y_train_tensor = torch.tensor(y_train).long()  # 长整型适用于分类问题的标签
    # X_val_tensor = torch.tensor(X_val).float()
    # y_val_tensor = torch.tensor(y_val).long()

    # 创建数据加载器
    train_dataset = MyDataset(datas=X_train, labels=y_train, phase='train')
    val_dataset = MyDataset(datas=X_val, labels=y_val, phase='train')
    # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = generate_model(cnn_name=args.cnn_name, n_classes=args.n_classes, in_channels=args.in_channels)
    # model = DenseNet264(spatial_dims=3, in_channels=1, out_channels=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    num_epochs = 100
    min_val_f1_score = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        train_correct_sum = 0
        train_simple_cnt = 0
        train_f1_score = 0
        y_train_true = []
        y_train_pred = []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, train_predicted = torch.max(output.data, 1)
            train_correct_sum += (target.data == train_predicted).sum().item()
            train_simple_cnt += target.size(0)
            y_train_true.extend(np.ravel(np.squeeze(target.cpu().detach().numpy())).tolist())
            y_train_pred.extend(np.ravel(np.squeeze(train_predicted.cpu().detach().numpy())).tolist())
        # 验证模型
        model.eval()
        val_acc = 0
        val_correct_sum = 0
        val_simple_cnt = 0
        val_loss = 0 
        val_f1_score = 0
        y_val_true = []
        y_val_pred = []
        with torch.no_grad():
            for data, target in val_loader:     
                data, target = data.to(device), target.to(device)     
                output = model(data)
                val_loss += criterion(output, target).item()
                _, val_predicted = torch.max(output.data, 1)
                val_correct_sum += (target.data == val_predicted).sum().item()
                val_simple_cnt += target.size(0)
                y_val_true.extend(np.ravel(np.squeeze(target.cpu().detach().numpy())).tolist())
                y_val_pred.extend(np.ravel(np.squeeze(val_predicted.cpu().detach().numpy())).tolist())


        # 打印每个epoch的损失和准确率
        train_loss = train_loss/len(train_loader)
        val_loss = val_loss/len(val_loader)
        train_acc = train_correct_sum/train_simple_cnt
        val_acc = val_correct_sum/val_simple_cnt
        train_f1_score = f1_score(y_train_true,y_train_pred,average='macro')
        val_f1_score = f1_score(y_val_true,y_val_pred,average='macro')
        print('Epochs: {}/{}...'.format(epoch+1, num_epochs),
            'Train Loss:{:.3f}...'.format(train_loss),
            'Train Accuracy:{:.3f}...'.format(train_acc),
            'Train F1 Score:{:.3f}...'.format(train_f1_score),
            'Val Loss:{:.3f}...'.format(val_loss),
            'Val Accuracy:{:.3f}...'.format(val_acc),
            'Val F1 Score:{:.3f}'.format(val_f1_score))
        if val_f1_score > min_val_f1_score:
            torch.save(model, os.path.join(args.model_weights_path, args.cnn_name + '_2' + '.pth'))
            print('Save model at path: {}'.format(os.path.join(args.model_weights_path, args.cnn_name + '_2' + '.pth')))
            epochs_no_improve = 0
            min_val_f1_score = val_f1_score
        else:
            epochs_no_improve += 1
            if epochs_no_improve == 10:
                print('Early stopping!')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--cnn_name", type=str, default="cnn")
    args.add_argument("--n_classes", type=int, default=3)
    args.add_argument("--in_channels", type=int, default=1)
    args.add_argument("--model_weights_path", type=str, default="weights")

    args = args.parse_args()
    main(args)
