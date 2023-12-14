import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from utils import load_dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import resnet
from MyDataSet import MyDataset

def generate_model(no_cuda=False, phase='train'):
    model = resnet.resnet10(sample_input_W=79, sample_input_H=95, sample_input_D=79, num_seg_classes=3)
    
    if not no_cuda:
        if torch.cuda.device_count()> 1:
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            net_dict = model.state_dict() 
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()
    
    if phase != 'test':
        pretrain = torch.load("resnet_10_23dataset.pth")
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = [] 
        for pname, p in model.named_parameters():
            for layer_name in ['avgpool','fc']:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 'new_parameters': new_parameters}

        return model, parameters
    return model, model.parameters()

def save_checkpoint(epochs,optimizer,model,filepath):
    checkpoint = {'epochs':epochs, 'optimizer_state_dict':optimizer.state_dict(), 'model_state_dict':model.state_dict()}
    torch.save(checkpoint,filepath)


if __name__ == '__main__':
    # 检查CUDA是否可用，并设置device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 假定X_train和y_train已经被定义并且格式正确
    X_train = load_dataset("../data/train_pre_data.h5")
    y_train = np.array(pd.read_csv("../data/train_pre_label.csv")["label"])
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    train_datasets = MyDataset(datas=X_train, labels=y_train, phase='train')
    val_datasets = MyDataset(datas=X_val, labels=y_val, phase='train')
    
    train_loader = DataLoader(dataset=train_datasets, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_datasets, batch_size=8, shuffle=False)
    
    model, parameters = generate_model()
    criterion = nn.CrossEntropyLoss()
    params = [
            { 'params': parameters['base_parameters'], 'lr': 0.001 }, 
            { 'params': parameters['new_parameters'], 'lr': 0.001*100 }
            ]
    optimizer = optim.Adam(params, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
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
            output = model.forward(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, train_predicted = torch.max(output.data, 1)
            train_correct_sum += (target.data == train_predicted).sum().item()
            train_simple_cnt += target.size(0)
            y_train_true.extend(np.ravel(np.squeeze(target.cpu().detach().numpy())).tolist())
            y_train_pred.extend(np.ravel(np.squeeze(train_predicted.cpu().detach().numpy())).tolist())
    
        scheduler.step()
    
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
            save_checkpoint(epoch+1, optimizer, model, "medicalnet_3d_resnet10_checkpoint.pth")
            epochs_no_improve = 0
            min_val_f1_score = val_f1_score
        else:
            epochs_no_improve += 1
            if epochs_no_improve == 10:
                print('Early stopping!')
