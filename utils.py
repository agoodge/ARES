import torch
import torchvision
import numpy as np
import pandas as pd

def get_data(dataset):

    if dataset == "mnist":
        train = torchvision.datasets.MNIST(root='data/',
                                       train=True, 
                                       transform=None,
                                       download=True)
        height, width = train.data.shape[1:]
        train_x = train.data.reshape(-1,height*width)/255.0
        train_y = train.targets
    
        test = torchvision.datasets.MNIST(root='data/',
                                       train=False, 
                                       transform=None,
                                       download=True)
        test_x = test.data.reshape(-1,height*width)/255.0
        test_y = test.targets

    if dataset == "fmnist":
        train = torchvision.datasets.FashionMNIST(root='data/',
                                       train=True, 
                                       transform=None,
                                       download=True)
        height, width = train.data.shape[1:]
        train_x = train.data.reshape(-1,height*width)/255.0
        train_y = train.targets
        
        test = torchvision.datasets.FashionMNIST(root='data/',
                                       train=False, 
                                       transform=None,
                                       download=True)
        test_x = test.data.reshape(-1,height*width)/255.0
        test_y = test.targets

    elif dataset in ['otto','snsr']: 
        if dataset == 'otto':
            data = pd.read_csv("data/otto/train.csv")                    
        elif dataset == 'snsr':
            data = pd.read_csv("data/snsr/snsr.csv")
        data = torch.tensor(data.to_numpy())
        data[:,-1] -= 1
        test_idx = np.random.choice(np.arange(0,len(data)), np.int(0.2*len(data)), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(data)), test_idx)
        train_x, train_y = data[train_idx,:-1], data[train_idx,-1]
        test_x, test_y = data[test_idx,:-1],  data[test_idx,-1]
    
    elif dataset == 'eopt':
        data = torch.tensor(pd.read_csv("data/eopt/HRSS.csv").to_numpy())
        data[:,-1] == 1
        normal_idx = np.argwhere(data[:,-1] == 0).squeeze(); anom_idx = np.argwhere(data[:,-1] == 1).squeeze()
        test_idx = np.concatenate((anom_idx,np.random.choice(normal_idx, len(anom_idx), replace = False)))
        train_idx = np.setdiff1d(np.arange(len(data)), test_idx)
        train_data = data[train_idx]
        test_data = data[test_idx]
        train_x, train_y = train_data[:,:-1], train_data[:,-1]
        test_x, test_y = test_data[:,:-1], test_data[:,-1]
        
    elif dataset == 'mi-v':
        df = pd.read_csv("data/mi/experiment_01.csv")
        for i in ['02','03','11','12','13','14','15','17','18']:
            data = pd.read_csv("data/mi/experiment_%s.csv" %i)
            df = pd.concat([df,data], ignore_index = True)
        normal_count = len(df)
        for i in ['06','08','09','10']:
            data = pd.read_csv("data/mi/experiment_%s.csv" %i)
            df = pd.concat([df,data], ignore_index = True)    
        machining_process_one_hot = pd.get_dummies(df['Machining_Process'])
        df = pd.concat([df.drop(['Machining_Process'],axis=1),machining_process_one_hot],axis=1)
        data = df.to_numpy()
        normal_data = data[:normal_count]
        anomaly_data = data[normal_count:]
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anomaly_data), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = np.zeros(len(train_x))
        test_x = np.concatenate((anomaly_data,normal_data[test_idx]))
        test_y  = np.concatenate((np.ones(len(anomaly_data)),np.zeros(len(test_idx))))

    elif dataset == 'mi-f':
        df = pd.read_csv("data/mi/experiment_01.csv")
        for i in ['02','03','06','08','09','10','11','12','13','14','15','17','18']:
            data = pd.read_csv("data/mi/experiment_%s.csv" %i)
            df = pd.concat([df,data], ignore_index = True)
            normal_count= len(df)
        for i in ['04', '05', '07', '16']: 
            data = pd.read_csv("data/mi/experiment_%s.csv" %i)
            df = pd.concat([df,data], ignore_index = True)  
        machining_process_one_hot = pd.get_dummies(df['Machining_Process'])
        df = pd.concat([df.drop(['Machining_Process'],axis=1),machining_process_one_hot],axis=1)
        normal_data = df.values[:normal_count]
        anomaly_data = df.values[normal_count:]
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anomaly_data), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = np.zeros(len(train_x))
        test_x = np.concatenate((anomaly_data,normal_data[test_idx]))
        test_y  = np.concatenate((np.ones(len(anomaly_data)),np.zeros(len(test_idx))))
    
    return train_x, train_y, test_x, test_y

def split_classes(x, y, normal_classes, anom_classes):
# seperate normal from anomalous data
    idx_normal = []
    idx_anom = []
    for i in range(len(y)):
        if y[i] in normal_classes:
            idx_normal.append(i)
        if y[i] in anom_classes:
            idx_anom.append(i)
    #get normal classes only
    x_normal = x[idx_normal]
    y_normal = torch.zeros(len(idx_normal))
    x_anom = x[idx_anom]
    y_anom = torch.ones(len(idx_anom))
    
    return x_normal, y_normal, x_anom, y_anom

def downsample(test_x_normal,test_y_normal,test_x_anom,test_y_anom):
    if len(test_x_anom) > len(test_x_normal):
        rand_idx = np.random.randint(low = 0, high = len(test_x_anom), size=len(test_x_normal))
        test_x_anom = test_x_anom[rand_idx]
        test_y_anom = test_y_anom[rand_idx]    
    else:
        rand_idx = np.random.randint(low = 0, high = len(test_x_normal), size =  len(test_x_anom))
        test_x_normal = test_x_normal[rand_idx]
        test_y_normal = test_y_normal[rand_idx]

    x = torch.cat((test_x_normal,test_x_anom), 0)
    y = torch.cat((test_y_normal,test_y_anom), 0)
    
    return x, y

def get_hidden(dataset):

    if dataset in ['mnist','fmnist']:
        hidden_size = [[784, 600, 500, 400, 300, 200, 100, 20],
                     [20, 100, 200, 300, 400, 500, 600, 784]]
        
    elif dataset == 'otto':
        hidden_size = [[93, 88, 84, 79, 74, 70, 65],
                       [65, 70, 74, 79, 84, 88, 93]]
        
    elif dataset == 'snsr':
        hidden_size = [[48, 43, 37, 32, 27, 21, 16],
                       [16, 21, 27, 32, 37, 43, 48]]

    elif dataset == 'eopt':
        hidden_size = [[20, 18, 16, 14, 12, 10, 8],
                      [8, 10, 12, 14, 16, 18, 20]]
        
    elif dataset in ['mi-v', 'mi-f']:
        hidden_size = [[58, 52, 47, 41, 35, 30, 24],
                       [24, 30, 35, 41, 47, 52, 58]]
    
    return hidden_size