import model
import utils

import argparse
import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.dataset in ['eopt','mi-v','mi-f']:
        #load train and test data
        train_x, train_y, test_x, test_y = utils.get_data(args.dataset)
        train_x, val_x = train_test_split(train_x, test_size = 0.1, shuffle = True)

        model_save_path = "saved_models/%s/novelty/%s/0" %(args.dataset,args.normality)

    else:
        #load train and test data
        train_x, train_y, test_x, test_y = utils.get_data(args.dataset)
               
        if args.normality == 'one-class':
            normal_classes = [args.target_class]
            anom_classes = np.setdiff1d(np.unique(train_y).tolist(),normal_classes).tolist()
        if args.normality == 'multi-class':
            anom_classes = [args.target_class]
            normal_classes = np.setdiff1d(np.unique(train_y).tolist(),anom_classes).tolist()

        # split the data from normal classes from the anomalous classes
        train_x, train_class, train_x_anom, train_y_anom = utils.split_classes(train_x,train_y,normal_classes,anom_classes)
        test_x_normal, test_y_normal, test_x_anom, test_y_anom = utils.split_classes(test_x, test_y, normal_classes, anom_classes)
        
        # downsample to 50:50 normal:anomaly split in the test set
        test_x, test_y = utils.downsample(test_x_normal, test_y_normal, test_x_anom, test_y_anom)

        train_x, val_x = train_test_split(train_x, test_size = 0.1, shuffle = True)

        model_save_path = "saved_models/%s/novelty/%s/%d" %(args.dataset,args.normality,args.target_class)
        
    # normalize
    if args.dataset in ['mnist','fmnist']:
        train_mean = train_x.mean(); train_std = train_x.std()
        train_x = (train_x -  train_mean)/train_std
        val_x = (val_x -  train_mean)/train_std
        test_x = (test_x -  train_mean)/train_std
    else:
        # column-wise normalization
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)
        test_x = scaler.transform(test_x)

        # torch tensors
        train_x = torch.tensor(train_x, dtype = torch.float32)
        val_x = torch.tensor(val_x, dtype = torch.float32)
        test_x = torch.tensor(test_x, dtype = torch.float32)

    if args.load_model:
        checkpoints = torch.load("%s/ae.pth" %(model_save_path))
        enc_hidden = checkpoints['hidden_size'][0]
        dec_hidden = checkpoints['hidden_size'][1]
        ae = model.AE(enc_hidden, dec_hidden).to(device)
        ae.load_state_dict(checkpoints['model_state_dict'])
    else:
        # instantiate autoencoder model to be trained
        enc_hidden_size, dec_hidden_size = utils.get_hidden(args.dataset)   
        ae = model.AE(enc_hidden_size, dec_hidden_size).to(device)
        # train model
        ae, train_loss, val_loss, epoch = model.train_model(ae, train_x, val_x, args, device)
        #save trained model
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': ae.state_dict(),
                    'hidden_size': [enc_hidden_size,dec_hidden_size],
                    'train_loss': train_loss,
                    'val_loss': val_loss
                    }, "%s/ae.pth" %model_save_path)

    # test model
    score = model.test_model(ae, train_x, test_x, args, device)
    # print score
    print(f" dataset: {args.dataset} \t n_neighbours: {args.n_neighbours} \t score {roc_auc_score(test_y,score)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type = int, default = 250)
    parser.add_argument("--n_epochs", type = int, default = 350)
    parser.add_argument("--dataset", type=str, default = 'mnist')
    parser.add_argument("--n_neighbours", type = int, default = 10)
    parser.add_argument("--alpha", type = float, default = 0.5, help = "scaling factor")
    parser.add_argument("--lr", type = float, default = 0.001, help = "learning rate")
    parser.add_argument("--wd", type = float, default = 1e-6, help = "weight decay")
    parser.add_argument("--verbose", type = int, default = 1, help = "view training progress")
    parser.add_argument("--normality", type = str, default = 'one-class', help = "one-class vs multi-class normality setting")
    parser.add_argument("--load_model", type = int, default = 1, help = "load a previously trained model")
    parser.add_argument("--target_class", type = int, default = 0, help = "anomalous class if --normality == one-class, normal class if --normality == multi-class")
    args = parser.parse_args()

    main(args)