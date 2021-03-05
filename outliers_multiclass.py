"""
main file for experiments with contaminated training set for multi-class datasets
"""

                        ############# IMPORTS #################

import variables as var
import model
import utils

import torch
import os
import numpy as np
import sklearn.neighbors as neighbours
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

                     ################ SETUP #######################      
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#"MNIST", "FMNIST", "OTTO", "STL", "SNSR"
dataset ='FMNIST'
data_type = 'outlier'
#"unimodal" or "multimodal" normal set
normality_type = 'multimodal'
# load a trained model or train a new one
load_model = True
#store scores for each model
AE = []
L_D = []
L_R = []
ARES_G = []
ARES = []
#load train and test data
all_train_x, all_train_y, all_test_x, all_test_y = utils.get_data(dataset)
#collect all the class labels in dataset
all_classes = np.unique(all_train_y).tolist()
# repeat experiments over each class
for i in [0]:
    digit = i
    
            ############# DATA PROCESSING #####################
    if normality_type == 'unimodal':
        normal_classes = [digit]
        anom_classes = np.setdiff1d(all_classes,normal_classes).tolist()
    if normality_type == 'multimodal':
        anom_classes = [digit]
        normal_classes = np.setdiff1d(all_classes,anom_classes).tolist()
    # split the normal classes from the anomalous classes
    train_x, train_class, train_x_anom, train_y_anom = utils.split_classes(all_train_x,all_train_y,normal_classes,anom_classes)
    test_x_normal, test_y_normal, test_x_anom, test_y_anom = utils.split_classes(all_test_x, all_test_y, normal_classes, anom_classes)

    # include outliers equal to n% of the training set
    n = 0.005 #0.01, 0.05, 0.1, 0.2
    idx = np.random.choice(np.arange(len(train_x_anom)), size = np.int(n*len(train_x)/(1-n)), replace = True)
    train_x = torch.cat((train_x, train_x_anom[idx]),0)
    train_class = torch.cat((train_class, train_y_anom[idx]),0)

    if dataset == 'STL':
        # STL is a smaller dataset, so we use the anomalies discarded from the train set too to increase the size
        test_x_anom = torch.cat((train_x_anom,test_x_anom),0)
        test_y_anom = torch.cat((train_y_anom,test_y_anom),0)
            
    # downsample to get 50:50 normal:anomaly split in the test set
    test_x_normal, test_y_normal, test_x_anom, test_y_anom = utils.downsample(test_x_normal, test_y_normal, test_x_anom, test_y_anom)
    # combine normal and anomaly test data into one test set
    test_x, test_class, test_y = utils.combine(test_x_normal, test_y_normal, test_x_anom, test_y_anom)
    # shuffle rows
    perm = torch.randperm(len(train_x))
    train_x = train_x[perm]
    train_class = train_class[perm]
    perm = torch.randperm(len(test_x))
    test_x = test_x[perm]
    test_class = test_class[perm]
    test_y = test_y[perm]
    # normalize
    if dataset in ['MNIST','FMNIST']:
        # normalize over all pixels rather than by column
        train_x, mean, std = utils.normalize(train_x)
        test_x, _, _ = utils.normalize(test_x, mean, std)
    else:
        # normalize by column for tabular data
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
        train_x = torch.tensor(train_x, dtype = torch.float32); test_x = torch.tensor(test_x, dtype = torch.float32)
        
    # move to gpu
    train_x = train_x.to(device)
    test_x = test_x.to(device)
   
              ############## MODEL LOADING ##################    
    path = "saved_models/%s/%s/%s/%s/%d" %(dataset,data_type,normality_type,str(n),digit)
    if load_model == True:

        checkpoints = torch.load("%s/ae.pth" %(path))
        enc_hidden = checkpoints['hidden_size'][0]
        dec_hidden = checkpoints['hidden_size'][1]
        ae = model.AE(enc_hidden, dec_hidden).to(device)
        ae.load_state_dict(checkpoints['model_state_dict'])

    else: 
        #retrieve hidden layer sizes for the given dataset
        hidden = var.get_hidden(dataset)
        ae = model.AE(hidden[0], hidden[1]).to(device)

        # train model
        ae, train_loss, val_loss, epoch = model.ae_train(ae, train_x, var.criterion)

        #SAVE MODEL
        if not os.path.exists(path):
            os.mkdir(path)

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': ae.state_dict(),
                    'hidden_size': [hidden[0],hidden[1]],
                    'train_loss': train_loss,
                    'val_loss': val_loss
                    }, "%s/ae.pth" %path)
                    ################ ANOMALY DETECTION ####################
    ae.eval()
    with torch.no_grad():
        train_enc, train_rec = ae(train_x)
        test_enc, test_rec = ae(test_x)

    train_error = ((train_rec - train_x)**2).mean(axis=1)
    test_error = ((test_rec - test_x)**2).mean(axis=1)
    train_enc, train_rec = train_enc.cpu().detach().numpy(), train_rec.cpu().detach().numpy()
    test_enc, test_rec= test_enc.cpu().detach().numpy(), test_rec.cpu().detach().numpy()
    train_error, test_error = train_error.cpu().detach().numpy(), test_error.cpu().detach().numpy()
    
    density_type = 'LOF'
    n_neighbors = 10
    # LOCAL DENSITY SCORE
    nb = neighbours.NearestNeighbors(algorithm = 'brute', n_neighbors = 100)
    nb.fit(train_enc)
    nb_distance, nb_idx = nb.kneighbors(test_enc, n_neighbors = 100)
    
    if density_type == 'KNN': #n_neighbors = 5,20,40
        local_density_score = np.mean(nb_distance[:,:n_neighbors],axis=1)
    
    if density_type == 'GAUSSIAN': #n_neighbors = 100
        mu, variance = utils.find_gaussians(train_enc, train_class, all_classes)
        class_distance = utils.distance(test_enc, mu, variance)
        local_density_score = np.nanmin(class_distance,axis=1)
    
    if density_type =='LOF': #n_neighbors = 10, 40, 100
        lof = neighbours.LocalOutlierFactor(novelty=True, n_neighbors = n_neighbors)
        lof.fit(train_enc)
        local_density_score = -lof.decision_function(test_enc)
    
    # LOCAL RECONSTRUCTION SCORE
    nb_reconstruction = np.empty((len(test_enc),100))
    for i in range(len(test_enc)):
        nb_reconstruction[i] = train_error[nb_idx[i]] 
    nb_median = np.median(nb_reconstruction[:,:n_neighbors],axis=1)
    local_reconstruction_score = test_error - nb_median

    #scaling factor
    alpha = 0.5
                        ############# SCORES #################
    # regular AE score
    scores = test_error
    AE.append((100*roc_auc_score(test_y,scores)))

    # regular local density score
    scores = local_density_score
    L_D.append((100*roc_auc_score(test_y,scores)))
    
    # local reconstruction score
    scores = local_reconstruction_score
    L_R.append((100*roc_auc_score(test_y,scores)))

    # AE and local density score
    scores = test_error + alpha*local_density_score
    ARES_G.append((100*roc_auc_score(test_y,scores)))

    # ARES: local density and reconstruction score
    scores = local_reconstruction_score + alpha*local_density_score
    ARES.append((100*roc_auc_score(test_y,scores)))

# print scores
print('AE', [ round(elem, 2) for elem in AE],
      round(np.mean(AE),2), 
      round(np.std(AE),2))

print('L_D',[ round(elem, 2) for elem in L_D],
      round(np.mean(L_D),2), 
      round(np.std(L_D),2))

print('L_R',[ round(elem, 2) for elem in L_R],
      round(np.mean(L_R),2), 
      round(np.std(L_R),2))

print('ARES_G',[ round(elem, 2) for elem in ARES_G],
      round(np.mean(ARES_G),2), 
      round(np.std(ARES_G),2))

print('ARES',[ round(elem, 2) for elem in ARES],
      round(np.mean(ARES),2), 
      round(np.std(ARES),2))