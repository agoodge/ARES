"""
main file for experiments with contaminated training set for MI-F, MI-V and EOPT datasets
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
#EOPT", "MI-V", "MI-F"
dataset ='EOPT'
data_type = 'outlier'
# load a trained model or train a new one
load_model = True
#store scores for each model
AE = []
L_D = []
L_R = []
ARES_G = []
ARES = []
#load train and test data
train_x, train_y, test_x, test_y = utils.get_data(dataset)
# all available class labels in the dataset
all_classes = np.unique(train_y).tolist()
for i in [0]:
    
            ############# DATA PROCESSING #####################
            
    # include outliers equal to n% of the training set
    n = 0.01 #0.01, 0.05, 0.1, 0.2
    train_idx = np.random.choice(np.argwhere(test_y==1).squeeze(), size = np.int(n*len(train_x)/(1-n)), replace = True)
    test_idx = np.setdiff1d(np.arange(len(test_y)),train_idx)
    train_x = torch.cat((train_x, test_x[train_idx]),0)
    train_y = torch.cat((train_y, test_y[train_idx]),0)    
    test_x = test_x[test_idx]
    test_y = test_y[test_idx]    
    # shuffle
    perm = torch.randperm(len(train_x))
    train_x = train_x[perm]
    train_y = train_y[perm]
    perm = torch.randperm(len(test_x))
    test_x = test_x[perm]
    test_y = test_y[perm]
    # normalize
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    train_x = torch.tensor(train_x, dtype = torch.float32); test_x = torch.tensor(test_x, dtype = torch.float32)
        
    # move to gpu
    train_x = train_x.to(device)
    test_x = test_x.to(device)
    path = "saved_models/%s/%s/%s" %(dataset,data_type,str(n))
        
              ############## MODEL LOADING ##################    
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
   
    density_type = 'KNN'
    n_neighbors = 5
    # LOCAL DENSITY SCORE
    nb = neighbours.NearestNeighbors(algorithm = 'brute', n_neighbors = 100)
    nb.fit(train_enc)
    nb_distance, nb_idx = nb.kneighbors(test_enc, n_neighbors = 100)
    
    if density_type == 'KNN': #n_neighbors = 5,20,40
        local_density_score = np.mean(nb_distance[:,:n_neighbors],axis=1)
    
    if density_type == 'GAUSSIAN': #n_neighbors = 100
        mu, variance = utils.find_gaussians(train_enc, train_y, all_classes)
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