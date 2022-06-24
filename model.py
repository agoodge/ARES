"""
Autoencoder class, training method and ARES testing method
"""
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import sklearn.neighbors as neighbours

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class AE(nn.Module):
    def __init__(self,enc_hidden,dec_hidden):
        
        super(AE,self).__init__()
        
        #encoder
        self.enc_list = []
        for i in range(1,len(enc_hidden)):
            self.enc_list.append(nn.Linear(enc_hidden[i-1],enc_hidden[i]))
            self.enc_list.append(nn.LeakyReLU())
            self.enc_list.append(nn.BatchNorm1d(enc_hidden[i]))
        self.enc_list.pop()
        self.enc_list.pop()
        self.enc_list = nn.ModuleList(self.enc_list)
        
        #decoder
        self.dec_list = [nn.LeakyReLU(), nn.BatchNorm1d(enc_hidden[-1])]
        for i in range(1,len(dec_hidden)):
            self.dec_list.append(nn.Linear(dec_hidden[i-1],dec_hidden[i]))
            self.dec_list.append(nn.LeakyReLU())
            self.dec_list.append(nn.BatchNorm1d(dec_hidden[i]))
        self.dec_list.pop()
        self.dec_list.pop()
        self.dec_list = nn.ModuleList(self.dec_list)
        
    def forward(self,x):
        
        for f in self.enc_list:
            x = f(x)
           
        encoding = x
        
        for f in self.dec_list:
            x = f(x)
            
        reconstruction = x
        
        return encoding, reconstruction
    
def train_model(ae,train_x,val_x,args,device):

    criterion = nn.MSELoss()
    
    #dataloaders
    train_loader = torch.utils.data.DataLoader(train_x,args.batch_size,drop_last = True)
    #optimiser
    ae_optim = optim.Adam(ae.parameters(), lr = args.lr, weight_decay = args.wd)
    best_loss = 1000
    counter = 0
 
    for epoch in range(args.n_epochs):
        
        # training
        ae.train()
        #total loss for all batches
        running_train_loss = []
        # for each batch
        for data in train_loader:
            data = data.to(device)
            #adjust from batch*28x28 to batchx784
            encoding, reconstruction = ae(data)
            # Computes loss
            train_loss = criterion(data,reconstruction)
            running_train_loss.append(train_loss.item())
            #back prop
            train_loss.backward()
            #update parameters
            ae_optim.step()
            #zero gradient 
            ae.zero_grad()
        
        # validation
        ae.eval()
        encoding, reconstruction = ae(val_x.to(device))
        val_loss = criterion(val_x.to(device),reconstruction)
    
        if args.verbose == 1:
            print("Epoch: %d \t Train Loss: %.6f \t Val Loss %.6f" %(epoch, np.mean(running_train_loss), val_loss.item()))

        if val_loss < best_loss:
            counter = 0
            best_loss = val_loss
        else:
            counter +=1
            if counter == 20:
                if args.verbose == 1:
                    print('Early Stopping...')
                break
            
    return ae, np.mean(running_train_loss), val_loss.item(), epoch


def test_model(ae, train_x, test_x, args, device):     
   
    ae.eval()
    with torch.no_grad():
        train_enc, train_rec = ae(train_x.to(device))
        test_enc, test_rec = ae(test_x.to(device))

    train_x = train_x.numpy()
    test_x = test_x.numpy()
    train_enc = train_enc.cpu().detach().numpy()
    train_rec = train_rec.cpu().detach().numpy()
    test_enc = test_enc.cpu().detach().numpy()
    test_rec = test_rec.cpu().detach().numpy()

    train_error = ((train_rec - train_x)**2).mean(axis=1)
    test_error = ((test_rec - test_x)**2).mean(axis=1)

    nb = neighbours.LocalOutlierFactor(n_neighbors = args.n_neighbours,novelty=True)
    nb.fit(train_enc)

    nb_distance, nb_idx = nb.kneighbors(test_enc, n_neighbors = args.n_neighbours)

    # LOCAL DENSITY SCORE
    local_density_score = -nb.decision_function(test_enc)

    # LOCAL RECONSTRUCTION SCORE
    local_reconstruction_score = test_error - np.median(train_error[nb_idx],axis = 1)
    
    # ARES SCORE
    score = local_reconstruction_score + args.alpha*local_density_score
        
    return score
    