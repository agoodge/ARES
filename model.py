"""
Autoencoder model and training function
"""


import torch
from torch import nn
import torch.optim as optim
import time
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import variables as var

np.random.seed(0)
torch.manual_seed(0)

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
    
def ae_train(ae,train_x, criterion, val = 0.9):
    
    train, val = train_x[0:int(val*len(train_x))], train_x[int(val*len(train_x)):]
    #dataloaders
    train_loader = torch.utils.data.DataLoader(train,var.batch_size,shuffle = True,drop_last = True)
    #optimiser
    ae_optim = optim.Adam(ae.parameters(), lr = 0.001, weight_decay = 1e-6)
    best_loss = 1000
    counter = 0

    start = time.time()   
    for epoch in range(var.epoch):
        
        # training
        ae.train()
        #total loss for all batches
        running_train_loss = []
        # for each batch
        for data in train_loader:
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
        encoding, reconstruction = ae(val)
        val_loss = criterion(val,reconstruction)
    
        print("Epoch: %d   Train Loss: %.6f     Val Loss %.6f" 
              %(epoch, np.mean(running_train_loss), val_loss.item()))

        if val_loss < best_loss:
            counter = 0
            best_loss = val_loss
        if val_loss > best_loss:
            counter +=1
            if counter == 20:
                print('Early Stopping...')
                break
        
    end = time.time()
    print("Training time: %.2f minutes" %((end-start)/60))
    
    return ae, np.mean(running_train_loss), val_loss.item(), epoch



    
    