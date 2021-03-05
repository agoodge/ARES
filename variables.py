"""
hyperparameter settings
"""


from torch import nn

############################# model variables #################################

batch_size = 250
epoch = 350
criterion = nn.MSELoss()

def get_hidden(dataset):

    if dataset in ['MNIST','FMNIST']:
        hidden_size = [[784, 600, 500, 400, 300, 200, 100, 20],
                     [20, 100, 200, 300, 400, 500, 600, 784]]
        
    if dataset == 'OTTO':
        hidden_size = [[93, 88, 84, 79, 74, 70, 65],
                       [65, 70, 74, 79, 84, 88, 93]]
            
    if dataset == 'STL':
        hidden_size = [[27, 24, 21, 18, 14, 11, 8],
                       [8, 11, 14, 18, 21, 24, 27]]
        
    if dataset == 'SNSR':
        hidden_size = [[48, 43, 37, 32, 27, 21, 16],
                       [16, 21, 27, 32, 37, 43, 48]]

    if dataset == 'EOPT':
        hidden_size = [[20, 18, 16, 14, 12, 10, 8],
                      [8, 10, 12, 14, 16, 18, 20]]
        
    if dataset in ['MI-V', 'MI-F']:
        hidden_size = [[58, 52, 47, 41, 35, 30, 24],
                       [24, 30, 35, 41, 47, 52, 58]]
    
    return hidden_size