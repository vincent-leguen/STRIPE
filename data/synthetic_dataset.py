import numpy as np
import random
import torch

def create_synthetic_dataset_multimodal(N, N_input,N_output,sigma,nfutures=10):
    # N: number of samples in each split (train, test)
    # N_input: import of time steps in input series
    # N_output: import of time steps in output series
    # sigma: standard deviation of additional noise
    X = []
    for k in range(2*N): 
        i1 = random.randint(1,10)
        i2 = random.randint(10,18)
        j1 = random.random() # first peak amplitude
        j2 = random.random() # second peak amplitude
        
        for s in range(nfutures):
            serie = np.array([ sigma*random.random() for i in range(N_input+N_output)])
            interval = abs(i2-i1) + random.randint(-2,2)
            serie[i1:i1+1] += j1
            serie[i2:i2+1] += j2
            
            if (N_input+interval <= N_input):
                interval=1 
            serie[N_input+interval:] += (j2-j1) + ( (random.random()-0.5)/10  )
            X.append(serie)
    X = np.stack(X)
    return X[0:N*nfutures,0:N_input], X[0:N*nfutures, N_input:N_input+N_output], X[N*nfutures:2*N*nfutures,0:N_input], X[N*nfutures:2*N*nfutures, N_input:N_input+N_output]


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, X_input, X_target):
        super(SyntheticDataset, self).__init__()  
        self.X_input = X_input
        self.X_target = X_target
        
    def __len__(self):
        return (self.X_input).shape[0]

    def __getitem__(self, idx):
        return (self.X_input[idx,:,np.newaxis], self.X_target[idx,:,np.newaxis])
        