import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import warnings; warnings.simplefilter('ignore')
from data.synthetic_dataset import create_synthetic_dataset_multimodal, SyntheticDataset
from models.models import cVAE, STRIPE, STRIPE_conditional, TestSampler_Sequential
from trainer.trainer import train_model, train_STRIPE, eval_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)


batch_size = 100
N = 100
N_input = 20
N_output = 20  
sigma = 0.01
gamma = 0.01

X_train_input,X_train_target,X_test_input,X_test_target = create_synthetic_dataset_multimodal(N,N_input,N_output,sigma)
dataset_train = SyntheticDataset(X_train_input,X_train_target)
dataset_test  = SyntheticDataset(X_test_input,X_test_target)
trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=0)
testloader  = DataLoader(dataset_test, batch_size=10,shuffle=False, num_workers=0)


input_size = 1
rnn_units = 128
nlayers = 1
bidirectional = False
latent_dim = 16
fc_units = 10

### Step 1: train the predictive model, here a cVAE
print('STEP 1')
model_dilate = cVAE(input_size,rnn_units,nlayers,bidirectional,latent_dim,fc_units,N_output,device).to(device)
train_model(model_dilate,trainloader,testloader,loss_type='dilate',nsamples=10,learning_rate=0.001, 
    device=device,epochs=1, gamma=gamma, alpha=0.5, print_every=50, eval_every=100,verbose=1)
           
#torch.save(model_dilate.state_dict(),'save/model_dilate.pth') 
             

### Step 2: train STRIPE-shape
print('STEP 2')
nshapes = 10
stripe_shape = STRIPE('shape',nshapes, latent_dim, N_output, rnn_units).to(device)
train_STRIPE(cvae=model_dilate, stripe=stripe_shape, trainloader=trainloader, testloader=testloader, device=device, mode_stripe='shape',
    nsamples=nshapes, quality='', diversity_kernel='dtw',  learning_rate=0.001, epochs=16, print_every=2,eval_every=5, alpha=0.5)      
#torch.save(stripe_shape.state_dict(),'save/stripe_shape.pth')     
   
   
### Step 3: train STRIPE-time 
print('STEP 3')           
ntimes = 10
stripe_time = STRIPE_conditional('time',ntimes, latent_dim, N_output, rnn_units).to(device)
train_STRIPE(cvae=model_dilate,stripe=stripe_time, trainloader=trainloader, testloader=testloader, device=device, mode_stripe='time',nsamples=ntimes, 
        quality='',diversity_kernel='tdi', learning_rate=0.001, epochs=1, print_every=16,eval_every=5, alpha=0.5)
#torch.save(stripe_time.state_dict(),'save/stripe_time.pth')                  


### Step 4: evaluation with the sequential sampling scheme
print('STEP 4')
test_sampler = TestSampler_Sequential(model_dilate, stripe_shape, stripe_time)
_,_ = eval_model(test_sampler, testloader,nsamples=10, device=device, gamma=0.01,mode='test_sampler')


print('FINISH !!!')
