import torch
import numpy as np
from loss.dilate_loss import dilate_loss
from loss.losses import loss_kullback_leibler, diversity_loss, dilate_eval
from torch.optim.lr_scheduler import ReduceLROnPlateau
import properscoring as ps
import time
import random

def train_model(net,trainloader,testloader,loss_type,nsamples,learning_rate, device, epochs=1000, gamma = 0.01, alpha=0.5,
                print_every=50,eval_every=50, verbose=1):

    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3,factor=0.1,verbose=True)
    criterion = torch.nn.MSELoss()
    epoch_list = []
    mse_list,dtw_list,tdi_list,dilate_list,best_mse_list,best_dtw_list,best_tdi_list,best_dilate_list,card_shape_list,card_time_list=[],[],[],[],[],[],[],[],[],[]
    
    best_mse = float('inf')
    for epoch in range(epochs): 
        net.train()
        t0 = time.time()
        
        for i, data in enumerate(trainloader, 0):
            inputs, target = data
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            batch_size, N_output = target.shape[0:2]                     

            outputs, z_mu, z_logvar = net(inputs, target) # outputs [batch, seq_len, nfeatures]
            loss_recons, loss_KL = 0,0
            if (loss_type=='mse'):
                loss_mse = criterion(target,outputs)
                loss_recons = loss_mse                   

            if (loss_type=='dilate'):    
                loss_dilate, loss_shape, loss_temporal = dilate_loss(target,outputs,alpha, gamma, device)             
                loss_recons = loss_dilate
            
            loss_KL = loss_kullback_leibler(z_mu, z_logvar) # Kullback-Leibler
            loss = loss_recons + loss_KL          
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()          

        if(verbose):
            if (epoch % print_every == 0):
                print('ep ', epoch, ' loss ',loss.item(), ' rec ',loss_recons.item(),' KL=',loss_KL.item(),' time ',time.time()-t0)
            if (epoch % eval_every == 0):    
                mse,dilate = eval_model(net,testloader,nsamples,device,gamma,'vae')
                
                scheduler.step(dilate)               
                mse_list.append(mse)
                dilate_list.append(dilate)
                epoch_list.append(epoch)                
    return 0


      
def eval_model(net,loader,nsamples, device,gamma=0.01,mode='vae',mode_stripe='shape',stripe=None,verbose=1,nsamples_ds=10): 
    # testloader a nfutures possibles pour chaque input
    # prendre la batch_size egale à nfuture
    criterion = torch.nn.MSELoss()
    losses_mse = []
    losses_dtw = []
    losses_tdi = []   
    losses_dilate = []
    recall_mse,recall_dtw,recall_tdi,recall_dilate,precision_mse,precision_dtw,precision_tdi,precision_dilate=[],[],[],[],[],[],[],[]
    CRPS = 0
    
    for i, data in enumerate(loader, 0):  # pour un batch, c'est le même input et nfutures differents futurs
        loss_mse, loss_dtw, loss_tdi = torch.tensor(0),torch.tensor(0),torch.tensor(0)
        inputs, target = data
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)  # [nfutures, seq_len, nfeatures] 
        target = torch.tensor(target, dtype=torch.float32).to(device)  # [nfutures, seq_len, nfeatures]
        nfutures, N_output, nfeatures = target.shape
        batch_size = nfutures 
                
        if (mode=='stripe'):
            output_phi, h = net.encoder.rnn_phi(inputs) # in stripe mode, 'net' is the cvae  
            sampled_z = stripe(h) #  sampled_z : [batch_size, latent_dim * nsamples]

            z_fixed = torch.randn((batch_size, stripe.half_latent_dim*nsamples), dtype=torch.float32).to(device)
            outputs = torch.zeros([batch_size, nsamples, N_output, nfeatures]  ).to(device)
            for k in range(0,nsamples):                    
                z = sampled_z[:,stripe.half_latent_dim*k:stripe.half_latent_dim*(k+1)] # [batch_size, half_latent_dim]
                z_f =  z_fixed[:,stripe.half_latent_dim*k:stripe.half_latent_dim*(k+1)]
                if (mode_stripe=='shape'):
                    z = torch.cat( (z, z_f), dim=1)
                else: # mode = time
                    z = torch.cat( (z_f, z), dim=1)   
                
                x_mu = net.decoder(inputs, h, z) # [batch_size, target_length, nfeatures]    
                outputs[:,k,:,:] = x_mu      # outputs [nfutures, nsamples, seq_len, nfeatures]      
            # just keep outputs[0,:,:,:] corresponding to the first input
            outputs = outputs[0,:,:,:]
         
            
        if (mode=='vae'): # mode cVAE, il faut sampler plusieurs predictions
            outputs = torch.zeros([nsamples, N_output, 1]  ).to(device)
            for k in range(nsamples):
                output_k = net.sample(inputs[0:1,:,:], target[0:1,:,:]) # output_k [1, seq_len, nfeatures]
                outputs[k,:,:] = output_k[0,:,:]
                
        if (mode=='test_sampler'): # mode diverse sampler
            outputs = net(inputs).to(device)
            indd = random.sample(range(0,100),nsamples_ds)
            #indd = range(0,10)
            outputs = outputs[indd ,:,:]   

        ### CRPS
        target_cpu = target.detach().cpu().numpy()
        output_cpu = outputs.detach().cpu().numpy()
        for a in range(N_output):
            CRPS += ps.crps_ensemble( target_cpu[0,a,0] , output_cpu[:,a,0])        
        
        
        ### RECALL
        recall_mse_i, recall_dtw_i, recall_tdi_i, recall_dilate_i = 0,0,0,0
        for future in range(nfutures):
            min_mse,min_dtw,min_tdi,min_dilate = float('inf'),float('inf'),float('inf'),float('inf')
            for ind in range(nsamples):
                loss_mse = criterion(target[future,:,:],outputs[ind,:,:]).detach().cpu().numpy()        
                loss_dtw, loss_tdi = dilate_eval(target[future,:,:],outputs[ind,:,:])
                loss_dilate = loss_dtw + loss_tdi          
                min_mse = np.minimum(min_mse , loss_mse)
                min_dtw = np.minimum(min_dtw, loss_dtw)
                min_tdi = np.minimum(min_tdi, loss_tdi)
                min_dilate = np.minimum(min_dilate, loss_dilate)            
            recall_mse_i += min_mse    
            recall_dtw_i += min_dtw   
            recall_tdi_i += min_tdi   
            recall_dilate_i += min_dilate           
        recall_mse.append( recall_mse_i / nfutures )
        recall_dtw.append( recall_dtw_i / nfutures )
        recall_tdi.append( recall_tdi_i / nfutures )
        recall_dilate.append( recall_dilate_i / nfutures )
        
        ### PRECISION
        precision_mse_i, precision_dtw_i, precision_tdi_i, precision_dilate_i = 0,0,0,0
        for ind in range(nsamples):
            min_mse,min_dtw,min_tdi,min_dilate = float('inf'),float('inf'),float('inf'),float('inf')
            for future in range(nfutures):
                loss_mse = criterion(target[future,:,:],outputs[ind,:,:]).detach().cpu().numpy()
                loss_dtw, loss_tdi = dilate_eval(target[future,:,:],outputs[ind,:,:])
                loss_dilate = loss_dtw + loss_tdi   
                
                # Global scores
                losses_mse.append(loss_mse.item())
                losses_dtw.append(loss_dtw)
                losses_tdi.append(loss_tdi)
                losses_dilate.append(loss_dilate)
                
                min_mse = np.minimum(min_mse , loss_mse)
                min_dtw = np.minimum(min_dtw, loss_dtw)
                min_tdi = np.minimum(min_tdi, loss_tdi)
                min_dilate = np.minimum(min_dilate, loss_dilate)                  
            precision_mse_i += min_mse    
            precision_dtw_i += min_dtw   
            precision_tdi_i += min_tdi   
            precision_dilate_i += min_dilate  
        precision_mse.append( precision_mse_i / nsamples )
        precision_dtw.append( precision_dtw_i / nsamples )
        precision_tdi.append( precision_tdi_i / nsamples )
        precision_dilate.append( precision_dilate_i / nsamples )
        
    mse = np.array(losses_mse).mean()
    DTW = np.array(losses_dtw).mean()
    tdi = np.array(losses_tdi).mean()
    dilate = np.array(losses_dilate).mean()
    recall_mse = np.array(recall_mse).mean()
    recall_dtw = np.array(recall_dtw).mean()
    recall_tdi = np.array(recall_tdi).mean()
    recall_dilate = np.array(recall_dilate).mean()    
    precision_mse = np.array(precision_mse).mean()
    precision_dtw = np.array(precision_dtw).mean()
    precision_tdi = np.array(precision_tdi).mean()
    precision_dilate = np.array(precision_dilate).mean()   
    
    print( '--- Eval mse= ', mse ,' dtw= ',DTW ,' tdi= ', tdi, ' dilate ' , dilate) 
    print( '- R_mse= ', recall_mse ,' R_dtw= ',recall_dtw,' R_tdi= ', recall_tdi,' R_di= ',recall_dilate) 
    print( '- P_mse= ', precision_mse ,' P_dtw= ',precision_dtw,' P_tdi= ', precision_tdi,' P_di= ',precision_dilate) 
    print('-----------  CRPS = ', CRPS/(N_output*len(loader)))
    return mse,dilate


def train_STRIPE(cvae, stripe, trainloader, testloader, device, mode_stripe, nsamples,quality,diversity_kernel, learning_rate, epochs=1000, gamma=0.01,
                print_every=50,eval_every=50, alpha=0.5, ldtw=10, verbose=1):
    device = cvae.device
    optimizer = torch.optim.Adam(stripe.parameters(),lr=learning_rate)
    epoch_list = []
    mse_list,dtw_list,tdi_list,best_mse_list,best_dtw_list,best_tdi_list=[],[],[],[],[],[]

    for epoch in range(epochs): 
        stripe.train()
        t0 = time.time()       
        for i, data in enumerate(trainloader, 0):
            inputs, target = data
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device) # [nfutures, seq_len, nfeatures] 
            target = torch.tensor(target, dtype=torch.float32).to(device)
            batch_size, N_output, nfeatures = target.shape    
             
            output_phi, h = cvae.encoder.rnn_phi(inputs) # h: last latent state
            #h_penultimate = output_phi[:,-2,:] # for the decoder
            ###### STRIPE PROPOSAL: h -> z
            sampled_z = stripe(h) # sampled_z : [batch_size, latent_dim * nsamples]

            ###### DECODING:  concatenate z with zfixed and decode
            z_fixed = torch.randn((batch_size, stripe.half_latent_dim*nsamples), dtype=torch.float32).to(device)
            outputs = torch.zeros([batch_size, nsamples, N_output, nfeatures]  ).to(device)
            for k in range(0,nsamples):                    
                z = sampled_z[:,stripe.half_latent_dim*k:stripe.half_latent_dim*(k+1)] # [batch_size, half_latent_dim]
                z_f =  z_fixed[:,stripe.half_latent_dim*k:stripe.half_latent_dim*(k+1)]
                if (mode_stripe=='shape'):
                    #z = torch.cat( (z, z_f), dim=1)
                    z = torch.cat( (z_f, z), dim=1)
                else: # mode = time
                    z = torch.cat( (z_f, z), dim=1)   
                
                x_mu = cvae.decoder(inputs, h, z) # [batch_size, target_length, nfeatures]    
                outputs[:,k,:,:] = x_mu
                                
            dpp_loss = diversity_loss(outputs,target,quality,diversity_kernel,alpha,gamma,ldtw,device)
            loss = dpp_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

        if(verbose):
            if (epoch % print_every == 0):
                print('ep ', epoch, ' dpp_loss ',dpp_loss.item(), ' time ',time.time()-t0)
            if (epoch % eval_every == 0):    
                _,_ = eval_model(cvae,testloader,nsamples,device,gamma,'stripe',mode_stripe=mode_stripe,stripe=stripe)
               
    return 0
