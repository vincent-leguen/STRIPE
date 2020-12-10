import torch
import tslearn
from tslearn.metrics import dtw, dtw_path
from loss.dtw_loss import dtw_loss
from loss.dilate_loss import dilate_loss

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_kullback_leibler( mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def true_dilate(target, pred, alpha):  # target, pred [seq_length]
    N_output = target.shape[0]
    loss_dtw = dtw(target,pred)
    path, sim = dtw_path(target, pred)   
    Dist = 0
    for ii,jj in path:
            Dist += (ii-jj)*(ii-jj)
    loss_tdi = Dist / (N_output*N_output)            
    loss_dilate = alpha*loss_dtw + (1-alpha)*loss_tdi
    return loss_dtw, loss_tdi, loss_dilate


def dilate_eval(target, output):
    N_output = target.shape[0]
    target = target.detach().cpu().numpy()
    output = output.detach().cpu().numpy()    
    loss_dtw = tslearn.metrics.dtw(target,output)
    path, sim = tslearn.metrics.dtw_path(target, output)   
    Dist = 0
    for ii,jj in path:
            Dist += (ii-jj)*(ii-jj)
    loss_tdi = Dist / (N_output*N_output)  
    return loss_dtw, loss_tdi


def diversity_loss(predictions,target, quality, diversity_kernel, alpha, gamma,ldtw, device):
    # predictions [batch_size, nsamples, seq_len, nfeatures]
    # target [batch_size, seq_len, nfeatures]
    criterion = torch.nn.MSELoss()
    nsamples = predictions.shape[1]

    S = torch.zeros((nsamples,nsamples)).to(device) # similarity matrix
    for i in range(0,nsamples):
        for j in range(0,nsamples):
            if i<=j :    
                if diversity_kernel == 'dtw':
                    dtw = dtw_loss(predictions[:,i,:,:],predictions[:,j,:,:], gamma, device) 
                    S[i,j] = dtw    
                    
                if diversity_kernel == 'tdi':
                    dilate, shape, tdi = dilate_loss(predictions[:,i,:,:],predictions[:,j,:,:],alpha,gamma,device)
                    S[i,j] = tdi
                                                            
                if diversity_kernel == 'mse':
                    S[i,j] = criterion(predictions[:,i,:,:],predictions[:,j,:,:]) 
            S[j,i] = S[i,j] # matrix symmetrization

    # Kernel computation:
    S_mean = torch.mean(S)
    if diversity_kernel == 'dtw':
        Lambda = 1 /torch.abs(S_mean)
        K = torch.exp(-Lambda * S)
    
    if diversity_kernel == 'tdi':
        Lambda = 1 /torch.abs(S_mean)
        K = torch.exp(10* Lambda * S) 
        
    if diversity_kernel == 'mse':
        Lambda = S_mean        
        K = torch.exp(-Lambda * S)

    I = torch.eye((nsamples)).to(device)
    M = I - torch.inverse(K+I)
    dpp_loss = - torch.trace(M) 
    return dpp_loss