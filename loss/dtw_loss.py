import torch
from . import soft_dtw
from . import path_soft_dtw 

def dtw_loss(outputs, targets, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		D[k:k+1,:,:] = Dk     
	loss_shape = softdtw_batch(D,gamma)
	return loss_shape