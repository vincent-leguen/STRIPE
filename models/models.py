import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_encoder(nn.Module):
    def __init__(self, input_size, rnn_units, nlayers, bidirectional,latent_dim, device):
        super(VAE_encoder, self).__init__()
        # input_shape: 1 for univariate time series
        # rnn_units: nb of units of the RNN (e.g. 128)
        # nlayers: nb of layers of the RNN
        # latent_dim: dimension of latent vector z (=latent_dim_zs + latent_dim_zt)
        self.n_directions = 2 if bidirectional==True else 1
        self.device = device
        
        self.rnn_phi = nn.GRU(input_size=input_size, hidden_size=rnn_units, num_layers=nlayers,
            bidirectional=bidirectional, batch_first=True)
            
        self.rnn_x = nn.GRU(input_size=input_size, hidden_size=rnn_units, num_layers=nlayers,
            bidirectional=bidirectional, batch_first=True)

        self.flat_dim = self.n_directions * rnn_units * 2
        self.encoder_mu = nn.Sequential(
            #nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, latent_dim)
        )
        self.encoder_logvar = nn.Sequential(
           # nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, latent_dim)
        )
   
    def forward(self, phi, x): # x [batch, seq_len, input_dim] 
        output_phi, hidden_phi = self.rnn_phi(phi) 
        # output_phi [batch, seq_len, rnn_units*n_directions], hidden_phi [nlayers*n_directions, batch, rnn_units]       
        output_x, hidden_x  = self.rnn_x(x) # hidden_x [nlayers*n_directions, batch, rnn_units]   
        features = torch.cat((hidden_phi[0], hidden_x[0]), dim=1) # [batch, 2*rnn_units*n_directions]
        z_mu = self.encoder_mu(features)
        z_logvar = self.encoder_logvar(features)
        h = output_phi[:,-2,:] # penultimate latent state for init the decoder [batch,rnn_units*n_directions]
        #print('h ',h.shape)
        return z_mu, z_logvar , h
    

class VAE_decoder(nn.Module):
    def __init__(self, input_size, rnn_units, fc_units, nlayers, latent_dim, target_length, device):
        super(VAE_decoder, self).__init__()
        # input_shape: 1 for univariate time series
        # rnn_units: nb of units of the RNN (e.g. 128)
        # nlayers: nb of layers of the RNN
        self.target_length = target_length
        self.device = device
              
        self.rnn_prediction = nn.GRU(input_size=input_size, hidden_size=rnn_units+latent_dim, num_layers=nlayers,
            bidirectional=False, batch_first=True)
        self.fc = nn.Linear(rnn_units+latent_dim, fc_units)
        self.output_layer = nn.Linear(fc_units, input_size)  # output_size = input_size           
        
    def forward(self, phi, h, z): # phi [batch, seq_len, input_dim]    z [batch_size, latent_dim]
        decoder_input = phi[:,-1,:].unsqueeze(1) # first decoder input = last element of input sequence
        
        if(len(h.shape) == 2):  # nb of channels
            decoder_hidden = torch.cat( (h.unsqueeze(0) , z.unsqueeze(0)), dim=2 ) # [nlayers*n_directions, batch, rnn_units]
        else:
            decoder_hidden = torch.cat( (h , z.unsqueeze(0)), dim=2 )   
        
        outputs = torch.zeros([phi.shape[0], self.target_length, phi.shape[2]]  ).to(self.device)
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.rnn_prediction(decoder_input, decoder_hidden)
            output = F.relu( self.fc(decoder_output) )
            output = self.output_layer(output)   
            decoder_input = output # no teacher forcing            
            outputs[:,di:di+1,:] = output           
        return outputs
    
    
class cVAE(nn.Module):
    def __init__(self, input_size, rnn_units, nlayers, bidirectional, 
                latent_dim, fc_units, target_length, device):
        super(cVAE, self).__init__()
        self.input_size = input_size # 1 for univariate time series
        self.rnn_units = rnn_units
        self.latent_dim = latent_dim # z dimension 
        self.target_length = target_length
        self.device = device
        self.encoder = VAE_encoder(input_size,rnn_units, nlayers, bidirectional, latent_dim, device) 
        self.decoder = VAE_decoder(input_size,rnn_units, fc_units, nlayers, latent_dim, target_length, device)
    
    def reparameterize(self, mu, logvar): 
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, phi, x):
        z_mu, z_logvar, h = self.encoder(phi, x)  # x [batch_size, target_length, input_size]
        z = self.reparameterize(z_mu, z_logvar)
        x_mu = self.decoder(phi, h, z)
        return x_mu, z_mu, z_logvar
    
    def sample(self, phi, x):
        with torch.no_grad():
            _, _, hidden_init = self.encoder(phi, x)
            z = torch.randn(1, self.latent_dim, device=self.device) # one by one in test mode
            x_mu = self.decoder(phi, hidden_init, z)
            return x_mu
            
        
        
class STRIPE(nn.Module):
    def __init__(self, mode, nsamples, latent_dim, target_length, rnn_units):
        super(STRIPE, self).__init__()        
        self.mode = mode # 'shape' or 'time'
        self.nsamples = nsamples
        self.target_length = target_length
        self.half_latent_dim = latent_dim//2
        
        self.MLP =  nn.Sequential(
            nn.BatchNorm1d(rnn_units),
            nn.Linear(rnn_units, 512),        
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),            
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.half_latent_dim*self.nsamples) # generate half of the latent state
        )
    
    def forward(self, h): # phi [batch_size,seq_length,input_size]
        # h = hidden_phi (h in paper) [nlayers*ndirections, batch_size, rnn_units]       
        sampled_z = self.MLP(h[0,:,:]) # sampled_z : [batch_size, half_latent_dim * nsamples]
        sampled_z = torch.nn.functional.tanh(sampled_z)*3
        return sampled_z  


class STRIPE_conditional(nn.Module):
    def __init__(self, mode, nsamples, latent_dim, target_length, rnn_units):
        super(STRIPE_conditional, self).__init__()        
        self.mode = mode # 'shape' or 'time'
        self.nsamples = nsamples
        self.target_length = target_length
        self.half_latent_dim = latent_dim//2
        
        self.MLP =  nn.Sequential(
            nn.BatchNorm1d(rnn_units+self.half_latent_dim),
            nn.Linear(rnn_units+self.half_latent_dim, 512),        
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),            
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.half_latent_dim*self.nsamples) # generate half of the latent state
        )
    
    def forward(self, h, zs=None): # phi [batch_size,seq_length,input_size], zs [batch_size, half_latent_dim]
        batch_size = h.shape[1]
        # h = hidden_phi (h in paper) [nlayers*ndirections, batch_size, rnn_units]
        if (zs==None): # None only when training and evaluating STRIPE-time, not in the sequential sampling scheme
            zs = torch.randn((batch_size, self.half_latent_dim), dtype=torch.float32).to(h.device)    
            
        h = h[0,:,:]
        temp = torch.cat( (h, zs), dim=1)
        sampled_z = self.MLP(temp) # sampled_z : [batch_size, half_latent_dim * nsamples]
        sampled_z = torch.nn.functional.tanh(sampled_z)*3
        return sampled_z  
    
    

class TestSampler_Sequential(nn.Module):
    def __init__(self, cvae, stripe_shape, stripe_time):
        super(TestSampler_Sequential, self).__init__()
        self.cvae = cvae
        self.stripe_shape = stripe_shape
        self.stripe_time  = stripe_time
        self.half_latent_dim = stripe_shape.half_latent_dim
        self.nshapes = stripe_shape.nsamples
        self.ntimes = stripe_time.nsamples
    
    def forward(self, inputs): # phi [batch_size,seq_length,input_size]
        self.stripe_shape.eval()
        self.stripe_time.eval()
        batch_size = inputs.shape[0] # 10 car il y a 10 traj en test
        
        output_phi, h = self.cvae.encoder.rnn_phi(inputs) # h: last latent state
        outputs = torch.zeros((self.nshapes*self.ntimes,self.cvae.target_length, self.cvae.input_size) ) 
          
        sampled_zs = self.stripe_shape(h) # sampled_zs: [batch_size, half_latent_dim * nshapes]    
        for k in range(0,self.nshapes):
            zs_k = sampled_zs[:,self.half_latent_dim*k:self.half_latent_dim*(k+1)] # [batch_size, half_latent_dim]
            sampled_zt = self.stripe_time(h,zs_k)  # sampled_zt: [batch_size, half_latent_dim * ntimes]   
            
            for l in range(0,self.ntimes):
                zt_kl = sampled_zt[:,self.half_latent_dim*l:self.half_latent_dim*(l+1)] # [batch_size, half_latent_dim]
                z = torch.cat( (zs_k, zt_kl), dim=1)         
                x_mu = self.cvae.decoder(inputs, h, z) # x_mu: [batch_size, target_length, nfeatures]    
                x_mu = x_mu[0,:,:] # keep only the trajectory corresponding to the first input (all inputs equal in test in this batch)
                outputs[k*self.nshapes+l,:,:] = x_mu               
        return outputs #[nshapes*ntimes,seq_len,nfeatures]       
