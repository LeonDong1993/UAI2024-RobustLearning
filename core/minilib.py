import numpy as np 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from models.ours.Gaussians import MultivariateGaussain, MixMG

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from typing import List, TypeVar
Tensor = TypeVar('torch.tensor')
MNIST_IMG_SIZE = (28,28)

def transform_to_mnist(img, down_sample = False, normalize = True):
    if len(img.shape) == 3:
        # this is a rgb image, convert to gray scale
        # img_gray = 0.21 * img[:,:,0] + 0.72 * img[:,:,1] + 0.07 * img[:,:,2]
        img_gray = img_rgb.mean(axis=2)
    else:
        img_gray = img
    
    if down_sample:
        scaled_img = cv2.resize(img_gray, [x//2 for x in MNIST_IMG_SIZE])
    else:
        scaled_img = cv2.resize(img_gray, MNIST_IMG_SIZE)
        
    if normalize:
        scaled_img = scaled_img / MAX_VAL
        
    scaled_img = scaled_img.flatten()
    scaled_img = scaled_img.astype('f4')
    return scaled_img

def pca_analysis(data, percentiles):
    D = data.shape[1]
    obj = PCA().fit(data)
    acc_sum = [] ; S = 0
    for v in obj.explained_variance_ratio_:
        S += v
        acc_sum.append(S)
    acc_sum = np.array(acc_sum)

    ret = []
    for p in percentiles:
        left = np.sum(acc_sum > p) - 1
        num_p = D - left
        ret.append(num_p)
    return ret

def gaussian_noise(data, scale, thresh):
    '''
    - add gaussian noise to data
    - assume each row is a sample
    '''
    noise = np.clip(np.random.normal(size = data.shape, scale = scale), -thresh, thresh)
    return data.copy() + noise 

def pixel_jitter(data, n, l, r):
    '''
    - random jitter some part of the input 
    - at most n of them (for each row), if n is float, then at most (100*n) %
    - the juttered value is from range l to r 
    '''
    N, D = data.shape 
    if isinstance(n, float):
        n = max(int(n*D) ,1)
    
    col_ind = np.concatenate([np.random.choice(D, size = n, replace = False) for _ in range(N)], axis = 0)
    row_ind = np.broadcast_to(np.arange(N).reshape(N,1), shape = (N,n) ).flatten()
    
    ret = data.copy()
    ret[row_ind, col_ind] = np.random.uniform(size = row_ind.size) * (r-l) + l
    return ret

# this code is not scalable for dataset with 5k features for example
# need to use svd decomposition to speed up  
def feature_rank(X, k=None):
    # this code is for column, make it support row-wise
    X = X.T               
    M, _ = X.shape

    if k is None:
        k = int(M * 0.8)
    assert(k <= M)

    x_mu = np.mean(X,axis=1).reshape(M,1)
    W = X-x_mu
    A = np.dot(W,W.T)
    w,v = np.linalg.eig(A)
    # pdb.set_trace()

    idx = range(M)
    tmp = sorted(zip(idx,w), key = lambda x:x[1], reverse = True)
    
    eigV = np.zeros((M,k), dtype = 'f4')
    for i in range(k):
        idx = tmp[i][0]
        eigV[:,i] = v[:,idx]

    distProb = []
    for i in range(M):
        row = eigV[i,:]
        prob = np.mean( row ** 2)
        distProb.append(prob)

    rank = np.argsort(distProb)
    return rank, distProb/np.sum(distProb)

def variance_methods(data, num_thresh, mode = 'random', random_seed = 3):
    assert(mode in ['most','unif', 'random'])
    
    _, D = data.shape
    candidates, weight = feature_rank(data)
    num_r, num_N =  num_thresh
    N_cs = max(int(D*num_r+0.49), num_N, 1)
    
    if mode == 'most':
        cond = candidates[-N_cs:]
    if mode == 'unif':
        idx = np.round(np.linspace(0, len(candidates)-1, num = N_cs))
        idx = idx.astype(np.int32)
        cond = candidates[idx]
    if mode == 'random':
        np.random.seed(random_seed)
        idx = np.random.choice(D, size = N_cs,  replace=False, p=weight)
        cond = candidates[idx]
    
    return cond
  
class VAE_Wrapper:
    def __init__(self, epoches = 100, batchsize = 500, 
                 maxlr = 0.01, wd = 1e-5, device = 'cuda:0') -> None:
        self.epoches = epoches
        self.batchsize = batchsize
        self.maxlr = maxlr
        self.device = device
        self.wd = wd

    def fit(self, X, latent_dim = 20, weight_schedule = None):
        # input X is assume to be a 2d numpy array of N, 28*28
        N, D = X.shape
        assert( D == 28 * 28 ), "Not Implemented for general input shape!"
        
        model = VAE(1, latent_dim)
        model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), weight_decay = self.wd)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.maxlr, anneal_strategy='cos', 
                            pct_start=0.25, epochs=self.epoches, steps_per_epoch = 1, verbose = False)
        loss_func = model.loss_function
        
        X = torch.from_numpy(X).to(self.device)
        X = X.view(N,1,28,28)
        train_loader = DataLoader(X, batch_size=self.batchsize, shuffle=True, drop_last=True)
        
        model.train()
        for e in range(self.epoches):
            epoch_loss = 0.0
            
            if weight_schedule:
                kld_w = weight_schedule(e)
            else:
                kld_w = 0.1
            
            for data in train_loader:
                optimizer.zero_grad()
                out = model(data)
                loss = loss_func( *out, M_N = kld_w )['loss']
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            print('Epoch:{} loss:{}'.format(e+1, epoch_loss) ,end = '\r')
        print('')
            
        model.eval()
        self.model = model
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            
        N = X.shape[0]
        X = X.to(self.device, dtype = torch.float32)
        
        with torch.no_grad():
            ret = self.model.encode(X.view(N,1,28,28))[0]
        return ret.cpu().numpy()

    def inverse_transform(self, F):
        if isinstance(F, np.ndarray):
            F = torch.from_numpy(F)
        
        F = F.to(self.device, dtype = torch.float32)
        with torch.no_grad():
            ret = self.model.decode(F)
            ret = torch.flatten(ret, start_dim = 1)
        return ret.cpu().numpy()


class VAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List[int] = None) -> None:
        '''
        expected input image size is 28*28
        '''
        super().__init__()
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size = 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            p = 0 if i == 1 else 1
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size = 3,
                                       stride = 2,
                                       padding = 1,
                                       output_padding = p ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # pdb.set_trace()
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 256, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self, n, device = 'cuda:0'):
        z = torch.randn(n, self.latent_dim)
        return self.decode(z.to(device))

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        layers = []
        layers.append( nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2 ) )
        layers.append( nn.ReLU() )
        layers.append( nn.AvgPool2d(2, stride=2) )
        
        layers.append( nn.Conv2d(8, 64, kernel_size=5, stride=1, padding=2 ) )
        layers.append( nn.ReLU() )
        layers.append( nn.AvgPool2d(2, stride=2) )
        
        layers.append( nn.Flatten() )
        layers.append( nn.Linear(576, 128) )
        layers.append( nn.ReLU() )
        layers.append( nn.Linear(128, 128) )
        layers.append( nn.ReLU() )
        layers.append( nn.Linear(128, 64) )
        layers.append( nn.ReLU() )
        layers.append( nn.Linear(64, 10) )
        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        return self.layers(X)
    
    
class MixMGLearner:
    def __init__(self, n_components = 2, reg_covar = 1e-6, tol = 1e-3, max_iter = 100):
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.tol = tol
        self.max_iter = max_iter
        self.eps = reg_covar
    
    def fit(self, train, weight = None):
        N, D = train.shape
        if weight is None:
            weight = np.ones( shape = (N, ) )
        
        # init the component weights
        self.w = np.ones( shape = (self.n_components, ) ) / self.n_components
        
        # use kmeans to find the initial center 
        clf = KMeans(n_clusters = self.n_components, init='k-means++', random_state = 7).fit(train)
        
        sub_models = []
        for i in range(self.n_components):
            mg = MultivariateGaussain()
            mg.mu = clf.cluster_centers_[i,:]
            mg.S = np.identity(D)
            sub_models.append(mg)
        
        self.mgs = sub_models
        
        # define the Q, Q[i,j] the probility that sample i falls into j component
        self.Q = np.ones(shape = (N, self.n_components)) / self.n_components
        
        # define the V, V[i,j] is the density of sample i under j component
        self.V = np.ones(shape = (N, self.n_components)) / self.n_components
        
        # compute V for the e step
        for i in range(self.n_components):
            masses = self.mgs[i].mass(train, logmode = 1)
            # originally, it is power
            masses = masses * weight.flatten()
            self.V[:,i] = masses
        
        # need to update convergence criteria
        n_iter = 0
        converged = False
        while not converged:
            n_iter += 1
            self._estep()
            self._mstep(train, weight)
            if n_iter >= self.max_iter:
                break
        return self
    
    def _estep(self):
        w = self.w.reshape(1, -1)
        self.Q = (self.V + np.log(w))
        # basically numertical stable softmax here
        self.Q -= np.max(self.Q, axis = 1, keepdims = True)
        self.Q = np.exp(self.Q)
        row_sum = self.Q.sum(axis = 1, keepdims = True)
        self.Q = self.Q / row_sum
        
    def _mstep(self, data, weight):
        # update w
        w = self.Q.mean(axis=0)
        w += self.eps
        w = w/w.sum()
        self.w = w
        
        # update mu 
        weight = weight.reshape(-1, 1)
        wQ = weight * self.Q
        
        Qcol_sum = wQ.sum(axis = 0)
        for i in range(self.n_components):
            self.mgs[i].mu = np.sum( wQ[:, i:i+1] * data, axis = 0) / (Qcol_sum[i] + self.eps)
        
        # update cov matrix 
        for i in range(self.n_components):
            mu = self.mgs[i].mu.reshape(1, -1)
            mat = data - mu
            mat2 = mat.copy()
            mat2 = wQ[:, i:i+1] * mat2 / (self.w[i] * data.shape[0] + self.eps)
            S = mat.T @ mat2
            S += np.identity(mu.size) * self.reg_covar
            self.mgs[i].S = S
        
        # compute V for the e step
        for i in range(self.n_components):
            masses = self.mgs[i].mass(data, logmode = 1)
            # originally, it is power
            masses = masses * weight.flatten()
            self.V[:,i] = masses
        
        
    def get_model(self):
        model = MixMG()
        model.W = self.w
        model.models = self.mgs
        return model

    

# this works for MNIST ALIKE DATASET ONLY
# the input is assumed to be 28*28
class CNN_AE(nn.Module):
    def __init__(self):
        super(CNN_AE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # output shape should be 3*3*16
        
        # Bottleneck
        self.bottle = nn.Conv2d(16,8, kernel_size=3, stride=1, padding=1)
        # output shape is 3*3*8 = 72 dimension
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, X):
        # X should be in shape n,c,h,w
        out = self.encoder(X)
        F = self.bottle(out)
        # this F is of dimension n*8*3*3
        # return dimension is n*72
        return torch.flatten(F, start_dim = 1)
    
    def decode(self, F):
        N = F.shape[0]
        out = F.view(N,8,3,3)
        return self.decoder(out)
    
    def forward(self, x):
        encoded = self.encoder(x)
        bottleneck = self.bottle(encoded)
        decoded = self.decoder(bottleneck)
        return decoded


class CNN_AE_Wrapper:
    def __init__(self, epoches = 100, batchsize = 500, 
                 maxlr = 0.01, wd = 1e-5, device = 'cuda:0') -> None:
        self.epoches = epoches
        self.batchsize = batchsize
        self.maxlr = maxlr
        self.device = device
        self.wd = wd

    def fit(self, X):
        # input X is assume to be a 2d numpy array of N, 28*28
        N, D = X.shape
        assert( D == 28 * 28 ), "Not Implemented for general input shape!"
        
        model = CNN_AE()
        model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), weight_decay = self.wd)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.maxlr, anneal_strategy='cos', 
                            pct_start=0.25, epochs=self.epoches, steps_per_epoch = 1, verbose = False)
        loss_func = nn.MSELoss()
        
        X = torch.from_numpy(X).to(self.device)
        X = X.view(N,1,28,28)
        train_loader = DataLoader(X, batch_size=self.batchsize, shuffle=True, drop_last=True)
        
        model.train()
        for e in range(self.epoches):
            epoch_loss = 0.0
            for data in train_loader:
                optimizer.zero_grad()
                out = model(data)
                loss = loss_func(out, data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            print('Epoch:{} loss:{}'.format(e+1, epoch_loss) ,end = '\r')
        print('')
            
        model.eval()
        self.model = model
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            
        N = X.shape[0]
        X = X.to(self.device)
        
        with torch.no_grad():
            ret = self.model.encode(X.view(N,1,28,28))
        return ret.cpu().numpy()

    def inverse_transform(self, F):
        if isinstance(F, np.ndarray):
            F = torch.from_numpy(F)
        
        F = F.to(self.device)
        with torch.no_grad():
            ret = self.model.decode(F)
        return ret.cpu().numpy()