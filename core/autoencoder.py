import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np 

class DenseNN(nn.Module):
    def __init__(self, insize, outsize, hsize):
        super(DenseNN, self).__init__()
        layers = []
        layers.append( nn.Linear(insize, hsize[0]) )
        layers.append( nn.ReLU() )
        for i in range(0, len(hsize) -1 ):
            layers.append( nn.Linear(hsize[i], hsize[i+1]) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(hsize[-1], outsize) )
        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        return self.layers(X)
    

class DeepAutoEncoder(nn.Module):
    def __init__(self, insize, outsize, hsize):
        super(DeepAutoEncoder, self).__init__()
        self.encoder = DenseNN(insize, outsize, hsize)
        self.decoder = DenseNN(outsize, insize, list(reversed(hsize)))

    def forward(self, X):
        return self.decoder(self.encoder(X))


class LrateScheduler:
    def __init__(self, lr_max, init_epoches=5, final_epoches=90, init_scale=0.1, final_scale=0.1):
        self.lr_max = lr_max
        self.init_scale = init_scale
        self.init_epoches = init_epoches
        self.final_scale = final_scale
        self.final_epoch = final_epoches
        self.init_lr = lr_max * init_scale
        self.final_lr = lr_max * final_scale
        self.total_epoch = final_epoches + init_epoches

    def get_lrate(self,epoch):
        # linear warmup followed by cosine decay
        if epoch < self.init_epoches:
            lr = (self.lr_max - self.init_lr) * float(epoch) / self.init_epoches + self.init_lr
        elif epoch < self.total_epoch:
            lr = (self.lr_max - self.final_lr)*max(0.0, np.cos(((float(epoch) -
                    self.init_epoches)/(self.final_epoch - 1.0))*(np.pi/2.0))) + self.final_lr
        else:
            lr = self.final_lr
        return lr
    
    
class AutoEncoder:
    def __init__(self, n_components, hsize, epoches = 150, batchsize = 500, maxlr = 0.01, wd = 1e-5, device = 'cuda:0') -> None:
        self.k = n_components
        self.hs = hsize
        self.epoches = epoches
        self.batchsize = batchsize
        self.maxlr = maxlr
        self.device = device
        self.wd = wd

    def fit(self, X):
        D = X.shape[1]
        model = DeepAutoEncoder(D, self.k, self.hs)
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-08, weight_decay = self.wd)
        scheduler = LrateScheduler(self.maxlr, init_epoches=10, final_epoches = (self.epoches-15) )
        loss_func = nn.MSELoss()

        model.to(self.device)
        X = torch.from_numpy(X).to(self.device)
        train_loader = torch.utils.data.DataLoader(X, batch_size=self.batchsize, shuffle=True, drop_last=True)
        model.train()

        for e in range(self.epoches):
            lrate = scheduler.get_lrate(e)
            for g in optimizer.param_groups:
                g['lr'] = lrate
            
            epoch_loss = 0.0
            for X in train_loader:
                optimizer.zero_grad()
                out = model(X)
                loss = loss_func(out,X)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print('Epoch:{} loss:{}'.format(e+1, epoch_loss) ,end = '\r')
        print('')
            
        model.eval()
        model.to('cpu')
        self.model = model
        return self

    def __repr__(self) -> str:
        desc = 'AE-{}'.format(self.k)
        return desc

    def transform(self, X):
        with torch.no_grad():
            X = torch.from_numpy(X).to('cpu')
            F = self.model.encoder(X)
        return F

    def inverse_transform(self, F):
        with torch.no_grad():
            F = F.to('cpu')
            X = self.model.decoder(F)
        return X

