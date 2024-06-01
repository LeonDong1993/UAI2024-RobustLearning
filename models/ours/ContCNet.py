import torch
import pdb
import numpy as np
from utmLib import utils
import multiprocessing as mp



class ContCNet:
    def __init__(self, px, py, xids, yids, sample_size = 100):
        # py should be a NNGBN object
        self.px = px
        self.py = py
        self.xids = xids
        self.yids = yids
        self.sample_size = sample_size

    def mass(self, Z, logmode = True):
        # compute the density of samples, can be single sample as well
        if len(Z.shape) == 1:
            Z = Z.reshape(1, -1)

        X = Z[:, self.xids]
        Y = Z[:, self.yids]

        massX = self.px.mass(X, logmode = True)
        massY = self.py.mass(Y, X, logmode = True)

        ret = massX + massY
        if not logmode:
            ret = np.exp(ret)
        return ret

    def map_via_cutset_sampling(self, z, query, N = -1):
        # z has np.nan as query/missing entries

        # edit, should not clip during inference
        # this should be done after inference

        if N <= 0:
            N = self.sample_size

        np.random.seed(N) # fix seed
        nan_idx = np.where(np.isnan(z))[0]
        missing = utils.notin(nan_idx, query)

        x = z[self.xids]
        y = z[self.yids]
        assert( len(query) > 0 ), "No query variables."

        unknown = np.concatenate([query, missing])
        ux = []; uy = []
        for u in unknown:
            if u in self.xids:
                ux.append( self.xids.index(u) )
            else:
                uy.append( self.yids.index(u) )

        if len(ux) > 0 :
            sx = self.px.rvs( N*2, evi = x )
            # sx = np.clip(sx, 0, 1) # added 04-04-23 22:52

            # keep only top N best samples
            mass = self.px.mass(sx)
            mass_rank = np.argsort(mass)
            selector = mass_rank[-N:]
            sx = sx[selector]
        else:
            sx = np.array([x]).reshape(1,-1)

        size = sx.shape[0]
        A,B,S = self.py.nn.np_forward(sx)
        distY = [self.py((A[i],B[i],S[i]), False) for i in range(size)]

        if len(uy) > 0:
            sy = [m.predict(y) for m in distY]
        else:
            sy = [y for _ in range(size)]
        sy = np.array(sy)
        # sy = np.clip(sy, 0, 1) # added 04-04-23 22:45

        # pdb.set_trace()
        mass = []
        y_qe = [i for i,v in enumerate(self.yids) if v not in missing]
        for i in range(size):
            x = sx[i]
            y = sy[i]
            mass.append( self.px.mass(x) * distY[i].marginalize(y_qe).mass(y[y_qe]) )
        mass = np.array(mass)

        # return the sample with highest density
        best_idx = np.argmax(mass)
        ret = z.copy()
        ret[self.xids] = sx[best_idx]
        ret[self.yids] = sy[best_idx]
        return ret
    
    def optimize_assignment(self, z, query, max_iter = 50):
        assert(len(query) > 0)
        px = self.px
        py = self.py
        nn = py.nn
        
        # construct candidate initial point
        z = np.clip(z,0,1)
        z_copy = z.copy()
        best_val = self.mass(z)[0]
        best_pred = z
        
        if not hasattr(px, 'tensor_'):
            px._predict_setup_()
    
        if px.tensor_ is None or max_iter <= 0:
            return best_pred
        
        # fix parameters in py.nn
        U,P,C,W = px.tensor_
        for p in py.nn.parameters():
            p.require_grad = False

            
        # define the objective function 
        N = px.W.size
        def objective(p):
            cur = torch.from_numpy(z_copy).to(dtype=torch.float)
            cur[query] = torch.clip(p,0,1)
            x = cur[self.xids]
            y = cur[self.yids]
                    
            comp_ll = torch.zeros(N)
            for i in range(N):
                vec = (x - U[i]).reshape(-1,1)
                comp_ll[i] = -0.5 * (vec.T @ P[i] @ vec)
            comp_ll = comp_ll + C + W
            ll1 = torch.logsumexp(comp_ll, dim=0)
            
            out = py.nn.forward( x.reshape(1,-1) )
            ll2 = py.nn.loss(out, y.reshape(1,-1), return_mass = True).sum()
            
            return -(ll1+ll2), cur
        
        # define the parameter
        init = torch.from_numpy(z[query])
        assignment = torch.nn.Parameter( init.to(dtype=torch.float) )
        optimizer = torch.optim.Adam([assignment])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.2, anneal_strategy='cos', pct_start=0.25, \
                            epochs=max_iter, steps_per_epoch = 1, verbose = False)

#         init_mass = best_val
        
        # optimization 
        t = 0
        while t < max_iter:
            optimizer.zero_grad()
            val, _z = objective(assignment)
            val.backward()
            t += 1
            
            if assignment.grad.abs().max() < 1e-3:
                break
            
            if -val.item() > best_val:
                best_val = -val.item()
                best_pred = np.clip(_z.data.numpy(), 0, 1)
            
            # optimizer cannot step too early
            optimizer.step()
            scheduler.step()
        
#         after_mass = self.mass(best_pred)[0]
#         gain = after_mass - init_mass
#         print('Gain', gain)
#         assert(gain>=0)
        
        return best_pred


    


