import numpy as np

from pymanopt.manifolds.manifold import Manifold
from t3f import TensorTrain

class FixedRankTT(Manifold):
    
    # Manifold of tensors having fixed TT rank,

    def __init__(self, tt_cores, shape=None, tt_ranks=None, convert_to_tensors=True):
              
        tens = t3f.TensorTrain(tt_cores, shape=None, tt_ranks=None, convert_to_tensors=True)
        self._name = ("Manifold of TT_Tensors with fixed tt_rank " + str(tens.get_tt_ranks()))
        self._shape = t3f.lazy_shape(tens)
        self._tt_rank = t3f.lazy_tt_ranks(tens)

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return np.prod(self._shape)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def dist(self, X, Y):
        return t3f.frobenius_norm(X-Y)
        
    def inner(self, X, G, H):
        # i.e. inner prod on tangent space, see base class
        G_proj = t3f.project(G, X)
        H_proj = t3f.project(H, X)
        return t3f.pairwise_flat_inner_projected(G_proj, H_proj).numpy()

    def proj(self, X, Z):
        return t3f.project(Z, X)
    
    def egrad2rgrad(self, X, egrad):
        rgrad = t3f.project(egrad, X)
        rgrad = t3f.renormalize_tt_cores(rgrad)
        return rgrad

    def retr(self, X, Z):
        step_moved = X + Z
        retracted = t3f.round(step_moved,max_tt_rank=max(t3f.lazy_tt_ranks(X)))
        retracted = t3f.renormalize_tt_cores(retracted)
        return retracted
    
    def renormalize(self, X):
        return t3f.renormalize_tt_cores(X)

    def norm(self, X, G):
        return t3f.frobenius_norm(G)
    
    def rand(self):
        return t3f.random_tensor(shape=self._shape, tt_rank=self._tt_rank)
    
    def randvec(self, X):
        rand_manifold = self.rand()
        rand_tangent = t3f.project(rand_manifold, X)
        return rand_tangent
        
    def tangent2ambient(self, X, Z):
        return self.retr(X,Y)
    
    def transp(self, X1, X2, G):
        return t3f.project(G,X2)
    
    def exp(self, X, U):
        return t3f.round(X+U,max_tt_rank=max(t3f.lazy_tt_ranks(X)))

    def log(self, X, Y):
        return t3f.round(X-Y,max_tt_rank=max(t3f.lazy_tt_ranks(X)))

    def pairmean(self, X, Y):
        return t3f.round(t3f.multiply(X+Y,0.5),max_tt_rank=max(t3f.lazy_tt_ranks(X)))

    def zerovec(self, X):
        random_vec = self.randvec(X)
        return t3f.zeros_like(random_vec)
