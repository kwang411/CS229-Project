import numpy as np
from scipy.spatial.distance import pdist, squareform

import dpp.sampler.dpp as dpp
import dpp.sampler.mcdpp as mcdpp
import dpp.sampler.utils as utils

# currently only support cpu mode
flag_gpu = False

def sample_ids(points, scores, k, gamma=1.):
    points = np.array(points)
    scores = np.array(scores).reshape(-1)
    
    # Construct kernel matrix
    scores = scores**(gamma/4.)
    pairwise_dists = squareform(pdist(points, 'euclidean'))
    L = np.exp(-pairwise_dists ** 2 / 0.5 ** 2)
    L = (((L*scores).T)*scores).T
    
    # Get eigendecomposition of kernel matrix
    D, V = utils.get_eig(L, flag_gpu=flag_gpu)
    E = utils.get_sympoly(D, k, flag_gpu=flag_gpu)
    
    dpp_smpl  = dpp.sample(D, V, E=E, k=k, flag_gpu=flag_gpu)
    return dpp_smpl

def cond_sample_ids(points, scores, cond_ids, k, gamma):
    points = np.array(points)
    scores = np.array(scores).reshape(-1)
    
    # Construct kernel matrix
    scores = scores**(gamma/4.)
    pairwise_dists = squareform(pdist(points, 'euclidean'))
    L = np.exp(-pairwise_dists ** 2 / 0.5 ** 2)
    L = (((L*scores).T)*scores).T
    
    # conditioning
    if len(cond_ids)>0:
        cond_ids = np.sort(cond_ids)
        eye1 = np.eye(len(points))
        for id in cond_ids:
            eye1[id,id] = 0
        L = np.linalg.inv(L + eye1)
        mask = np.full(len(points), True, dtype=bool)
        mask[cond_ids] = False
        L = L[mask,:]
        L = L[:,mask]
        L = np.linalg.inv(L) - np.eye(len(points)-len(cond_ids))
    
    # Get eigendecomposition of kernel matrix
    D, V = utils.get_eig(L, flag_gpu=flag_gpu)
    E = utils.get_sympoly(D, k, flag_gpu=flag_gpu)
    
    dpp_smpl = dpp.sample(D, V, E=E, k=k, flag_gpu=flag_gpu)
    
    for id in cond_ids:
        dpp_smpl[dpp_smpl>=id] +=1
    return dpp_smpl