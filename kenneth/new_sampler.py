import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import pickle
import os.path

class Kernel(object):
    # Returns the submatrix of the kernel indexed by ps and qs
    def getKernel(self, ps, qs):
        return np.squeeze(ps[:,None]==qs[None,:]).astype(float)
        # Readable version: return np.array([[1. if p==q else 0. for q in qs] for p in ps])
    def __getitem__(self, args):
        ps, qs = args
        return self.getKernel(ps, qs)

class RBFKernel(Kernel):
    def __init__(self, R):
        self.R = R
    def getKernel(self, ps, qs):
        D = cdist(ps,qs)**2
        # Readable version: D = np.array([[np.dot(p-q, p-q) for q in qs] for p in ps])
        D = np.exp(-D/(2*self.R**2))
        return D
    
class ScoredKernel(Kernel):
    def __init__(self, R, space, scores, alpha=2, gamma=1):
        self.R = R
        self.space = space
        if alpha > 0:
            self.scores = scores**(gamma/2./alpha)
        else:
            self.scores = scores
    def getKernel(self, p_ids, q_ids):
        p_ids = np.squeeze(np.array([p_ids]))
        if len(p_ids.shape)<1:
            p_ids = np.array([p_ids])
        q_ids = np.squeeze(np.array([q_ids]))
        if len(q_ids.shape)<1:
            q_ids = np.array([q_ids])
        
        ps = np.array(self.space)[p_ids]
        qs = np.array(self.space)[q_ids]
        D = cdist(ps,qs)**2
        
        # Readable version: D = np.array([[np.dot(p-q, p-q) for q in qs] for p in ps])
        D = np.exp(-D/(2*self.R**2))
        
        # I added the below line to have different sample scores
        D = ((D*self.scores[p_ids]).T * self.scores[q_ids]).T
        return D
    
class ConditionedScoredKernel(Kernel):
    def __init__(self, R, space, scores, cond_ids, alpha=2, gamma=1):
        self.R = R
        self.space = np.array(space)
        self.cond_ids = np.sort(cond_ids)
        if alpha > 0:
            self.scores = np.array(scores**(gamma/2./alpha)).reshape(-1)
        else:
            self.scores = scores
        self.kernel = self.computeFullKernel()
        
        
    def computeFullKernel(self):
        # Construct kernel matrix
        pairwise_dists = squareform(pdist(self.space, 'euclidean'))
        L = np.exp(-pairwise_dists ** 2 / 0.5 ** 2)
        L = (((L*self.scores).T)*self.scores).T
        
        # conditioning
        if len(self.cond_ids)>0:
            eye1 = np.eye(len(self.space))
            for id in self.cond_ids:
                eye1[id,id] = 0
            L = np.linalg.inv(L + eye1)
            mask = np.full(len(self.space), True, dtype=bool)
            mask[self.cond_ids] = False
            L = L[mask,:]
            L = L[:,mask]
            L = np.linalg.inv(L) - np.eye(len(self.space)-len(self.cond_ids))
            for id in self.cond_ids: # cond_ids is sorted
                L = np.insert(L, id, 0, axis=0)
                L = np.insert(L, id, 0, axis=1)
                
        return L
    
    def getKernel(self, p_ids, q_ids):
        p_ids = np.squeeze(np.array([p_ids]))
        if len(p_ids.shape)<1:
            p_ids = np.array([p_ids])
        q_ids = np.squeeze(np.array([q_ids]))
        if len(q_ids.shape)<1:
            q_ids = np.array([q_ids])
        
        D = self.kernel[p_ids,:]
        D = D[:,q_ids]
        return D

class Sampler(object):
    def __init__(self, kernel, space, k):
        self.kernel = kernel
        self.space = space
        self.k = k
        # norms will hold the diagonals of the kernel
        self.norms = np.array([self.kernel[p_id, p_id][0][0] for p_id in range(len(self.space))])
        self.clear()
    def clear(self):
        # S will hold chosen set of k points
        self.S = []
        # M will hold the inverse of the kernel on S
        self.M = np.zeros(shape=(0, 0))
    def makeSane(self):
        self.M = np.linalg.pinv(self.kernel[self.S, self.S])
    def testSanity(self):
        eps = 1e-4
        assert self.M.shape == (len(self.S), len(self.S))
        if len(self.S)>0:
            diff = np.abs(np.dot(self.M, self.kernel[self.S, self.S])-np.eye(len(self.S)))
            assert np.all(np.abs(diff)<=eps)
    def append(self, ind):
        if len(self.S)==0:
            self.S = [ind]
            self.M = np.array([[1./self.norms[ind]]])
        else:
            u = self.kernel[self.S, ind]
            # Compute Schur complement inverse
            scInv = 1./(self.norms[ind]-np.dot(u.T, np.dot(self.M, u)))
            v = np.dot(self.M, u)
            self.M = np.block([[self.M+scInv*np.outer(v, v), -scInv*v], [-scInv*v.T, scInv]])
            self.S.append(ind)
    def remove(self, i):
        if len(self.S)==1:
            self.S = []
            self.M = np.zeros(shape=(0, 0))
        else:
            mask = [True]*len(self.S)
            mask[i] = False
            # Readable version: mask = [j!=i for j in range(len(self.S))]
            scInv = self.M[i, i]
            v = self.M[mask, i]
            self.M = self.M[mask, :][:, mask] - np.outer(v, v)/scInv
            self.S = self.S[:i] + self.S[i+1:]
    # Return array containing ratio of increase in kernel determinant after adding each point in space
    def ratios(self):
        if len(self.S)==0:
            return self.norms
        else:
            U = self.kernel[np.arange(len(self.space)), self.S]
            return self.norms - np.sum(np.dot(U, self.M)*U, axis=1)
    # Finds greedily the item to add to maximize the determinant of the kernel
    def addGreedy(self):
        self.append(np.argmax(self.ratios()))
    # Important step, because we need to start from a point whose probability is not too small
    def warmStart(self):
        self.clear()
        for i in range(self.k):
            self.addGreedy()
    # Run one step of Markov chain
    def step(self, alpha=1.):
        self.remove(np.random.randint(len(self.S)))
        probs = np.maximum(self.ratios(), 0.)**alpha
        probs /= sum(probs)
        self.append(np.random.choice(len(probs), p=probs))
    def sample(self, alpha=2., steps=1000, makeSaneEvery=10):
        assert(len(self.S)==self.k) # otherwise appropriate warmstart didn't run
        for iter in range(steps):
            self.step(alpha=alpha)
            if iter%makeSaneEvery==0:
                self.makeSane()
        return self.S

def setup_sampler(space, scores, k, alpha, gamma):
    points = [entry for entry in space]
    s = Sampler(ScoredKernel(1., space, scores, alpha, gamma), space, k)
    s.warmStart()
    return s

def sample_ids_mc(points, scores, k, alpha=2., gamma=1.):
    points = [p for p in points]
    scores = np.array(scores).reshape(-1,1)
    s = setup_sampler(points, scores, k, alpha, gamma)
    x = s.sample(alpha=alpha)
    return x

def sample_mc(points, scores, k, alpha=2., gamma=1.):
    x = sample_ids_mc(points, scores, k, alpha, gamma)
    return np.array(points)[x]




def cond_setup_sampler(space, scores, cond_ids, k, alpha, gamma):
    points = [entry for entry in space]
    s = Sampler(ConditionedScoredKernel(1., space, scores, cond_ids, alpha, gamma), space, k)
    s.warmStart()
    return s

def cond_sample_ids_mc(points, scores, cond_ids, k, alpha=2., gamma=1.):
    points = [p for p in points]
    scores = np.array(scores).reshape(-1,1)
    s = cond_setup_sampler(points, scores, cond_ids, k, alpha, gamma)
    x = s.sample(alpha=alpha)
    return x

def cond_sample_mc(points, scores, cond_ids, k, alpha=2., gamma=1.):
    x = sample_ids_mc(points, scores, cond_ids, k, alpha, gamma)
    return np.array(points)[x]