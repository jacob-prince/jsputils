# citation: https://github.com/nathankong/robustness_primary_visual_cortex/tree/0fb7c1b5d945a5e0ed87339300d63b92d63e934b/robust_spectrum/eig_analysis

import os
import pickle

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Eigenspectrum():
    def __init__(self):
        self.data = None
        self.ss = None
        self.alpha = None
        self.ypred = None

    def compute_eigenspectrum(self):
        """
        Output:
            array of variance explained of size n_components
        """
        assert self.data is not None
        self.ss = self.pca_func(self.data)
        return self.ss

    def estimate_power_law_coefficient(self, fit_range=None):
        assert self.ss is not None
        assert fit_range is not None

        fit_range = np.arange(fit_range.min(), min(fit_range.max()+1, self.ss.size))

        # Fit power law coefficient
        self.alpha, self.ypred = get_powerlaw(self.ss/self.ss.sum(), fit_range.astype(int)) 
        return self.alpha, self.ypred

    def set_results(self, results):
        assert "ss" in results.keys()
        assert "ypred" in results.keys()
        assert "alpha" in results.keys()

        self.ss = results["ss"]
        self.alpha = results["alpha"]
        self.ypred = results["ypred"]

    def save_results(self, fname=None, results_dir=None):
        assert self.ss is not None
        assert self.alpha is not None
        assert self.ypred is not None

        if fname is None or results_dir is None:
            print("Did not save results. Filename is None.")
            return

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        fname = results_dir + f"/{fname}"

        results = dict()
        results["explained_variance_proportion"] = self.ss
        results["alpha"] = self.alpha
        results["y_pred"] = self.ypred
        pickle.dump(results, open(fname, "wb"))

    def pca_func(self, X):
        """
        Input:
            X : some data matrix (dimensions depends on pca_func implementation
        Output:
            array of variance explained of size n_components
        """
        raise NotImplementedError

class ArtificialNeuralResponseSpectrum(Eigenspectrum):
    def __init__(self, model, layer_name, dataloader, n_comp, n_batches=None):
        """
        Inputs:
            model      : (torch.nn.Module) PyTorch model.
            layer_name : (string) Layer from which to obtain features (activations).
            dataloader : (torch.utils.data.Dataloader) Dataloader for images.
            n_comp     : (int) Number of components for PCA.
            n_batches  : (int or None) Number of batches to obtain images from the 
                         dataloader. If the value is None, then use all the images.
        """
        super(Eigenspectrum ,self).__init__()
        fe = FeatureExtractor(
            dataloader,
            n_batches=n_batches,
            vectorize=True,
            debug=False
        )
        self.data = get_layer_features(fe, layer_name, model)
        self.n_components = n_comp

    def pca_func(self, X):
        """
        Input:
            X  : numpy array of dimensions (n_samples, n_features)
        Output:
            ss : numpy array of variance explained by each component
                 dimensions of (n_components,)
        """
        num_components = min(self.n_components, X.shape[0]-1, X.shape[1]-1)

        # Scale data
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X)
        X = scaler.transform(X)

        pca = PCA(n_components=num_components, svd_solver="full", whiten=False).fit(X)
        return pca.explained_variance_

class BiologicalNeuralResponseSpectrum(Eigenspectrum):
    def __init__(self, neural_data, n_comp=2000, n_shuffles=5):
        """
        Inputs:
            neural_data : NeuralDataset object. Contains information about the neural
                          responses and stimulus set used.
            n_comp      : int. Number of components for cross-validated PCA.
            n_shuffles  : int. Number of shuffles for cross-validated PCA.
        """
        super(Eigenspectrum, self).__init__()
        self.data = neural_data
        self.n_shuffles = n_shuffles
        self.n_components = n_comp

    def pca_func(self, X):
        """
        Input:
            X  : numpy array of dimensions (2, n_stimuli, n_neurons)
        Output:
            ss : numpy array of variance explained by each component
                 dimensions of (n_components,)
        """
        assert X.ndim == 3
        assert X.shape[0] == 2
        ss = shuff_cvPCA(X, self.n_components, nshuff=self.n_shuffles)
        ss = ss.mean(axis=0)
        return 
    
    
#########

# Functions below are from https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/utils.py
def get_powerlaw(ss, trange):
    ''' fit exponent to variance curve'''
    logss = np.log(np.abs(ss))
    y = logss[trange][:,np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:,np.newaxis], np.ones((nt,1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:,np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()
    
    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:,np.newaxis], np.ones((ss.size,1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    return alpha,ypred

def shuff_cvPCA(X, n_comp, nshuff=5):
    ''' X is 2 x stimuli x neurons '''
    nc = min(n_comp, X.shape[1])
    ss=np.zeros((nshuff,nc))
    for k in range(nshuff):
        print(f"Shuffle {k+1}...")
        iflip = np.random.rand(X.shape[1]) > 0.5
        X0 = X.copy()
        X0[0,iflip] = X[1,iflip]
        X0[1,iflip] = X[0,iflip]
        ss[k]=cvPCA(X0, n_comp)
    return ss

def cvPCA(X, n_comp):
    ''' X is 2 x stimuli x neurons '''
    X = X - np.mean(X, axis=1)[:,np.newaxis,:]

    pca = PCA(n_components=min(n_comp, X.shape[1]), svd_solver="full").fit(X[0].T)

    u = pca.components_.T
    sv = pca.singular_values_
    
    xproj = X[0].T @ (u / sv)
    cproj0 = X[0] @ xproj
    cproj1 = X[1] @ xproj
    ss = (cproj0 * cproj1).sum(axis=0)
    return ss

def my_pca(X, n_comp=2000):
    ''' X is 2 x stimuli x neurons '''
    pca = PCA(n_components=min(n_comp, X.shape[1]), svd_solver="full", whiten=True).fit(X[0])
    return pca.explained_variance_