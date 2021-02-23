import numpy as np
import copy
from scipy.stats import multivariate_normal
from scipy import spatial, linalg
from scipy.special import logsumexp
from scipy.sparse import diags
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _compute_log_det_cholesky,
    _estimate_gaussian_covariances_full,
    _estimate_gaussian_covariances_tied,
    _estimate_gaussian_covariances_diag,
    _estimate_gaussian_covariances_spherical,
    _check_means,
    _check_precisions,
    _check_shape,
)
from sklearn.mixture._base import _check_X, check_random_state, ConvergenceWarning
import warnings
from .mat_utils import mkvc
from ..maps import IdentityMap, Wires
from ..regularization import (
    SimplePGI,
    Simple,
    PGI,
    Tikhonov,
    SimplePGIwithRelationships,
)
from .pgi_utils import (
    WeightedGaussianMixture,
    GaussianMixtureWithPrior,
)

class GaussianMixtureMarkovRandomField(WeightedGaussianMixture):

    def __init__(
        self,
        n_components,
        mesh,
        actv=None,
        kdtree=None, indexneighbors=None,
        boreholeidx=None,
        T=12.,
        kneighbors=0,
        norm_neighbors=2,
        init_params='kmeans', max_iter=100,
        covariance_type='full',
        means_init=None, n_init=10, precisions_init=None,
        random_state=None, reg_covar=1e-06, tol=0.001, verbose=0,
        verbose_interval=10, warm_start=False, weights_init=None,
        anisotropy=None,
        #unit_anisotropy=None, # Dictionary with unit, anisotropy and index
        #unit_kdtree=None, # List of KDtree
        index_anisotropy=None, # Dictionary with anisotropy and index
        index_kdtree=None,# List of KDtree
        #**kwargs
    ):

        super(GaussianMixtureMarkovRandomField, self).__init__(
            n_components=n_components,
            mesh=mesh,
            actv=actv,
            covariance_type=covariance_type,
            init_params=init_params,
            max_iter=max_iter,
            means_init=means_init,
            n_init=n_init,
            precisions_init=precisions_init,
            random_state=random_state,
            reg_covar=reg_covar,
            tol=tol,
            verbose=verbose,
            verbose_interval=verbose_interval,
            warm_start=warm_start,
            weights_init=weights_init,
            #boreholeidx=boreholeidx
            # **kwargs
        )
        # setKwargs(self, **kwargs)
        self.kneighbors = kneighbors
        self.T = T
        self.boreholeidx = boreholeidx
        self.anisotropy = anisotropy

        if self.mesh.gridCC.ndim == 1:
            xyz = np.c_[self.mesh.gridCC]
        elif self.anisotropy is not None:
            xyz = self.anisotropy.dot(self.mesh.gridCC.T).T
        else:
            xyz = self.mesh.gridCC
        if self.actv is None:
            self.xyz = xyz
        else:
            self.xyz = xyz[self.actv]
        if kdtree is None:
            print('Computing KDTree, it may take several minutes.')
            self.kdtree = spatial.KDTree(self.xyz)
        else:
            self.kdtree = kdtree
        if indexneighbors is None:
            print('Computing neighbors, it may take several minutes.')
            _, self.indexneighbors = self.kdtree.query(self.xyz, k=self.kneighbors+1, p=norm_neighbors)
        else:
            self.indexneighbors = indexneighbors

        self.indexpoint = copy.deepcopy(self.indexneighbors)
        self.index_anisotropy = index_anisotropy
        self.index_kdtree = index_kdtree
        if self.index_anisotropy is not None and self.mesh.gridCC.ndim != 1:

            self.unitxyz = []
            for i, anis in enumerate(self.index_anisotropy['anisotropy']):
                self.unitxyz.append((anis).dot(self.xyz.T).T)

            if self.index_kdtree is None:
                self.index_kdtree = []
                print('Computing rock unit specific KDTree, it may take several minutes.')
                for i, anis in enumerate(self.index_anisotropy['anisotropy']):
                    self.index_kdtree.append(spatial.KDTree(self.unitxyz[i]))

            #print('Computing new neighbors based on rock units, it may take several minutes.')
            #for i, unitindex in enumerate(self.index_anisotropy['index']):
        #        _, self.indexpoint[unitindex] = self.index_kdtree[i].query(self.unitxyz[i][unitindex], k=self.kneighbors+1)


    def computeG(self, z, w, X):

        #Find neighbors given the current state of data and model
        if self.index_anisotropy is not None and self.mesh.gridCC.ndim != 1:
            prediction = self.predict(X)
            self.unit_index = []
            self.indexpointlist = []
            logG = np.zeros([self.xyz.shape[0],self.n_components])
            for i in range(self.n_components):
                unitindex = np.where(prediction==i)[0]
                self.unit_index.append(unitindex)
                _, idxpt = self.index_kdtree[i].query(
                    self.unitxyz[i][unitindex],
                    k=self.index_anisotropy['kneighbors'][i]+1,
                    p=self.index_anisotropy['norm'][i]
                )
                self.indexpointlist.append(idxpt)
                logG[unitindex] = (self.T/(2.*(self.index_anisotropy['kneighbors'][i]+1))) * (
                    (z[idxpt] + w[idxpt]).sum(axis=1)
                )

        else:
            logG = (self.T/(2.*(self.kneighbors+1))) * (
                (z[self.indexpoint] + w[self.indexpoint]).sum(axis=1)
            )
        return logG

    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        _, self.means_, self.covariances_ = (
            self._estimate_gaussian_parameters(X, self.mesh, np.exp(log_resp), self.reg_covar,self.covariance_type)
        )
        #self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

        logweights = logsumexp(np.c_[[log_resp, self.computeG(np.exp(log_resp), self.weights_,X)]], axis=0)
        logweights = logweights - logsumexp(
            logweights, axis=1, keepdims=True
        )

        self.weights_ = np.exp(logweights)
        if self.boreholeidx is not None:
            aux = np.zeros((self.boreholeidx.shape[0],self.n_components))
            aux[np.arange(len(aux)), self.boreholeidx[:,1]]=1
            self.weights_[self.boreholeidx[:,0]] = aux


    def _check_weights(self, weights, n_components, n_samples):
        """Check the user provided 'weights'.
        Parameters
        ----------
        weights : array-like, shape (n_components,)
            The proportions of components of each mixture.
        n_components : int
            Number of components.
        Returns
        -------
        weights : array, shape (n_components,)
        """
        weights = check_array(
            weights, dtype=[np.float64, np.float32],
            ensure_2d=True
        )
        _check_shape(weights, (n_components, n_samples), 'weights')

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        n_samples, n_features = X.shape
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        if self.weights_init is not None:
            self.weights_init = self._check_weights(
                self.weights_init,
                n_samples,
                self.n_components
            )

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, n_features)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(self.precisions_init,
                                                     self.covariance_type,
                                                     self.n_components,
                                                     n_features)

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = self._estimate_gaussian_parameters(
            X, self.mesh, resp, self.reg_covar, self.covariance_type)
        weights /= n_samples

        self.weights_ = (weights*np.ones((n_samples,self.n_components)) if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init



class GaussianMixtureMarkovRandomFieldWithPrior(GaussianMixtureWithPrior):

    def __init__(
        self, gmmref, kappa=0., nu=0., zeta=0.,
        prior_type='semi',  # semi or full
        update_covariances=False,
        fixed_membership=None,
        boreholeidx=None,
        init_params='kmeans', max_iter=100,
        means_init=None, n_init=10, precisions_init=None,
        random_state=None, reg_covar=1e-06, tol=0.001, verbose=0,
        verbose_interval=10, warm_start=False, weights_init=None,
        #kdtree=None, indexneighbors=None,
        #T=12., kneighbors=0,
        #anisotropy=None,
        #unit_anisotropy=None, # Dictionary with unit, anisotropy and index
        #unit_kdtree=None, # List of KDtree
        #index_anisotropy=None, # Dictionary with anisotropy and index
        #index_kdtree=None,# List of KDtree
        #**kwargs
    ):

        super(GaussianMixtureMarkovRandomFieldWithPrior, self).__init__(
            gmmref=gmmref,
            kappa=kappa, nu=nu, zeta=zeta,
            prior_type=prior_type,
            update_covariances=update_covariances,
            fixed_membership=fixed_membership,
            init_params=init_params,
            max_iter=max_iter,
            means_init=means_init,
            n_init=n_init,
            precisions_init=precisions_init,
            random_state=random_state,
            reg_covar=reg_covar,
            tol=tol,
            verbose=verbose,
            verbose_interval=verbose_interval,
            warm_start=warm_start,
            weights_init=weights_init,
            #boreholeidx=boreholeidx
            # **kwargs
        )
        # setKwargs(self, **kwargs)

        self.kdtree=self.gmmref.kdtree
        self.indexneighbors=self.gmmref.indexneighbors
        self.T=self.gmmref.T
        self.kneighbors=self.gmmref.kneighbors
        self.anisotropy=self.gmmref.anisotropy
        #self.unit_anisotropy=None, # Dictionary with unit, anisotropy and index
        #self.unit_kdtree=None, # List of KDtree
        self.index_anisotropy=self.gmmref.index_anisotropy
        self.index_kdtree=self.gmmref.index_kdtree
        self.boreholeidx = self.gmmref.boreholeidx

        if self.mesh.gridCC.ndim == 1:
            xyz = np.c_[self.mesh.gridCC]
        elif self.anisotropy is not None:
            xyz = self.anisotropy.dot(self.mesh.gridCC.T).T
        else:
            xyz = self.mesh.gridCC
        if self.actv is None:
            self.xyz = xyz
        else:
            self.xyz = xyz[self.actv]
        #if kdtree is None:
        #    print('Computing KDTree, it may take several minutes.')
        #    self.kdtree = spatial.KDTree(self.xyz)
        #else:
        #    self.kdtree = kdtree
        #if indexneighbors is None:
        #    print('Computing neighbors, it may take several minutes.')
        #    _, self.indexneighbors = self.kdtree.query(self.xyz, k=self.kneighbors+1)
        #else:
        #    self.indexneighbors = indexneighbors

        self.indexpoint = copy.deepcopy(self.indexneighbors)
        #self.index_anisotropy = index_anisotropy
        #self.index_kdtree = index_kdtree
        if self.index_anisotropy is not None and self.mesh.gridCC.ndim != 1:

            self.unitxyz = []
            for i, anis in enumerate(self.index_anisotropy['anisotropy']):
                self.unitxyz.append((anis).dot(self.xyz.T).T)

            if self.index_kdtree is None:
                self.index_kdtree = []
                print('Computing rock unit specific KDTree, it may take several minutes.')
                for i, anis in enumerate(self.index_anisotropy['anisotropy']):
                    self.index_kdtree.append(spatial.KDTree(self.unitxyz[i]))

            #print('Computing new neighbors based on rock units, it may take several minutes.')
            #for i, unitindex in enumerate(self.index_anisotropy['index']):
        #        _, self.indexpoint[unitindex] = self.index_kdtree[i].query(self.unitxyz[i][unitindex], k=self.kneighbors+1)


    def computeG(self, z, w, X):

        #Find neighbors given the current state of data and model
        if self.index_anisotropy is not None and self.mesh.gridCC.ndim != 1:
            prediction = self.predict(X)
            self.unit_index = []
            self.indexpointlist = []
            logG = np.zeros([self.xyz.shape[0],self.n_components])
            for i in range(self.n_components):
                unitindex = np.where(prediction==i)[0]
                self.unit_index.append(unitindex)
                _, idxpt = self.index_kdtree[i].query(
                    self.unitxyz[i][unitindex],
                    k=self.index_anisotropy['kneighbors'][i]+1,
                    p=self.index_anisotropy['norm'][i]
                )
                self.indexpointlist.append(idxpt)
                logG[unitindex] = (self.T/(2.*(self.index_anisotropy['kneighbors'][i]+1))) * (
                    (z[idxpt] + w[idxpt]).sum(axis=1)
                )

        else:
            logG = (self.T/(2.*(self.kneighbors+1))) * (
                (z[self.indexpoint] + w[self.indexpoint]).sum(axis=1)
            )
        return logG

    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        _, self.means_, self.covariances_ = (
            self._estimate_gaussian_parameters(X, self.mesh, np.exp(log_resp), self.reg_covar,self.covariance_type)
        )
        #self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

        logweights = logsumexp(np.c_[[log_resp, self.computeG(np.exp(log_resp), self.weights_, X)]], axis=0)
        logweights = logweights - logsumexp(
            logweights, axis=1, keepdims=True
        )

        self.weights_ = np.exp(logweights)
        if self.boreholeidx is not None:
            aux = np.zeros((self.boreholeidx.shape[0],self.n_components))
            aux[np.arange(len(aux)), self.boreholeidx[:,1]]=1
            self.weights_[self.boreholeidx[:,0]] = aux


    def _check_weights(self, weights, n_components, n_samples):
        """Check the user provided 'weights'.
        Parameters
        ----------
        weights : array-like, shape (n_components,)
            The proportions of components of each mixture.
        n_components : int
            Number of components.
        Returns
        -------
        weights : array, shape (n_components,)
        """
        weights = check_array(
            weights, dtype=[np.float64, np.float32],
            ensure_2d=True
        )
        _check_shape(weights, (n_components, n_samples), 'weights')

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        n_samples, n_features = X.shape
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        if self.weights_init is not None:
            self.weights_init = self._check_weights(
                self.weights_init,
                n_samples,
                self.n_components
            )

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, n_features)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(self.precisions_init,
                                                     self.covariance_type,
                                                     self.n_components,
                                                     n_features)


    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = self._estimate_gaussian_parameters(
            X, self.mesh, resp, self.reg_covar, self.covariance_type)
        weights /= n_samples

        self.weights_ = (weights*np.ones((n_samples,self.n_components)) if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init


def GibbsSampling_PottsDenoising(mesh, minit, log_univar, Pottmatrix,
                                 indActive=None,
                                 neighbors=8, norm=2,
                                 weighted_selection=True,
                                 compute_score=False,
                                 maxit=None,
                                 verbose=False,
                                 anisotropies=None):

    denoised = copy.deepcopy(minit)
    # Compute Tree for neighbors finding
    if mesh.dim == 1:
        GRIDCC = mkvc(mesh.gridCC, numDims=2)
    else:
        GRIDCC = mesh.gridCC
    if indActive is None:
        pass
    else:
        GRIDCC = GRIDCC[indActive]

        if self.index_kdtree is None:
            self.index_kdtree = []
            print('Computing rock unit specific KDTree, it may take several minutes.')
            for i, anis in enumerate(self.index_anisotropy['anisotropy']):
                self.index_kdtree.append(spatial.KDTree(self.unitxyz[i]))

    tree = spatial.KDTree(GRIDCC)
    n_components = log_univar.shape[1]

    if weighted_selection or compute_score:
        _, idx = tree.query(GRIDCC, k=neighbors + 1, p=norm)
        idx = idx[:, 1:]

    if weighted_selection:
        logprobnoise = -np.sum(np.r_[[Pottmatrix[minit[j], minit[idx[j]]]
                                      for j in range(len(minit))]], axis=1)
        idxmin = np.where(logprobnoise == logprobnoise.min())
        logprobnoise[idxmin] = -np.inf
        probnoise = np.exp(logprobnoise - logsumexp(logprobnoise))
        choice = np.arange(len(minit))
        if maxit is None:
            maxit = int(
                (1 + len(GRIDCC) - len(idxmin[0])) * np.log(1 + len(GRIDCC) - len(idxmin[0])))
            if verbose:
                print('max iterations: ', maxit)

    if compute_score:
        logprob_obj = []
        # Compute logprobability of the model, should increase
        unnormlogprob = np.sum(np.r_[[log_univar[i, denoised[i]] for i in range(len(denoised))]]) + np.sum(
            np.r_[[Pottmatrix[denoised[j], denoised[idx[j]]] for j in range(len(denoised))]])
        logprob_obj.append(unnormlogprob)

    if maxit is None:
        maxit = int((mesh.nC) * np.log(mesh.nC))
        if verbose:
            print('max iterations: ', maxit)

    for i in range(maxit):
        # select random point and neighbors
        if weighted_selection:
            j = np.random.choice(choice, p=probnoise)
            idxj = idx[j]
        else:
            j = np.random.randint(mesh.nC)
            if not weighted_selection or compute_score:
                _, idxj = tree.query(mesh.gridCC[j], k=neighbors+1, p=norm)

        # compute Probability
        postlogprob = np.zeros_like(log_univar[j])
        for k in range(n_components):
            postlogprob[k] = log_univar[j][k] + \
                np.sum([Pottmatrix[k, denoised[idc]] for idc in idxj])
        postprobj = np.exp(postlogprob - logsumexp(postlogprob))

        denoised[j] = np.random.choice(np.arange(n_components), p=postprobj)

        if compute_score:
            # Compute logprobability of the model, should increase
            unnormlogprob = np.sum(np.r_[[log_univar[i, denoised[i]] for i in range(len(denoised))]]) + np.sum(
                np.r_[[Pottmatrix[denoised[j], denoised[idx[j]]] for j in range(len(denoised))]])
            logprob_obj.append(unnormlogprob)

        if weighted_selection:
            # Update the probability of being noisy
            logprobnoise[j] = - \
                np.sum(np.r_[Pottmatrix[denoised[j], denoised[idx[j]]]])
            probnoise = np.exp(logprobnoise - logsumexp(logprobnoise))

    if compute_score and weighted_selection:
        return [denoised, probnoise, logprob_obj]

    elif not(compute_score or weighted_selection):
        return [denoised]

    elif compute_score:
        return [denoised, logprob_obj]

    elif weighted_selection:
        return [denoised, probnoise]


def ICM_PottsDenoising(mesh, minit, log_univar, Pottmatrix,
                       indActive=None,
                       neighbors=8, norm=2,
                       weighted_selection=True,
                       compute_score=False,
                       maxit=None,
                       verbose=True,
                       anisotropies=None):

    denoised = copy.deepcopy(minit)
    # Compute Tree for neighbors finding
    if mesh.dim == 1:
        GRIDCC = mkvc(mesh.gridCC, numDims=2)
    else:
        GRIDCC = mesh.gridCC
    if indActive is None:
        pass
    else:
        GRIDCC = GRIDCC[indActive]

#if self.index_anisotropy is not None and self.mesh.gridCC.ndim != 1:

#    self.unitxyz = []
#    for i, anis in enumerate(self.index_anisotropy['anisotropy']):
#        self.unitxyz.append((anis).dot(self.xyz.T).T)

#    if self.index_kdtree is None:
#        self.index_kdtree = []
#        print('Computing rock unit specific KDTree, it may take several minutes.')
#        for i, anis in enumerate(self.index_anisotropy['anisotropy']):
#            self.index_kdtree.append(spatial.KDTree(self.unitxyz[i]))


    #print('Computing new neighbors based on rock units, it may take several minutes.')
    #for i, unitindex in enumerate(self.index_anisotropy['index']):
#        _, self.indexpoint[unitindex] = self.index_kdtree[i].query(self.unitxyz[i][unitindex], k=self.kneighbors+1)

#Find neighbors given the current state of data and model
#if self.index_anisotropy is not None and self.mesh.gridCC.ndim != 1:
#    prediction = self.predict(X)
#    unit_index = []
#    for i in range(self.n_components):
#        unit_index.append(np.where(prediction==i)[0])
#    for i, unitindex in enumerate(unit_index):
#        _, self.indexpoint[unitindex] = self.index_kdtree[i].query(self.unitxyz[i][unitindex], k=self.kneighbors+1)

    #compute all tree
    #
    tree = spatial.KDTree(GRIDCC)
    n_components = log_univar.shape[1]
    if anisotropies is not None:
        treelist = []
        ani_xyzlist = []
        for anis in anisotropies['anisotropy']:
            ani_xyzlist.append(anis.dot(GRIDCC.T).T)
            treelist.append(spatial.KDTree(ani_xyzlist[-1]))

    if weighted_selection or compute_score:
        _, idx = tree.query(GRIDCC, k=neighbors + 1, p=norm)
        idx = idx[:, 1:]

        if anisotropies is not None:
            idxlist=[]
            for i, (anitree, xyz) in enumerate(zip(treelist,ani_xyzlist)):
                _, idx_ani = anitree.query(xyz, k=anisotropies['kneighbors'][i] + 1, p=anisotropies['norm'][i])
                idx_ani = idx_ani[:, 1:]
                idxlist.append(idx_ani)

            n_units = Pottmatrix.shape[0]
            unit_index = []

            for i in range(n_units):
                unit_index.append(np.where(denoised==i)[0])
            #form the initial list of neighbors for checking
            idx = []
            for i in range(GRIDCC.shape[0]):
                idx.append(idxlist[denoised[i]][i])
            #for i, unitindex in enumerate(unit_index):
            #    idx[unitindex] = idxlist[i][unitindex]

    if weighted_selection:
        logprobnoise = np.zeros(GRIDCC.shape[0])
        for i, unitindex in enumerate(unit_index):
            if unitindex.size>0:
                logprobnoise[unitindex] = -np.sum(np.r_[[Pottmatrix[denoised[unitindex][j], denoised[idxlist[i][unitindex][j]]]
                                      for j in range(len(denoised[unitindex]))]], axis=1)
        idxmin = np.where(logprobnoise == logprobnoise.min())
        #logprobnoise[idxmin] = -np.inf
        probnoise = np.exp(logprobnoise - logsumexp(logprobnoise))
        probnoise = probnoise/np.sum(probnoise)
        choice = np.arange(len(minit))
        if maxit is None:
            maxit = int(
                (1 + len(GRIDCC) - len(idxmin[0])) * np.log(1 + len(GRIDCC) - len(idxmin[0])))
            if verbose:
                print('max iterations: ', maxit)

    if compute_score:
        logprob_obj = []
        # Compute logprobability of the model, should increase
        unnormlogprob = np.sum(np.r_[[log_univar[i, denoised[i]] for i in range(len(denoised))]]) + np.sum(
            np.r_[[Pottmatrix[denoised[j], denoised[idx[j]]] for j in range(len(denoised))]])
        logprob_obj.append(unnormlogprob)

    if maxit is None:
        maxit = int((mesh.nC) * np.log(mesh.nC))
        if verbose:
            print('max iterations: ', maxit)

    for i in range(maxit):
        # select random point and neighbors
        if weighted_selection:
            j = np.random.choice(choice, p=probnoise)
            #idxj = idx[j]
        else:
            j = np.random.randint(mesh.nC)
            if not weighted_selection or compute_score:
                _, idxj = tree.query(mesh.gridCC[j], k=neighbors+1, p=norm)

        # compute Probability
        postlogprob = np.zeros(n_components)
        for k in range(n_components):
            postlogprob[k] = log_univar[j][k] + \
                np.sum([Pottmatrix[k, denoised[idc]] for idc in idxlist[k][j]])
        postprobj = np.exp(postlogprob - logsumexp(postlogprob))

        denoised[j] = np.argmax(postprobj)
        #update kneighbors is anisotropy specific to unit
        if anisotropies is not None:
            idx[j] = idxlist[denoised[j]][j]

        if compute_score:
            # Compute logprobability of the model, should increase
            unnormlogprob = np.sum(np.r_[[log_univar[i, denoised[i]] for i in range(len(denoised))]]) + np.sum(
                np.r_[[Pottmatrix[denoised[j], denoised[idx[j]]] for j in range(len(denoised))]])
            logprob_obj.append(unnormlogprob)

        if weighted_selection:
            # Update the probability of being noisy
            logprobnoise[j] = - \
                np.sum(np.r_[Pottmatrix[denoised[j], denoised[idxlist[denoised[j]][j]]]])
            probnoise = np.exp(logprobnoise - logsumexp(logprobnoise))
            probnoise = probnoise/np.sum(probnoise)

    if compute_score and weighted_selection:
        return [denoised, idx, probnoise, logprob_obj]

    elif not(compute_score or weighted_selection):
        return [denoised, idx]

    elif compute_score:
        return [denoised, idx, logprob_obj]

    elif weighted_selection:
        return [denoised, idx, probnoise]
