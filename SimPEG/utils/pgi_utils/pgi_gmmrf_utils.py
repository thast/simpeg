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
from sklearn.mixture._base import check_random_state, ConvergenceWarning
import warnings
from ..mat_utils import mkvc
from ...maps import IdentityMap, Wires
from ...regularization import (
    SimplePGI,
    Simple,
    PGI,
    Tikhonov,
)
from .pgi_gmm_utils import (
    WeightedGaussianMixture,
    GaussianMixtureWithPrior,
)

def make_SimplePGI_GMMRF_continuity_regularization(
    mesh,
    gmmref,
    gmm=None,
    wiresmap=None,
    maplist=None,
    cell_weights_list=None,
    approx_hessian=True,
    approx_gradient=True,
    approx_eval=True,
    alpha_s=1.0,
    alpha_x=1.0,
    alpha_y=1.0,
    alpha_z=1.0,
    alpha_xx=0.0,
    alpha_yy=0.0,
    alpha_zz=0.0,
    **kwargs
):
    """
    Create a complete SimplePGI regularization term ComboObjectiveFunction with all
    necessary smallness and smoothness terms for any number of physical properties
    and associated mapping.

    Parameters
    ----------

    :param TensorMesh or TreeMesh mesh: TensorMesh or Treemesh object, used to weights
                        the physical properties by cell volumes when updating the
                        Gaussian Mixture Model (GMM)
    :param WeightedGaussianMixture gmmref: reference GMM.
    :param WeightedGaussianMixture gmm: Initial GMM. If not provided, gmmref is used.
    :param Wires wiresmap: Wires map to obtain the various physical properties from the model.
                        Optional for single physical property inversion. Required for multi-
                        physical properties inversion.
    :param list maplist: List of mapping for each physical property. Default is the IdentityMap for all.
    :param list cell_weights_list: list of numpy.ndarray for the cells weight to apply to each physical property.
    :param boolean approx_gradient: use the PGI least-squares approximation of the full nonlinear regularizer
                        for computing the regularizer gradient. Default is True.
    :param boolean approx_eval: use the PGI least-squares approximation of the full nonlinear regularizer
                        for computing the value of the regularizer. Default is True.
    :param float alpha_s: alpha_s multiplier for the PGI smallness.
    :param float or numpy.ndarray alpha_x: alpha_x multiplier for the 1st-derivative
                        Smoothness terms in X-direction for each physical property.
    :param float or numpy.ndarray alpha_y: alpha_y multiplier for the 1st-derivative
                        Smoothness terms in Y-direction for each physical property.
    :param float or numpy.ndarray alpha_z: alpha_z multiplier for the 1st-derivative
                        Smoothness terms in Z-direction for each physical property.
    :param float or numpy.ndarray alpha_x: alpha_x multiplier for the 2nd-derivatibe
                        Smoothness terms in X-direction for each physical property.
    :param float or numpy.ndarray alpha_y: alpha_y multiplier for the 2nd-derivatibe
                        Smoothness terms in Y-direction for each physical property.
    :param float or numpy.ndarray alpha_z: alpha_z multiplier for the 2nd-derivatibe
                        Smoothness terms in Z-direction for each physical property.


    Returns
    -------

    :param SimPEG.objective_function.ComboObjectiveFunction reg: Full regularization with simplePGIsmallness
                        and smoothness terms for all physical properties in all direction.
    """

    if wiresmap is None:
        if "indActive" in kwargs.keys():
            indActive = kwargs.pop("indActive")
            wrmp = Wires(("m", indActive.sum()))
        else:
            wrmp = Wires(("m", mesh.nC))
    else:
        wrmp = wiresmap

    if maplist is None:
        mplst = [IdentityMap(mesh) for maps in wrmp.maps]
    else:
        mplst = maplist

    if cell_weights_list is None:
        clwhtlst = [Identity() for maps in wrmp.maps]
    else:
        clwhtlst = cell_weights_list

    reg = SimplePGI(
        mesh=mesh,
        gmmref=gmmref,
        gmm=gmm,
        wiresmap=wiresmap,
        maplist=maplist,
        approx_hessian=approx_hessian,
        approx_gradient=approx_gradient,
        approx_eval=approx_eval,
        alpha_s=alpha_s,
        alpha_x=0.0,
        alpha_y=0.0,
        alpha_z=0.0,
        **kwargs
    )

    if cell_weights_list is not None:
        reg.objfcts[0].cell_weights = np.hstack(clwhtlst)

    if isinstance(alpha_x, float):
        alph_x = alpha_x * np.ones(len(wrmp.maps))
    else:
        alph_x = alpha_x

    if isinstance(alpha_y, float):
        alph_y = alpha_y * np.ones(len(wrmp.maps))
    else:
        alph_y = alpha_y

    if isinstance(alpha_z, float):
        alph_z = alpha_z * np.ones(len(wrmp.maps))
    else:
        alph_z = alpha_z

    for i, (wire, maps) in enumerate(zip(wrmp.maps, mplst)):
        reg += Simple(
            mesh=mesh,
            mapping=maps * wire[1],
            alpha_s=0.0,
            alpha_x=alph_x[i],
            alpha_y=alph_y[i],
            alpha_z=alph_z[i],
            cell_weights=clwhtlst[i],
            **kwargs
        )

    return reg


def make_PGI_GMMRF_continuity_regularization(
    mesh,
    gmmref,
    gmm=None,
    wiresmap=None,
    maplist=None,
    cell_weights_list=None,
    approx_hessian=True,
    approx_gradient=True,
    approx_eval=True,
    alpha_s=1.0,
    alpha_x=1.0,
    alpha_y=1.0,
    alpha_z=1.0,
    alpha_xx=0.0,
    alpha_yy=0.0,
    alpha_zz=0.0,
    **kwargs
):
    """
    Create a complete PGI regularization term ComboObjectiveFunction with all
    necessary smallness and smoothness terms for any number of physical properties
    and associated mapping.

    Parameters
    ----------

    :param TensorMesh or TreeMesh mesh: TensorMesh or Treemesh object, used to weights
                        the physical properties by cell volumes when updating the
                        Gaussian Mixture Model (GMM)
    :param WeightedGaussianMixture gmmref: reference GMM.
    :param WeightedGaussianMixture gmm: Initial GMM. If not provided, gmmref is used.
    :param Wires wiresmap: Wires map to obtain the various physical properties from the model.
                        Optional for single physical property inversion. Required for multi-
                        physical properties inversion.
    :param list maplist: List of mapping for each physical property. Default is the IdentityMap for all.
    :param list cell_weights_list: list of numpy.ndarray for the cells weight to apply to each physical property.
    :param boolean approx_gradient: use the PGI least-squares approximation of the full nonlinear regularizer
                        for computing the regularizer gradient. Default is True.
    :param boolean approx_eval: use the PGI least-squares approximation of the full nonlinear regularizer
                        for computing the value of the regularizer. Default is True.
    :param float alpha_s: alpha_s multiplier for the PGI smallness.
    :param float or numpy.ndarray alpha_x: alpha_x multiplier for the 1st-derivative
                        Smoothness terms in X-direction for each physical property.
    :param float or numpy.ndarray alpha_y: alpha_y multiplier for the 1st-derivative
                        Smoothness terms in Y-direction for each physical property.
    :param float or numpy.ndarray alpha_z: alpha_z multiplier for the 1st-derivative
                        Smoothness terms in Z-direction for each physical property.
    :param float or numpy.ndarray alpha_x: alpha_x multiplier for the 2nd-derivatibe
                        Smoothness terms in X-direction for each physical property.
    :param float or numpy.ndarray alpha_y: alpha_y multiplier for the 2nd-derivatibe
                        Smoothness terms in Y-direction for each physical property.
    :param float or numpy.ndarray alpha_z: alpha_z multiplier for the 2nd-derivatibe
                        Smoothness terms in Z-direction for each physical property.


    Returns
    -------

    :param SimPEG.objective_function.ComboObjectiveFunction reg: Full regularization with PGIsmallness
                        and smoothness terms for all physical properties in all direction.
    """

    if wiresmap is None:
        if "indActive" in kwargs.keys():
            indActive = kwargs.pop("indActive")
            wrmp = Wires(("m", indActive.sum()))
        else:
            wrmp = Wires(("m", mesh.nC))
    else:
        wrmp = wiresmap

    if maplist is None:
        mplst = [IdentityMap(mesh) for maps in wrmp.maps]
    else:
        mplst = maplist

    if cell_weights_list is None:
        clwhtlst = [Identity() for maps in wrmp.maps]
    else:
        clwhtlst = cell_weights_list

    reg = PGI(
        mesh=mesh,
        gmmref=gmmref,
        gmm=gmm,
        wiresmap=wiresmap,
        maplist=maplist,
        approx_hessian=approx_hessian,
        approx_gradient=approx_gradient,
        approx_eval=approx_eval,
        alpha_s=alpha_s,
        alpha_x=0.0,
        alpha_y=0.0,
        alpha_z=0.0,
        **kwargs
    )

    if cell_weights_list is not None:
        reg.objfcts[0].cell_weights = np.hstack(clwhtlst)

    if isinstance(alpha_x, float):
        alph_x = alpha_x * np.ones(len(wrmp.maps))
    else:
        alph_x = alpha_x

    if isinstance(alpha_y, float):
        alph_y = alpha_y * np.ones(len(wrmp.maps))
    else:
        alph_y = alpha_y

    if isinstance(alpha_z, float):
        alph_z = alpha_z * np.ones(len(wrmp.maps))
    else:
        alph_z = alpha_z

    for i, (wire, maps) in enumerate(zip(wrmp.maps, mplst)):
        reg += Tikhonov(
            mesh=mesh,
            mapping=maps * wire[1],
            alpha_s=0.0,
            alpha_x=alph_x[i],
            alpha_y=alph_y[i],
            alpha_z=alph_z[i],
            cell_weights=clwhtlst[i],
            **kwargs
        )

    return reg

class GaussianMixtureMarkovRandomField(WeightedGaussianMixture):

    def __init__(
        self,
        n_components,
        mesh,
        actv=None,
        kdtree=None, indexneighbors=None,
        boreholeidx=None,
        T=12., kneighbors=0, norm=2,
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
        self.norm = norm

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
            _, self.indexneighbors = self.kdtree.query(self.xyz, k=self.kneighbors+1, p=self.norm)
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
            unit_index = []
            for i in range(self.n_components):
                unit_index.append(np.where(prediction==i)[0])
            for i, unitindex in enumerate(unit_index):
                _, self.indexpoint[unitindex] = self.index_kdtree[i].query(
                    self.unitxyz[i][unitindex],
                    k=self.kneighbors+1,
                    p=self.index_anisotropy['norm'][i]
                )

        logG = (self.T/(2.*(self.kneighbors+1))) * (
            (z[self.indexpoint] + w[self.indexpoint]).sum(
                axis=1
            )
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
        #T=12., kneighbors=0, norm=2,
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
            unit_index = []
            for i in range(self.n_components):
                unit_index.append(np.where(prediction==i)[0])
            for i, unitindex in enumerate(unit_index):
                _, self.indexpoint[unitindex] = self.index_kdtree[i].query(
                    self.unitxyz[i][unitindex],
                    k=self.kneighbors+1,
                    p=self.index_anisotropy['norm'][i]
                )

        logG = (self.T/(2.*(self.kneighbors+1))) * (
            (z[self.indexpoint] + w[self.indexpoint]).sum(
                axis=1
            )
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
