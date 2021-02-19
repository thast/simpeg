from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import copy
from ..regularization import (
    SimpleSmall,
    Small,
    SparseSmall,
    Simple,
    Tikhonov,
    Sparse,
    SimplePGIsmallness,
    PGIsmallness,
    SimplePGIwithNonlinearRelationshipsSmallness,
    SimplePGI,
    PGI,
    SmoothDeriv,
    SimpleSmoothDeriv,
    SparseDeriv,
    SimplePGIwithRelationships,
)
from ..utils import (
    mkvc,
    WeightedGaussianMixture,
    GaussianMixtureWithPrior,
    GaussianMixtureWithNonlinearRelationships,
    GaussianMixtureWithNonlinearRelationshipsWithPrior,
    Zero,
    ICM_PottsDenoising,
    GibbsSampling_PottsDenoising
)
from ..directives import InversionDirective, MultiTargetMisfits
from ..utils.code_utils import deprecate_property

class PGI_GMMRF_IsingModel(InversionDirective):

    neighbors = None
    distance = 2
    weigthed_random_walk = True
    compute_score = False
    maxit = None
    verbose = False
    method = 'ICM'  # 'Gibbs'
    offdiag = 0.
    indiag = 1.
    Pottmatrix = None
    log_univar = None
    anisotropies = None

    def endIter(self):
        if self.invProb.reg.mrefInSmooth:
            if self.verbose:
                print("No more Ising update when mref is in Smooth")
        else:
            mesh = self.invProb.reg._mesh
            if self.neighbors is None:
                self.neighbors = 2 * mesh.dim

            m = self.invProb.model
            modellist = self.invProb.reg.wiresmap * m
            model = np.c_[
                [a * b for a, b in zip(self.invProb.reg.maplist, modellist)]].T
            minit = self.invProb.reg.gmm.predict(model)

            indActive = self.invProb.reg.regmesh.indActive

            if self.Pottmatrix is None:
                n_unit = self.invProb.reg.gmm.n_components
                Pott = np.ones([n_unit, n_unit]) * self.offdiag
                for i in range(Pott.shape[0]):
                    Pott[i, i] = self.indiag
                self.Pottmatrix = Pott

            # if self.log_univar is None:
            _, self.log_univar = self.invProb.reg.gmm._estimate_log_prob_resp(
                model
            )

            if self.method == 'Gibbs':
                denoised = GibbsSampling_PottsDenoising(
                    mesh, minit,
                    self.log_univar,
                    self.Pottmatrix,
                    indActive=indActive,
                    neighbors=self.neighbors,
                    norm=self.distance,
                    weighted_selection=self.weigthed_random_walk,
                    compute_score=self.compute_score,
                    maxit=self.maxit,
                    verbose=self.verbose,
                    anisotropies=self.anisotropies
                )
            elif self.method == 'ICM':
                denoised = ICM_PottsDenoising(
                    mesh, minit,
                    self.log_univar,
                    self.Pottmatrix,
                    indActive=indActive,
                    neighbors=self.neighbors,
                    norm=self.distance,
                    weighted_selection=self.weigthed_random_walk,
                    compute_score=self.compute_score,
                    maxit=self.maxit,
                    verbose=self.verbose,
                    anisotropies=self.anisotropies
                )

            self.invProb.reg.mref = mkvc(
                self.invProb.reg.gmm.means_[denoised[0]]
            )
