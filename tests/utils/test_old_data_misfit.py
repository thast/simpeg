from __future__ import print_function

import unittest

import numpy as np
import scipy.sparse as sp

import discretize as Mesh
from SimPEG import data_misfit as DataMisfit
from SimPEG import maps as Maps
from SimPEG import utils as Utils

# from SimPEG import Mesh, DataMisfit, Maps, Utils
# from SimPEG.EM.Static import DC
from SimPEG.electromagnetics.static import resistivity as DC

np.random.seed(17)


class DataMisfitTest(unittest.TestCase):
    def setUp(self):
        mesh = Mesh.TensorMesh([30, 30], x0=[-0.5, -1.0])
        sigma = np.ones(mesh.nC)
        model = np.log(sigma)

        prob = DC.Problem3D_CC(mesh, rhoMap=Maps.ExpMap(mesh))

        rx = DC.Rx.Pole(Utils.ndgrid([mesh.vectorCCx, np.r_[mesh.vectorCCy.max()]]))
        src = DC.Src.Dipole(
            [rx], np.r_[-0.25, mesh.vectorCCy.max()], np.r_[0.25, mesh.vectorCCy.max()]
        )
        survey = DC.Survey([src])

        prob.pair(survey)

        self.std = 0.01
        survey.std = self.std
        dobs = survey.makeSyntheticData(model)
        self.eps = 1e-8 * np.min(np.abs(dobs))
        survey.eps = self.eps
        dmis = DataMisfit.l2_DataMisfit(survey)

        self.model = model
        self.mesh = mesh
        self.survey = survey
        self.prob = prob
        self.dobs = dobs
        self.dmis = dmis

    def test_Wd_depreciation(self):
        with self.assertWarns(FutureWarning):
            self.dmis.Wd

        with self.assertWarns(FutureWarning):
            self.dmis.Wd = Utils.Identity()

    def test_DataMisfit_nP(self):
        self.assertTrue(self.dmis.nP == self.mesh.nC)

    def test_setting_W(self):
        Worig = self.dmis.W
        v = np.random.rand(self.survey.nD)

        self.dmis.W = v
        self.assertTrue(self.dmis.W.shape == (self.survey.nD, self.survey.nD))
        self.assertTrue(np.all(self.dmis.W.diagonal() == v))

        with self.assertRaises(Exception):
            self.dmis.W = np.random.rand(self.survey.nD + 10)

        self.dmis.W = Worig

    def test_DataMisfitOrder(self):
        self.dmis.test(x=self.model)

    def test_std_eps(self):
        stdtest = np.all(self.survey.std == self.dmis.std)
        epstest = np.all(self.survey.eps == self.dmis.eps)
        Wtest = np.allclose(
            np.abs(np.dot(self.dmis.W.todense(), self.dobs)),
            1.0 / self.std,
            atol=self.eps,
        )

        self.assertTrue(stdtest)
        self.assertTrue(epstest)
        self.assertTrue(Wtest)


if __name__ == "__main__":
    unittest.main()
