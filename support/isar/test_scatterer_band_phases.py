import unittest
import numpy as np
from scipy.sparse import csr_matrix
from audit_ship_scattering_envelopes import linear_basis
from audit_scatterer_band_phases import fit_phases
from sparse_positive_envelope import Normal
from rigid_patch import profile_static
from full_path_scene import solve_normal_phases


class ConstantSourcePhaseTests(unittest.TestCase):
    def test_fixed_relative_band_calibration_retains_common_source_phase_gauge(self):
        rng=np.random.default_rng(652);t=np.linspace(0,12,181)
        basis,_=linear_basis(t,[0,12],3.);basis=csr_matrix(basis)
        offsets=np.array([.2,-.7,1.1,0.,-1.3]);phase=np.array([.3,.8]);normals=[];groups=[]
        for c in range(5):
            h=rng.normal(size=(len(t),5,2))+1j*rng.normal(size=(len(t),5,2))
            coefficients=rng.uniform(.3,1.2,(basis.shape[1],2));root=np.ones(len(t));train=np.arange(len(t))%3!=2
            y=np.sum(h*((basis@coefficients)*np.exp(1j*(offsets[c]+phase)))[:,None,:],axis=-1)+root[:,None]*(.3+.2j)
            normals.append(Normal(h,y,root,train,basis));groups.append(np.tile(np.arange(2),basis.shape[1]))
        initial=np.r_[offsets+.45,.6]
        fitted,solutions,report=solve_normal_phases(normals,initial,groups,100,fixed_band_offsets=offsets)
        self.assertTrue(report['numerical_convergence'],report)
        np.testing.assert_allclose(fitted[:5]-fitted[3],offsets,atol=1e-14,rtol=0)
        self.assertLess(max(abs(fitted[:5]-offsets-.3)),1e-6)
        self.assertAlmostEqual(fitted[5],.5,places=6)
        for normal,solution in zip(normals,solutions):self.assertLess(abs(solution['cost'])/normal.energy,1e-12)

    def test_constant_source_phases_recover_excluded_data_without_phase_history(self):
        rng=np.random.default_rng(651);t=np.linspace(0,12,241)
        basis,_=linear_basis(t,[0,12],3.);basis=csr_matrix(basis)
        h=rng.normal(size=(len(t),7,3))+1j*rng.normal(size=(len(t),7,3))
        phase=np.array([.2,-.7,1.1]);coefficient=rng.uniform(.5,1.5,(basis.shape[1],3))
        root=rng.uniform(.8,1.2,len(t));groups=(t>=6).astype(int);train=np.arange(len(t))%3!=2
        source=np.sum(h*((basis@coefficient)*np.exp(1j*phase))[:,None,:],axis=-1)
        data=source+root[:,None]*(.3+.1j)
        normal=Normal(h,data,root,train,basis,groups)
        actual,solution,report=fit_phases(normal,phase+np.array([.2,-.3,.4]),np.tile(np.arange(3),basis.shape[1]),100)
        self.assertLess(max(abs(np.angle(np.exp(1j*(actual-phase))))),1e-6)
        amplitude=basis@solution['amplitude'].reshape(-1,3)
        prediction=np.sum(h*(amplitude*np.exp(1j*actual))[:,None,:],axis=-1)
        residual=profile_static(prediction-data,root,~train,groups)[~train]
        self.assertLess(np.linalg.norm(residual)/np.linalg.norm(source[~train]),1e-6)
        self.assertLess(report['amplitude_kkt'],1e-6)


if __name__=='__main__':unittest.main()
