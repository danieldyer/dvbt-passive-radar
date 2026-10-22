import unittest

import numpy as np

from profiled_arc_solver import ProfiledArcSolver,band_residual
from scipy.optimize import nnls
from test_common_physical_arc import fixture


class ProfiledArcSolverTests(unittest.TestCase):
    def test_adjoint_profile_matches_explicit_amplitude_derivative_tensor(self):
        common,truth=fixture();p=truth+np.sin(np.arange(len(truth)))*.0002
        mapped,_=common.mapping(p)
        for band in common.independent.bands:
            raw,derivative=band.response(mapped);basis=band.amplitude_basis
            design=band.nuisance.remove(band.reduced(raw[:,:,None]*basis[:,None,:])[band.selected])
            matrix=np.r_[design.real,design.imag];data=np.r_[band.clean.real,band.clean.imag]
            amplitude,_=nnls(matrix,data);residual=data-matrix@amplitude
            active=np.flatnonzero(amplitude>0);changes=[]
            for index in active:
                value=band.nuisance.remove(band.reduced(derivative*basis[:,None,index,None])[band.selected])
                changes.append(np.r_[value.real,value.imag])
            changes=np.stack(changes,axis=1);matrix=matrix[:,active]
            inverse=np.linalg.pinv(matrix,rcond=1e-10)
            mean=np.einsum('mkp,k->mp',changes,amplitude[active])
            expected=-mean+matrix@(inverse@mean)-inverse.T@np.einsum('mkp,m->kp',changes,residual)
            actual,jac=band_residual(band,mapped)
            np.testing.assert_allclose(actual,residual,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(jac,expected,atol=1e-10,rtol=1e-10)

    def test_profiled_residual_and_gradient_are_the_original_likelihood(self):
        common,truth = fixture();solver = ProfiledArcSolver(common)
        p = truth+np.sin(np.arange(len(truth)))*.0002
        residual,jacobian = solver.evaluate(p)
        value,gradient = common.objective(p)
        self.assertAlmostEqual(residual@residual,value,places=13)
        np.testing.assert_allclose(2*jacobian.T@residual,gradient,rtol=3e-8,atol=1e-10)

    def test_full_residual_jacobian_with_nonzero_model_error(self):
        common,truth = fixture()
        rng = np.random.default_rng(919)
        # Keep the original model and inject error before rebuilding the
        # stationary projection; the derivative must include amplitude changes.
        from physical_amplitude_arc import PhysicalAmplitudeArc,PhysicalAmplitudeFit
        from common_physical_arc import CommonPhysicalFit
        bands=[]
        for b in common.independent.bands:
            b.field.y += .005*(rng.normal(size=b.field.y.shape)+1j*rng.normal(size=b.field.y.shape))
            bands.append(PhysicalAmplitudeArc(b.field,b.training,b.knots,b.geometry,b.channel,b.midpoint))
        common=CommonPhysicalFit(PhysicalAmplitudeFit(bands));solver=ProfiledArcSolver(common)
        p=truth+np.sin(np.arange(len(truth)))*.0002
        residual,jacobian=solver.evaluate(p)
        direction=np.cos(np.arange(len(p)))*.01;step=2e-5
        finite=(solver.evaluate(p+step*direction)[0]-solver.evaluate(p-step*direction)[0])/(2*step)
        self.assertLess(np.linalg.norm(finite-jacobian@direction)/np.linalg.norm(finite),3e-5)
        _,gradient=common.objective(p)
        np.testing.assert_allclose(2*jacobian.T@residual,gradient,rtol=3e-8,atol=1e-10)


if __name__=='__main__':unittest.main()
