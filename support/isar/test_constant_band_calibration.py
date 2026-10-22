import unittest
import numpy as np
from scipy.optimize import minimize,root
from moment_phase_scene import MomentPhaseScene
from constant_band_calibration import ConstantBandCalibration
import test_profiled_phase_scene as fixture


class ConstantBandCalibrationTests(unittest.TestCase):
    def test_constant_band_phases_gradient_and_unknown_image_recovery(self):
        base,truth,x=fixture.ProfiledPhaseSceneTests().problem()
        phase=np.array([.15,-.2,.12,0.,-.18])
        data=[y*np.exp(1j*p) for y,p in zip(base.data,phase)]
        model=MomentPhaseScene(base.design,data,base.root,base.groups,base.selected,base.basis,base.k,base.strength,base.spectrum)
        calibrated=ConstantBandCalibration(model);at=np.r_[truth*.4,phase[calibrated.channels]*.4]
        value,gradient=calibrated.objective(at)
        step=1e-6
        fd=[(calibrated.objective(at+step*e)[0]-calibrated.objective(at-step*e)[0])/(2*step) for e in np.eye(len(at))]
        np.testing.assert_allclose(gradient,fd,rtol=5e-7,atol=2e-9)
        result=minimize(calibrated.objective,np.zeros(len(at)),jac=True,method='BFGS',options=dict(gtol=1e-10,maxiter=100))
        # At the noiseless solution the normalized cost is near machine
        # precision. Polish its analytic stationary equations instead of
        # asking a line search to resolve smaller cost differences.
        stationary=root(lambda p:calibrated.objective(p)[1],result.x,options=dict(xtol=1e-10))
        value,gradient,report=calibrated.objective(stationary.x,True)
        self.assertLess(max(abs(gradient)),1e-12)
        np.testing.assert_allclose(stationary.x,np.r_[truth,phase[calibrated.channels]],rtol=0,atol=1e-8)
        np.testing.assert_allclose(report['coefficient'],x,rtol=0,atol=1e-8)
        excluded=model.score(stationary.x[:3],report['coefficient'],[~mask for mask in model.selected])
        self.assertLess(excluded['cost']/excluded['energy'],1e-14)


if __name__=='__main__':unittest.main()
