import unittest
import numpy as np

from higher_order_autofocus import HigherOrderPhase


class HigherOrderAutofocusTests(unittest.TestCase):
    def test_analytic_gradient_and_motion_constraint(self):
        t=np.linspace(-15,15,181);train=np.arange(len(t))%4!=3;w=1+.2*np.sin(t)
        z=np.exp(.5j*np.sin(t/8))+.1*np.cos(t)
        model=HigherOrderPhase(t,z,w,train);p=.1*np.sin(np.arange(6))
        value,g=model.objective(p)
        for i in range(6):
            step=np.eye(6)[i]*1e-5
            finite=(model.objective(p+step)[0]-model.objective(p-step)[0])/2e-5
            self.assertAlmostEqual(finite,g[i],places=8)
        low=np.column_stack([np.ones(len(t)),t,t*t])
        np.testing.assert_allclose(low.T@(model.weight[:,None]*model.basis),0,atol=2e-12)

    def test_known_higher_phase_recovery_and_unused_data_independence(self):
        t=np.linspace(-15,15,181);train=np.arange(len(t))%4!=3;w=np.ones(len(t))
        reference=HigherOrderPhase(t,np.ones(len(t)),w,train)
        truth=reference.basis@np.array([1.5,.6,.1,0.,0.,0.]);z=(.8+.3j)*np.exp(1j*truth)
        phase,a=HigherOrderPhase(t,z,w,train).fit()
        self.assertLess(np.max(abs(phase-truth)),1e-6)
        changed=z.copy();changed[~train]*=np.exp(.37j)
        other,b=HigherOrderPhase(t,changed,w,train).fit()
        np.testing.assert_array_equal(phase,other)
        self.assertLess(a['maximum_gradient'],1e-7)


if __name__=='__main__':unittest.main()
