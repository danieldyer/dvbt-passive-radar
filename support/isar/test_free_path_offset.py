import unittest
import numpy as np
from scipy.optimize import minimize
from test_full_path_point import problem
from path_dynamics import constraint
from rigid_patch import profile_static


class FreePathOffsetTests(unittest.TestCase):
    def test_absolute_envelope_delay_is_independent_of_phase_gauge(self):
        b=problem(free_path_offset=True);old=b.b.shape[1]-1
        # Match the real field's constant carrier origin. A moving centre
        # deramp would make this moving test point almost stationary in the
        # digital frame, where constant-profile projection removes its power.
        b.field.data=dict(b.field.data)
        b.field.data['carrier_reference_path_m']=np.full(len(b.t),b.field.data['centre_path_m'][0])
        truth=np.r_[np.zeros(old),3.]
        h=b.waveform(truth,3,False);phase=.3
        b.field.y=h*np.exp(1j*phase)+b.field.root_information[:,None]*(.1+.2j)
        for train in [True,False]:
            mask=b.training if train else ~b.training
            y=profile_static(b.field.y,b.field.root_information,mask,b.groups)*mask[:,None]
            b.data[train]=y;b.energy[train]=np.vdot(y,y).real
        initial_phase=phase-b.field.k*3.
        def objective(x):
            path=np.r_[np.zeros(old),x[0]]
            actual_phase=initial_phase+b.field.k*x[0]+x[1]
            value=b.evaluate(path,actual_phase,3)
            gradient=value['gradient']
            return value['cost']/value['energy'],np.array([gradient[old]+b.field.k*gradient[-1],gradient[-1]])/value['energy']
        result=minimize(objective,[0.,0.],jac=True,method='BFGS',options=dict(gtol=1e-11,maxiter=100))
        self.assertLess(abs(result.x[0]-3.),1e-5)
        self.assertLess(result.fun,1e-14)
        self.assertGreater(objective(np.array([0.,0.]))[0],1e-4)
        # A constant physical delay changes the range envelope. A free
        # constant phase cannot repair that mismatch at the coarse delay.
        self.assertEqual(float((b.bd@truth).max()),0.)

    def test_constant_path_offset_changes_bounds_but_not_acceleration(self):
        grid=np.r_[np.repeat(0.,4),4.,8.,12.,np.repeat(16.,4)]
        base=np.linspace(100.,101.,len(grid)-4)
        n=len(base)-1
        fixed=constraint(grid,base,n+5,.2,10.,8.)
        free=constraint(grid,base,n+1+5,.2,10.,8.,True)
        acceleration_rows=len(free.lb)-len(base)
        np.testing.assert_array_equal(free.A[:acceleration_rows,n],0.)
        np.testing.assert_array_equal(free.A[acceleration_rows:,n],1.)
        np.testing.assert_array_equal(free.A[:,:n],fixed.A[:,:n])


if __name__=='__main__':unittest.main()
