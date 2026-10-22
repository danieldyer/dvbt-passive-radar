import unittest
import numpy as np
from field import average
from recorded_injection_fields import common_projection
from test_rigid_motion import fixture


class CommonProjectionTests(unittest.TestCase):
    def test_common_projection_preserves_complex_phase_weights_and_erasures(self):
        f=fixture();d=f.data;n=len(d['row_time'])
        amplitude=np.exp(.17j*d['row_time'])*(1+.02*d['row_time'])
        # A constant range profile makes the interpolation's exact DC response
        # known independently, while carrier deramp and weighted binning remain.
        raw=amplitude*np.exp(-1j*f.k*d['centre_path_m'])
        values=np.zeros((2,n,192),complex);values[0]=raw[:,None]
        rows=np.zeros((n,6));rows[:,1]=d['row_time']
        ch=dict(values=values,rows=rows)
        first=int(np.floor(d['centre_path_m'].min()*8e6/299792458.))-90
        expected=f.apply(np.repeat(average(amplitude,d['row_information'],4)[0][:,None],len(f.offsets),axis=1)).ravel()
        actual=common_projection(f,ch,first)
        np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-10)
        phase=np.exp(.83j);ch['values']=values*phase
        np.testing.assert_allclose(common_projection(f,ch,first),actual*phase,rtol=2e-12,atol=2e-10)


if __name__=='__main__':unittest.main()
