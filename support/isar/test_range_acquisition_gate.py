import unittest
import numpy as np
from full_ship_field import select_range
from test_full_path_point import problem


class TestRangeGate(unittest.TestCase):
    def test_raw_selection_covariance_and_carrier_origin_remain_consistent(self):
        b=problem();f=b.field;small=select_range(f,-5,3);keep=(f.offsets>=-5)&(f.offsets<=3)
        np.testing.assert_array_equal(small.data['value'],f.data['value'][:,keep])
        for key in ['row_time','row_information','centre_path_m','centre_ecef_m']:np.testing.assert_array_equal(small.data[key],f.data[key])
        lag=f.offsets[None,:]-.27;raw=f.pulse(lag);a=f.apply(raw);s=small.apply(raw[:,keep])
        covariance=np.linalg.inv(f.whiten)@np.linalg.inv(f.whiten).conj().T;subset=covariance[np.ix_(keep,keep)]
        expected=np.einsum('tr,rs,ts->',raw[:,keep].conj()*f.root_information[:,None],np.linalg.inv(subset),raw[:,keep]*f.root_information[:,None]).real
        self.assertLess(abs(np.vdot(s,s).real-expected)/expected,1e-12)
        self.assertLessEqual(np.vdot(s,s).real,np.vdot(a,a).real*(1+1e-12))


if __name__=='__main__':unittest.main()
