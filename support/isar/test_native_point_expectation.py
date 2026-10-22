import unittest
import numpy as np
from prepare_native_point_expectation import paired_support


class NativePointExpectationTests(unittest.TestCase):
    def test_short_derived_export_cannot_reset_original_native_groups(self):
        rows=np.zeros((100,6));rows[:,0]=np.arange(100)*8704;rows[:,1]=np.arange(100)*.001;rows[:,4]=1
        baseline=dict(rows=rows,information=1+np.arange(100)*.01,pulse=np.repeat(np.arange(100)[:,None],129,axis=1).astype(complex))
        linear={key:value[12:88].copy() for key,value in baseline.items()}
        use,at=paired_support(linear,baseline,.020,.080)
        np.testing.assert_array_equal(at,np.arange(16,80))
        np.testing.assert_array_equal(use,np.arange(4,68))
        linear['pulse'][20,64]+=1
        with self.assertRaises(AssertionError):paired_support(linear,baseline,.020,.080)


if __name__=='__main__':unittest.main()
