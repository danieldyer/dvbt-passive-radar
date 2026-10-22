import unittest
import numpy as np
from training_aperture import select


class NativeTrainingFoldTests(unittest.TestCase):
    def test_support_and_fold_follow_native_intervals_not_approximate_mean_epochs(self):
        time=3.+np.arange(16*9)*.001088;information=np.ones(len(time));information[48:64]=0
        # A weighted block mean could move to one edge; identities still
        # follow actual first/last epochs, never a nearest-mean heuristic.
        information[16]=10000;training=np.arange(9)%2==0
        first=time.reshape(-1,4)[:,0];last=time.reshape(-1,4)[:,-1]
        expected=np.repeat(training,4)&np.repeat(information.reshape(-1,16).sum(axis=1)>0,4)
        np.testing.assert_array_equal(select(first,last,time,information,training),expected)
        self.assertFalse(select(np.array([time[14]]),np.array([time[17]]),time,information,training)[0])
        self.assertFalse(select(np.array([2.]),np.array([2.003]),time,information,training)[0])

if __name__=='__main__':unittest.main()
