import unittest
import numpy as np
from native_folds import native_indices,training_mask,fitting_mask,fit_sample_selection


class NativeFoldTests(unittest.TestCase):
    def test_new_fits_use_all_samples_and_legacy_replays_keep_their_likelihood(self):
        time=np.array([1.,1.29,1.51,2.3,3.1])
        self.assertTrue(np.all(fitting_mask(time,1.)))
        self.assertEqual(fit_sample_selection(),'all')
        self.assertEqual(fit_sample_selection(saved={}), 'all')
        self.assertEqual(fit_sample_selection(saved={},reproduce=True),'diagnostic-subset')
        self.assertEqual(fit_sample_selection(saved={'fit_samples':'all'},reproduce=True),'all')
        np.testing.assert_array_equal(fitting_mask(time,1.,'diagnostic-subset'),training_mask(time,1.))
        with self.assertRaises(ValueError):fit_sample_selection('all',saved={},reproduce=True)
        with self.assertRaises(ValueError):fitting_mask(time,1.,'unknown')

    def test_periodic_unused_samples_introduce_the_expected_one_hz_grating_lobe(self):
        time=(np.arange(32768)+.5)/4096
        phasor=np.exp(2j*np.pi*time)
        omitted=~training_mask(time,0.)
        whole=abs(phasor.mean())**2
        periodic=abs(phasor[omitted].mean())**2
        self.assertLess(whole,1e-28)
        expected=(np.sin(np.pi*.25)/(1024*np.sin(np.pi/4096)))**2
        self.assertAlmostEqual(periodic,expected,places=13)

    def test_folds_partition_complete_weighted_bins_without_splitting_symbols(self):
        rng=np.random.default_rng(813);n=16*91+7
        rows=np.zeros((n,6));rows[:,1]=157.97+np.arange(n)*.0043;rows[:,4]=1
        information=rng.uniform(.2,4.,n);rows[16*35:16*36,4]=0
        first,last=158.,163.;choices={name:native_indices(rows,information,first,last,name) for name in ('all','training','testing')}
        np.testing.assert_array_equal(np.sort(np.r_[choices['training'],choices['testing']]),choices['all'])
        self.assertEqual(len(np.intersect1d(choices['training'],choices['testing'])),0)
        for fold in ('training','testing'):
            for block in choices[fold].reshape(-1,16):
                np.testing.assert_array_equal(block,np.arange(block[0],block[0]+16))
                w=information[block]*rows[block,4]
                t=np.average(rows[block,1],weights=w) if w.sum() else rows[block,1].mean()
                self.assertEqual(bool(training_mask([t],first)[0]),fold=='training')
        self.assertTrue(np.all(choices['all']<n-7))


if __name__=='__main__':unittest.main()
