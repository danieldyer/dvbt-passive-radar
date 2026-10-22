import unittest
import numpy as np
from discover_coherent_ship_arcs import scores


class DiscoveryTests(unittest.TestCase):
    def test_profiled_tone_is_invariant_to_strong_static_return(self):
        time=np.linspace(-1.5,1.5,151);weight=1+.2*np.sin(time)
        k=4.;rates=np.linspace(-2,2,81);truth=.35
        phase=.7*time+.1*time**2
        moving=(.7+.4j)*np.exp(-1j*k*truth*time)
        clean,retained=scores(moving,weight,time,phase,k,rates,1.)
        dirty,_=scores(moving+(40-30j)*np.exp(1j*phase),weight,time,phase,k,rates,1.)
        np.testing.assert_allclose(clean,dirty,rtol=2e-11,atol=2e-10)
        self.assertAlmostEqual(rates[np.argmax(clean)],truth)
        self.assertGreater(retained[np.argmax(clean)],.9)

    def test_zero_information_and_stationary_degeneracy_are_explicit(self):
        time=np.linspace(-1,1,51);one=np.ones(len(time))
        with self.assertRaises(ValueError):scores(one,one*0,time,time,4.,np.array([0.]),1.)
        value,retained=scores(one,one,time,time*0,4.,np.array([0.]),1.)
        self.assertLess(retained[0],1e-12);self.assertEqual(value[0],0.)


if __name__=='__main__':unittest.main()
