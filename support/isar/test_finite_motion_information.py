import unittest
import numpy as np
from finite_reference import echo
from finite_motion_information import echo_jacobian, local_bias


class FiniteMotionInformationTests(unittest.TestCase):
    def test_physical_echo_derivatives(self):
        rng=np.random.default_rng(670);s,p=rng.normal(size=(2,129))+1j*rng.normal(size=(2,129))
        h=np.exp(.2*rng.normal(size=129)+.2j*rng.normal(size=129))
        origin,delay,fd=13.123,28.354,13.8
        q,d,f=echo_jacobian(s,p,h,origin,delay,fd,nfft=256,rate=2560)
        np.testing.assert_allclose(q,echo(s,p,h,origin,delay,fd,nfft=256,rate=2560),atol=1e-14)
        for index,analytic in [(0,d),(1,f)]:
            x=[delay,fd];a=x.copy();b=x.copy();a[index]+=1e-5;b[index]-=1e-5
            numeric=(echo(s,p,h,origin,*a,nfft=256,rate=2560)-echo(s,p,h,origin,*b,nfft=256,rate=2560))/(2e-5)
            np.testing.assert_allclose(analytic,numeric,atol=2e-9,rtol=2e-8)

    def test_amplitude_nuisance_does_not_bias_motion(self):
        rng=np.random.default_rng(419);s=rng.normal(size=200)+1j*rng.normal(size=200)
        j=rng.normal(size=(200,2))+1j*rng.normal(size=(200,2))
        np.testing.assert_allclose(local_bias((.8+.4j)*s,s,j),0,atol=1e-15)
        delta=np.array([1e-6,-2e-6])
        np.testing.assert_allclose(local_bias((.8+.4j)*(s+j@delta),s,j),delta,rtol=1e-5)


if __name__=='__main__':unittest.main()
