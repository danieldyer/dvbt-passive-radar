import unittest
import numpy as np
from audit_ship_residual_histories import common_path
from spline_motion import basis


class ResidualHistoryTests(unittest.TestCase):
    def test_shared_physical_phase_at_independent_band_epochs_and_unused_points(self):
        time=[np.linspace(.013*c,12-.017*c,401-c*7) for c in range(5)]
        information=[1+.25*np.sin(t*.3+c) for c,t in enumerate(time)]
        selected=[np.floor((t-.2)/.5).astype(int)%2==0 for t in time]
        k=2*np.pi*np.array([177.5,184.5,191.625,219.5,226.5])/299.792458
        interval=[min(t[0] for t in time),max(t[-1] for t in time)]
        b=[basis(t,interval,2.) for t in time]
        x=[np.c_[np.ones(len(t)),t-np.mean(interval)] for t in time]
        w=[i*s/np.mean(i[s]) for i,s in zip(information,selected)]
        projection=np.linalg.solve(sum(a.T@(i[:,None]*a) for a,i in zip(x,w)),sum(a.T@(i[:,None]*h) for a,i,h in zip(x,w,b)))
        coefficients=.15*np.sin(np.arange(b[0].shape[1])*.7)
        truth=[(h-a@projection)@coefficients for h,a in zip(b,x)]
        amplitude=np.array([1+.4j,.6j,-.7+.2j,.4-.3j])
        values=[np.exp(-1j*kc*p[:,None])*amplitude[None]*np.exp(.2j*c) for c,(kc,p) in enumerate(zip(k,truth))]
        path,report=common_path(time,values,information,selected,k,[0,1,2])
        self.assertTrue(report['converged'])
        for found,true in zip(path,truth):np.testing.assert_allclose(found,true,atol=2e-4)
        # A fourth scatterer and all validation values cannot drive the fit.
        changed=[z.copy() for z in values]
        for z,s in zip(changed,selected):
            z[~s]*=np.exp(2.8j);z[:,3]*=np.exp(2j*np.arange(len(z)))
        second,_=common_path(time,changed,information,selected,k,[0,1,2])
        for first,other in zip(path,second):np.testing.assert_allclose(first,other,atol=1e-13)


if __name__=='__main__':unittest.main()
