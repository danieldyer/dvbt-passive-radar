import unittest
import numpy as np

from omega_k import BistaticOmegaK,ParallelGeometry


def problem(sign=1,u_offset=0.):
    g=ParallelGeometry(np.array([-35000.*sign,9000.,700.]),np.array([-8000.*sign,-2500.,30.]))
    u=np.arange(-64,65)*1.+u_offset;k=np.linspace(3.925,4.075,24)
    return BistaticOmegaK(g,u,k,maximum_along_offset_m=100.)


class OmegaKTests(unittest.TestCase):
    def test_numerical_reference_matches_independent_two_leg_bp(self):
        m=problem();points=np.array([[0.,0.],[25.,-31.],[-45.,52.]])
        data=np.zeros((len(m.u),len(m.k)),complex)
        for (x,y),gain in zip(points,[1.,.7+.2j,-.2+.4j]):
            pos=np.c_[np.full(len(m.u),x),m.u+y,np.zeros(len(m.u))]
            path=sum(np.sqrt(np.sum((pos-ant)**2,axis=1)) for ant in (m.geometry.transmitter,m.geometry.receiver))
            path-=np.linalg.norm(m.geometry.transmitter-m.geometry.receiver)
            data+=gain*np.exp(-1j*(path-m.reference)[:,None]*m.k)
        image,y=m.reference_image(data,points[:,0])
        for c,p in enumerate(points):
            value=image[np.argmin(abs(y-p[1])),c]
            self.assertLess(abs(value-m.direct(data,[p])[0])/abs(value),2e-10)

    def test_stolt_operator_weights_and_true_snr_for_both_axis_directions(self):
        for sign in (-1,1):
            m=problem(sign);x=np.array([-40.,0.,25.]);points=[(0.,0.),(25.,-31.)]
            target=points[1];data=np.exp(-1j*(m.geometry.path(target[0],m.u+target[1])-m.reference)[:,None]*m.k)
            image,y,meta=m.stolt_image(data,x,weight_points=points,range_period_m=800.)
            for px,py in points:
                w=meta['operator_weights'][f'{px:g},{py:g}'];value=image[np.argmin(abs(y-py)),np.argmin(abs(x-px))]
                self.assertLess(abs(np.sum(w*data)-value)/max(abs(value),1e-100),2e-10)
            w=meta['operator_weights']['25,-31']
            efficiency=abs(np.sum(w*data))**2/np.sum(abs(w)**2)/np.sum(abs(data)**2)
            self.assertLess(-10*np.log10(efficiency),.5)

    def test_nonzero_epoch_offset_has_the_same_bp_coordinates(self):
        m=problem(u_offset=.17);p=(25.,-31.)
        data=np.exp(-1j*(m.geometry.path(p[0],m.u+p[1])-m.reference)[:,None]*m.k)
        image,y=m.reference_image(data,[p[0]])
        value=image[np.argmin(abs(y-p[1])),0];expected=m.direct(data,[p])[0]
        self.assertLess(abs(value-expected)/abs(expected),2e-5)


if __name__=='__main__':unittest.main()
