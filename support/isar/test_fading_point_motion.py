import unittest
import numpy as np

from correlated_likelihood import FourierMetric
from fading_point_motion import PlanarGeometry,RadialGeometry,FadingPointArc,FadingPointFit,TurningGeometry,TurningPointArc
from coordinated_turn import trajectory
from recovery_experiment import SCALE
from test_recovery_experiment import problem


class FadingPointMotionTests(unittest.TestCase):
    def test_radial_subspace_preserves_cartesian_rays_and_profiled_gradient(self):
        ex=problem();geometry=RadialGeometry(ex.receiver,ex.rotation,ex.transmitters,ex.state)
        theta=np.array([.1,.2,-.15]);p,v,a,_,_=geometry.state(theta)
        parameters=np.r_[theta,np.zeros(5)];arcs=[]
        for c,f in enumerate(ex.fields):
            dt=f.data['row_time']-ex.midpoint;local=p+dt[:,None]*v+.5*dt[:,None]**2*a
            velocity=v+dt[:,None]*a
            f.y=f.trajectory(ex.receiver+local@ex.rotation,velocity@ex.rotation)*(.8+.3j)
            knots=np.linspace(f.data['row_time'][0],f.data['row_time'][-1],6)
            b=FadingPointArc(f,ex.training[c],knots,geometry,c,ex.midpoint,ex.designs[c])
            parameters[3+c]=np.angle(.8+.3j)-f.k*(geometry.prediction(theta)[c,0]-b.reference_path)
            arcs.append(b)
        model=FadingPointFit(arcs);at=parameters+np.sin(np.arange(8))*.0002
        value,gradient=model.objective(at);direction=np.cos(np.arange(8))*.1;step=2e-5
        finite=(model.objective(at+step*direction)[0]-model.objective(at-step*direction)[0])/(2*step)
        self.assertAlmostEqual(finite,gradient@direction,delta=2e-6+abs(finite)*3e-4)

    def test_fading_operator_matches_independent_point_and_preserves_excluded_data(self):
        ex=problem();geometry=PlanarGeometry(ex.receiver,ex.rotation,ex.transmitters,ex.state)
        p=np.array([.02,-.03,.04,-.02,.01,.005]);theta=p*SCALE/geometry.sigma
        params=np.r_[theta,np.zeros(5)];arcs=[]
        for c,f in enumerate(ex.fields):
            f.y=ex.independent_signal(p,c,[.8+.3j]).reshape(f.y.shape)
            knots=np.linspace(f.data['row_time'][0],f.data['row_time'][-1],6)
            keep=ex.training[c];n=int(keep.sum());r=len(f.offsets)
            metric=FourierMetric(np.zeros(n,int),np.eye(r)[None]*3)
            b=FadingPointArc(f,keep,knots,geometry,c,ex.midpoint,ex.designs[c],metric)
            params[6+c]=np.angle(.8+.3j)-f.k*(geometry.prediction(theta)[c,0]-b.reference_path)
            raw,_=b.response(params)
            predicted=b.reduced(raw[:,:,None])[:,0]*abs(.8+.3j)
            expected=b.fixed_metric.apply(f.y.ravel())
            self.assertLess(np.linalg.norm(predicted-expected)/np.linalg.norm(expected),3e-7)
            # No Fourier coefficient can mix a held-out epoch into training.
            changed=f.y.ravel().copy();changed[np.repeat(~keep,r)]+=1e6*(1+2j)
            np.testing.assert_array_equal(b.fixed_metric.apply(changed).reshape(f.y.shape)[keep],expected.reshape(f.y.shape)[keep])
            arcs.append(b)
        model=FadingPointFit(arcs);at=params+np.sin(np.arange(11))*.0002
        value,gradient=model.objective(at);direction=np.cos(np.arange(11))*.1;step=2e-5
        finite=(model.objective(at+step*direction)[0]-model.objective(at-step*direction)[0])/(2*step)
        self.assertAlmostEqual(finite,gradient@direction,delta=2e-6+abs(finite)*3e-4)

    def test_turning_arc_matches_ray_operator_and_profiled_gradient(self):
        ex=problem();c=3;f=ex.fields[c]
        geometry=TurningGeometry(ex.receiver,ex.rotation,ex.transmitters,ex.state)
        theta=np.array([.02,-.03,.04,-.02,-1.2,-.8]);p,v,a,_,_=geometry.state(theta)
        local,velocity,_=trajectory(f.data['row_time']-ex.midpoint,p,v,a)
        f.y=f.trajectory(ex.receiver+local@ex.rotation,velocity@ex.rotation)*(.8+.3j)
        knots=np.linspace(f.data['row_time'][0],f.data['row_time'][-1],6)
        b=TurningPointArc(f,ex.training[c],knots,geometry,c,ex.midpoint,ex.designs[c])
        params=np.r_[theta,np.zeros(5)]
        params[6+c]=np.angle(.8+.3j)-f.k*(geometry.prediction(theta)[c,0]-b.reference_path)
        raw,j=b.response(params);expected=b.reduced(raw[:,:,None])[:,0]*abs(.8+.3j)
        self.assertLess(np.linalg.norm(expected-f.y.ravel())/np.linalg.norm(f.y),3e-7)
        direction=np.cos(np.arange(11))*.1;step=2e-5
        finite=(b.response(params+step*direction)[0]-b.response(params-step*direction)[0])/(2*step)
        self.assertLess(np.linalg.norm(finite-j@direction)/np.linalg.norm(finite),3e-4)


if __name__=='__main__':unittest.main()
