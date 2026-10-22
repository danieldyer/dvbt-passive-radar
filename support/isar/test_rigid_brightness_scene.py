import unittest
import numpy as np
from rigid_brightness_scene import RigidBrightnessBand,RigidBrightnessFit
from rigid_patch import RigidPatch
from test_recovery_experiment import problem


class RigidBrightnessSceneTests(unittest.TestCase):
    def test_continuous_positions_and_shared_phase_gauge_have_correct_derivatives(self):
        ex=problem();f=ex.fields[1];t=f.data['row_time'];dt=t-ex.midpoint;p,v,a=np.asarray(ex.state)
        position=ex.receiver+(p+dt[:,None]*v+.5*dt[:,None]**2*a)@ex.rotation
        velocity=(v+dt[:,None]*a)@ex.rotation
        kin=RigidPatch(t,ex.midpoint,position,velocity,ex.rotation.T,ex.transmitters[2],ex.receiver)
        points=np.array([[0.,0.,0.],[90.,-30.,0.]]);knots=np.linspace(t[0],t[-1],5)
        gauge=np.array([[2.3,-4.1],[2.4,-4.2]]);f.y[:]=1.
        b=RigidBrightnessBand(f,ex.training[1],knots,kin,points,.4,ex.designs[1],position_gauge=gauge)
        truth=np.array([.03,-.001,.0002,.17,.003,.2,-.5,.13,-.23])
        raw,change=b.responses(truth);direction=np.sin(np.arange(9))*.03;step=2e-5
        upper,_=b.responses(truth+step*direction,False);lower,_=b.responses(truth-step*direction,False)
        for h,u,l in zip(change,upper,lower):
            finite=(u-l)/(2*step)
            self.assertLess(np.linalg.norm(finite-h@direction)/np.linalg.norm(finite),3e-4)
        design=np.concatenate([b.reduced(h[:,:,None]*b.amplitude_basis[:,None,:]) for h in raw],axis=1)
        f.y=(design@(1+.1*np.arange(design.shape[1]))).reshape(f.y.shape)
        b=RigidBrightnessBand(f,ex.training[1],knots,kin,points,.4,ex.designs[1],position_gauge=gauge)
        at=truth+direction*.02;value=b.evaluate_scene(at)
        finite=(b.evaluate_scene(at+step*direction)['rss']-b.evaluate_scene(at-step*direction)['rss'])/(2*step)
        self.assertAlmostEqual(finite,value['gradient']@direction,delta=2e-5+abs(finite)*3e-4)

    def test_native_positive_brightness_profiled_motion_and_shared_phase_gradient(self):
        ex=problem();points=np.array([[0.,0.,0.],[90.,-30.,0.]])
        truth=np.array([.03,-.001,.0002,.17,.003,.2,-.5]);bands=[]
        for c,f in enumerate(ex.fields):
            t=f.data['row_time'];dt=t-ex.midpoint;p,v,a=np.asarray(ex.state)
            position=ex.receiver+(p+dt[:,None]*v+.5*dt[:,None]**2*a)@ex.rotation
            velocity=(v+dt[:,None]*a)@ex.rotation
            kin=RigidPatch(t,ex.midpoint,position,velocity,ex.rotation.T,ex.transmitters[2],ex.receiver)
            knots=np.linspace(t[0],t[-1],5);f.y[:]=1.
            b=RigidBrightnessBand(f,ex.training[c],knots,kin,points,.3*c,ex.designs[c])
            raw,_=b.responses(truth,False)
            columns=np.concatenate([b.reduced(h[:,:,None]*b.amplitude_basis[:,None,:]) for h in raw],axis=1)
            coefficients=1+.3*np.sin(np.arange(columns.shape[1])+c)
            f.y=(columns@coefficients).reshape(f.y.shape)
            bands.append(RigidBrightnessBand(f,ex.training[c],knots,kin,points,.3*c,ex.designs[c]))
        model=RigidBrightnessFit(bands);at=truth+np.array([.0002,-.00003,.00002,.0004,-.00002,.001,-.001])
        value,gradient=model.objective(at);direction=np.cos(np.arange(len(at)))*.03;step=2e-5
        finite=(model.objective(at+step*direction)[0]-model.objective(at-step*direction)[0])/(2*step)
        self.assertAlmostEqual(finite,gradient@direction,delta=2e-6+abs(finite)*3e-4)
        checkpoints=[];fit=model.fit(at,max_iterations=120,progress=checkpoints.append)
        self.assertLess(fit['objective'],1e-9)
        self.assertEqual(len(checkpoints),fit['iterations'])
        np.testing.assert_allclose(checkpoints[-1]['parameters'],fit['parameters'],rtol=0,atol=0)
        information=model.information(truth)
        scale,_=model.settings();finite=[]
        def residual(p):
            pieces=[]
            for b in bands:
                value=b.evaluate_scene(p,full=True)
                r=b.clean-b.nuisance.remove(value['design'][b.selected])@value['amplitude']
                pieces.append(np.r_[r.real,r.imag])
            return np.concatenate(pieces)
        for j in range(model.size):
            delta=np.eye(model.size)[j]*scale[j]*1e-5
            finite.append((residual(truth+delta)-residual(truth-delta))/(2e-5))
        finite=np.column_stack(finite);expected=2*finite.T@finite
        actual=information['fixed_band_calibration']['joint_information_scaled']
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),3e-4)
        uncalibrated=information['unknown_constant_band_phases']['joint_information_scaled']
        self.assertGreater(np.linalg.eigvalsh(actual-uncalibrated).min(),-np.linalg.norm(actual)*1e-10)
        gauge=np.r_[np.zeros(5),np.ones(len(points))]
        self.assertLess(np.linalg.norm(uncalibrated@gauge)/np.linalg.norm(actual),1e-10)
        for band in bands:
            self.assertGreater(band.prediction(fit['parameters'])['explained_fraction'],1-1e-8)
            for fold in band.component_transfer(truth):
                np.testing.assert_allclose(fold['multiplier_real_imag'],np.tile([1.,0.],(2,1)),atol=2e-8)
            for block in band.component_histories(truth,band.field.time[0],block_s=2.):
                np.testing.assert_allclose(block['multiplier_real_imag'],np.tile([1.,0.],(2,1)),atol=2e-8)
            complete=RigidBrightnessBand(band.field,np.ones(len(band.field.time),bool),band.knots,
                band.kinematics,band.points,band.band_phase,band.design)
            residual=complete.residual_diagnostics(truth)
            self.assertGreater(residual['explained_fraction'],1-1e-12)
            self.assertEqual(residual['fitted_complex_coordinates'],band.field.y.size)
            self.assertEqual(residual['unused_complex_coordinates'],0)
            with self.assertRaises(ValueError):complete.prediction(truth)
            self.assertEqual(len(complete.component_transfer(truth)),1)


if __name__=='__main__':unittest.main()
