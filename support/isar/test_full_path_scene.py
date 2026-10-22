import unittest
import numpy as np
from full_path_scene import SceneBand,ScenePath,RefinedScenePath,warm_phases
from test_full_path_point import problem
from rigid_patch import profile_static


class FullPathSceneTests(unittest.TestCase):
    def test_ragged_sources_original_likelihood_path_and_phase_gradients(self):
        point=problem()
        points=np.array([[[0.,0.,0.],[27.,-19.,10.]],[[12.,-5.,0.],[-40.,35.,10.]]])
        active=np.array([[True,False],[True,True]])
        scene=SceneBand(point,points,[0.,8.,point.t[-1]],2.,active)
        n=point.b.shape[1];truth=np.linspace(-.02,.03,n);phase=np.array([.4,-.7,1.1])
        h=scene.design(truth,3,False)
        coeff=np.c_[.8+.1*np.cos(point.amplitude_nodes),.7+.1*np.sin(point.amplitude_nodes)]
        gain=(point.amplitude@coeff)*np.exp(1j*scene.phase_at_time(phase))
        point.field.y=np.sum(h*gain[:,None,:],axis=-1)+point.field.root_information[:,None]*(.1+.2j)
        for train in [True,False]:
            mask=point.training if train else ~point.training
            y=profile_static(point.field.y,point.field.root_information,mask,point.groups)*mask[:,None]
            point.data[train]=y;point.energy[train]=np.vdot(y,y).real
        fitted=scene.evaluate(truth,phase,3)
        excluded=scene.evaluate(truth,phase,3,training=False,derivatives=False,frozen_amplitude=fitted['amplitude'])
        self.assertLess(excluded['cost']/excluded['energy'],1e-11)
        at=truth*.5;ph=phase+.03;value=scene.evaluate(at,ph,3)
        for j in range(n+len(phase)):
            if j<n:
                step=np.eye(n)[j]*5e-5
                finite=(scene.evaluate(at+step,ph,3)['cost']-scene.evaluate(at-step,ph,3)['cost'])/1e-4
                analytic=value['path_gradient'][j]
            else:
                step=np.eye(len(phase))[j-n]*5e-5
                finite=(scene.evaluate(at,ph+step,3)['cost']-scene.evaluate(at,ph-step,3)['cost'])/1e-4
                analytic=value['phase_gradient'][j-n]
            self.assertLess(abs(finite-analytic)/value['energy'],2e-5)

    def test_variable_support_counts_keep_exact_phases_and_zero_inactive_columns(self):
        b=problem();points=np.array([[[0.,0.,0.],[27.,-19.,10.]],[[12.,-5.,0.],[-40.,35.,10.]]])
        mask=np.array([[True,False],[True,True]])
        full=SceneBand(b,points,[0.,8.,b.t[-1]],2.)
        h=full.design(np.zeros(b.b.shape[1]),3,False)
        band=SceneBand(b,points,[0.,8.,b.t[-1]],2.,mask)
        actual=band.design(np.zeros(b.b.shape[1]),3,False)
        np.testing.assert_allclose(actual,h*mask[band.window_index][:,None,:],rtol=2e-7,atol=1e-9)
        self.assertEqual(band.phase_count,3)
        phases=band.phase_at_time([.2,-.4,.7])
        np.testing.assert_array_equal(phases[band.window_index==0,0],.2)
        np.testing.assert_array_equal(phases[band.window_index==1,0],-.4)
        np.testing.assert_array_equal(phases[band.window_index==1,1],.7)

    def test_physical_spin_control_recovers_original_complex_data(self):
        from declared_ship_orientation import rotate_frame
        point=problem();points=np.array([[-90.,-12.,20.],[-25.,7.,10.],[45.,-8.,18.],[110.,12.,25.]])
        point.amplitude=__import__('scipy').sparse.csr_matrix(np.ones((len(point.t),1)))
        scene=SceneBand(point,points,amplitude_spacing=0.)
        n=point.b.shape[1];reference=point.frame.copy();reference_rate=point.frame_rate.copy()
        true_rate=-.13;phase=np.array([.2,-.8,.4,1.1]);amplitude=np.array([[1.,.8,1.2,.7]])
        rotated=rotate_frame(dict(body_to_ecef=reference,frame_velocity=reference_rate),point.t,8.,true_rate)
        point.frame=rotated['body_to_ecef'];point.frame_rate=rotated['frame_velocity']
        h=scene.design(np.zeros(n),3,False);point.field.y=np.sum(h*(amplitude*np.exp(1j*phase))[:,None,:],axis=-1)
        for train in [True,False]:
            mask=point.training if train else ~point.training
            y=profile_static(point.field.y,point.field.root_information,mask,point.groups)*mask[:,None]
            point.data[train]=y;point.energy[train]=np.vdot(y,y).real
        values=[]
        for rate in [true_rate-.1,true_rate,true_rate+.1,0.]:
            rotated=rotate_frame(dict(body_to_ecef=reference,frame_velocity=reference_rate),point.t,8.,rate)
            point.frame=rotated['body_to_ecef'];point.frame_rate=rotated['frame_velocity'];scene.warm=None
            fitted=scene.evaluate(np.zeros(n),phase,3)
            excluded=scene.evaluate(np.zeros(n),phase,3,training=False,derivatives=False,frozen_amplitude=fitted['amplitude'])
            values.append(excluded['cost']/excluded['energy'])
        self.assertLess(values[1],1e-20)
        self.assertGreater(min(values[0],values[2],values[3]),.1)

    def test_common_source_coordinate_derivatives(self):
        bands=[]
        points=np.array([[[0.,0.,0.],[27.,-19.,10.]],[[12.,-5.,0.],[-40.,35.,10.]]])
        for c in range(2):
            point=problem();band=SceneBand(point,points,[0.,8.,point.t[-1]],0.)
            n=point.b.shape[1];truth=np.linspace(-.02,.03,n);phase=np.array([.4,-.7,.6,1.1])+c*.2
            h=band.design(truth,c,False)
            point.field.y=np.sum(h*np.exp(1j*band.phase_at_time(phase))[:,None,:],axis=-1)+point.field.root_information[:,None]*(.1+.2j)
            for train in [True,False]:
                mask=point.training if train else ~point.training
                y=profile_static(point.field.y,point.field.root_information,mask,point.groups)*mask[:,None]
                point.data[train]=y;point.energy[train]=np.vdot(y,y).real
            bands.append(band)
        model=RefinedScenePath(bands)
        parameters=np.r_[truth*.5,.43,.63,0.,0.,0.,phase[1:]-phase[0],np.linspace(-.03,.04,8)]
        value,gradient=model.objective(parameters)
        for j in range(len(parameters)):
            step=np.eye(len(parameters))[j]*5e-5
            numeric=(model.objective(parameters+step)[0]-model.objective(parameters-step)[0])/1e-4
            self.assertLess(abs(numeric-gradient[j]),3e-5)
        # Refining a common source changes each band through its own exact
        # geometric path; no independent band coordinate can be introduced.
        model.set_points(parameters)
        np.testing.assert_array_equal(bands[0].points,bands[1].points)

    def test_local_reflector_phases_share_global_physical_motion(self):
        point=problem();points=np.array([[[0.,0.,0.],[27.,-19.,10.]],[[12.,-5.,0.],[-40.,35.,10.]]])
        scene=SceneBand(point,points,[0.,8.,point.t[-1]],2.);n=point.b.shape[1];truth=np.linspace(-.02,.03,n);phase=np.array([.4,-.7,.6,1.1])
        h=scene.design(truth,3,False);coeff=np.c_[.8+.1*np.cos(point.amplitude_nodes),.7+.1*np.sin(point.amplitude_nodes)];amp=point.amplitude@coeff
        point.field.y=np.sum(h*(amp*np.exp(1j*scene.phase_at_time(phase)))[:,None,:],axis=-1)+point.field.root_information[:,None]*(.1+.2j)
        for train in [True,False]:
            mask=point.training if train else ~point.training;y=profile_static(point.field.y,point.field.root_information,mask,point.groups)*mask[:,None]
            point.data[train]=y;point.energy[train]=np.vdot(y,y).real
        actual=scene.evaluate(truth,phase,3);unused=scene.evaluate(truth,phase,3,training=False,derivatives=False,frozen_amplitude=actual['amplitude'])
        self.assertLess(actual['cost']/actual['energy'],1e-11);self.assertLess(unused['cost']/unused['energy'],1e-11)
        at=truth*.5;ph=phase+.03;r=scene.evaluate(at,ph,3)
        for j in range(n+len(phase)):
            if j<n:
                step=np.eye(n)[j]*5e-5;finite=(scene.evaluate(at+step,ph,3)['cost']-scene.evaluate(at-step,ph,3)['cost'])/1e-4;analytic=r['path_gradient'][j]
            else:
                step=np.eye(len(phase))[j-n]*5e-5;finite=(scene.evaluate(at,ph+step,3)['cost']-scene.evaluate(at,ph-step,3)['cost'])/1e-4;analytic=r['phase_gradient'][j-n]
            self.assertLess(abs(finite-analytic)/r['energy'],2e-5)

    def test_constant_phase_block_initialization_uses_the_same_likelihood(self):
        points=np.array([[0.,0.,0.],[27.,-19.,10.]]);bands=[]
        for c,beta in enumerate([.4,-.7]):
            point=problem();band=SceneBand(point,points);n=point.b.shape[1];h=band.design(np.zeros(n),c,False)
            amplitude=point.amplitude@np.c_[.8+.1*np.cos(point.amplitude_nodes),.7+.1*np.sin(point.amplitude_nodes)]
            point.field.y=np.sum(h*(amplitude*np.exp(1j*(beta+np.array([0.,.8]))))[:,None,:],axis=-1)+point.field.root_information[:,None]*(.1+.2j)
            for train in [True,False]:
                mask=point.training if train else ~point.training;y=profile_static(point.field.y,point.field.root_information,mask,point.groups)*mask[:,None]
                point.data[train]=y;point.energy[train]=np.vdot(y,y).real
            bands.append(band)
        likelihood=ScenePath(bands);parameters=np.r_[np.zeros(n),.6,-.9,0.,0.,0.,.6]
        before=likelihood.objective(parameters)[0];result,report=warm_phases(likelihood,parameters,200)
        after=likelihood.objective(result)[0]
        self.assertLess(after,1e-9);self.assertLess(after,before*1e-5);self.assertAlmostEqual(after,report['objective'],places=11)

    def test_shared_motion_positive_scene_gradient_and_excluded_prediction(self):
        point=problem();points=np.array([[0.,0.,0.],[27.,-19.,10.]])
        point.groups=((point.t>=4)&(point.t<12)).astype(int)
        scene=SceneBand(point,points);n=point.b.shape[1];truth=np.linspace(-.03,.04,n);phase=np.array([.4,-.7])
        h=scene.design(truth,3,False);coeff=np.c_[.7+.1*np.cos(point.amplitude_nodes),.8+.2*np.sin(point.amplitude_nodes)]
        amplitude=point.amplitude@coeff;point.field.y=np.sum(h*(amplitude*np.exp(1j*phase))[:,None,:],axis=-1)+point.field.root_information[:,None]*(.3+.4j)
        for train in [True,False]:
            mask=point.training if train else ~point.training;y=profile_static(point.field.y,point.field.root_information,mask,point.groups)*mask[:,None]
            point.data[train]=y;point.energy[train]=np.vdot(y,y).real
        actual=scene.evaluate(truth,phase,3);unused=scene.evaluate(truth,phase,3,training=False,derivatives=False,frozen_amplitude=actual['amplitude'])
        self.assertLess(actual['cost']/actual['energy'],1e-12);self.assertLess(unused['cost']/unused['energy'],1e-12)
        at=truth*.5;ph=phase+np.array([.02,-.03]);r=scene.evaluate(at,ph,3)
        for j in range(n):
            step=np.eye(n)[j]*5e-5;finite=(scene.evaluate(at+step,ph,3)['cost']-scene.evaluate(at-step,ph,3)['cost'])/1e-4
            self.assertLess(abs(finite-r['path_gradient'][j])/r['energy'],2e-5)
        for j in range(2):
            step=np.eye(2)[j]*5e-5;finite=(scene.evaluate(at,ph+step,3)['cost']-scene.evaluate(at,ph-step,3)['cost'])/1e-4
            self.assertLess(abs(finite-r['phase_gradient'][j])/r['energy'],2e-5)

if __name__=='__main__':unittest.main()
