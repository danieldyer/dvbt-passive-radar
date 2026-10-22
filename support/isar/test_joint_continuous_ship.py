import json,tempfile,unittest
from pathlib import Path
import numpy as np
from antenna_motion import Antennas
from continuous_ship_fit import ContinuousScene
from full_path_scene import SceneBand
from joint_continuous_ship import JointCoordinates,JointPointBand,JointPrior
from kinematic_prior import KinematicPrior
from full_path_point import carrier_bin
from rigid_patch import profile_static
from fit_joint_continuous_sites import feasible_improvement
from test_continuous_ship_fit import fixture


def joint_fixture(channel=0):
    original,motion=fixture();antennas=Antennas(motion.rotation,['a','a','a','a','b'])
    model=JointCoordinates(motion,antennas)
    return JointPointBand(original.field,original.coordinate,model,channel),motion


class JointContinuousTests(unittest.TestCase):
    def test_fixed_whitened_up_preserves_all_six_physical_up_coordinates(self):
        _,motion=fixture();rng=np.random.default_rng(53)
        for sites in ['00002','02002','12112']:
            antennas=Antennas(motion.rotation,list(sites));q=rng.normal(size=antennas.count);q[2::3]=0.
            for channel in range(5):
                physical=antennas.offsets(q,channel)@motion.rotation.T
                np.testing.assert_allclose(physical[:,2],0.,rtol=0,atol=3e-13)

    def test_feasible_first_order_gate_ignores_fixed_coordinates(self):
        matrix=np.array([[0.,1.]]);lo=np.array([-1.]);hi=np.array([1.]);lower=np.array([0.,-2.]);upper=np.array([0.,2.])
        gradient=np.array([1e6,.4])
        self.assertAlmostEqual(feasible_improvement(np.array([0.,0.]),gradient,matrix,lo,hi,lower,upper),.4)
        self.assertAlmostEqual(feasible_improvement(np.array([0.,-1.]),gradient,matrix,lo,hi,lower,upper),0.)
        self.assertAlmostEqual(feasible_improvement(np.array([0.,-1.]),np.array([1e6,-.4]),matrix,lo,hi,lower,upper),.4)

    def test_motion_and_six_antenna_partials_include_native_bin_response(self):
        b,m=joint_fixture();rng=np.random.default_rng(19);q=rng.normal(size=b.model.n)*.002;point=np.array([25.,-40.,3.])
        values=b.waveform(q,0,True,point)
        for j in range(8):
            direction=rng.normal(size=len(q))
            if j<4:direction[:b.nmotion]=0.
            step=3e-5
            numeric=(b.waveform(q+step*direction,0,False,point)-b.waveform(q-step*direction,0,False,point))/(2*step)
            analytic=sum(d*(basis@direction)[:,None] for d,basis in zip(values[1:],b.motion_bases))
            self.assertLess(np.linalg.norm(numeric-analytic)/np.linalg.norm(analytic),4e-5)

    def test_profiled_joint_gradient_and_free_constant_band_phase(self):
        bands=[];truth=None
        for c in range(2):
            b,m=joint_fixture(c);scene=SceneBand(b,np.array([[0.,0.,0.],[25.,-30.,0.]]),amplitude_spacing=4.)
            if truth is None:truth=np.linspace(-.001,.001,b.model.n)
            h=scene.design(truth,c,False);phase=np.array([.3,-.2])+(.27 if c==0 else .2)
            gain=(b.amplitude@np.full((b.amplitude.shape[1],2),.8))*np.exp(1j*phase)
            b.field.y=np.sum(h*gain[:,None,:],axis=-1)+b.field.root_information[:,None]*(.2+.3j);b.refresh_data();bands.append(scene)
        objective=ContinuousScene(bands,b.model,[0.,.2],fit_band_phases=True)
        at=np.r_[truth*.6,[.31,-.19],.065];value,g=objective.objective(at)
        np.testing.assert_allclose(objective.from_phase_coordinates(objective.phase_coordinates(at)),at,rtol=0,atol=2e-16)
        rng=np.random.default_rng(29)
        for _ in range(8):
            direction=rng.normal(size=len(at));direction/=np.linalg.norm(direction);step=2e-5
            numeric=(objective.objective(at+step*direction)[0]-objective.objective(at-step*direction)[0])/(2*step)
            self.assertLess(abs(numeric-g@direction),3e-5)
        self.assertLess(objective.objective(np.r_[truth,[.3,-.2],.27])[0],1e-10)
        before=objective.objective(at)
        for scene in bands:scene.adjoint_gradient=True
        objective.cached=None;after=objective.objective(at)
        self.assertAlmostEqual(before[0],after[0],places=13)
        np.testing.assert_allclose(before[1],after[1],rtol=2e-10,atol=2e-11)
        transform,report=objective.preconditioner(at)
        self.assertEqual(transform.shape,(len(at),len(at)));self.assertTrue(np.isfinite(transform).all())
        _,_,solutions=objective.objective(at,True);information=objective.mean_information(at)
        profiled_information=objective.mean_information(at,profile_brightness=True)
        objective.profile_brightness=True
        profiled_transform,conditioning=objective.preconditioner(at)
        self.assertTrue(np.isfinite(profiled_transform).all())
        self.assertTrue(conditioning['brightness_nuisance_profiled_on_active_face'])
        from scipy.linalg import block_diag
        amplitude_design=[]
        for c,(scene,solution) in enumerate(zip(bands,solutions)):
            h=scene.design(at[:objective.n],c,False)
            phase=np.exp(1j*scene.phase_at_time(objective.phase(at,c)))
            design=h[:,:,None,:]*phase[:,None,None,:]*scene.point.amplitude.toarray()[:,None,:,None]
            design=profile_static(design,scene.point.field.root_information,scene.point.training).reshape(-1,solution['amplitude'].size)
            amplitude_design.append(design[:,solution['amplitude'].ravel()>0])
        amplitude_design=block_diag(*amplitude_design);real_design=np.r_[amplitude_design.real,amplitude_design.imag]
        def prediction(q):
            result=[]
            for c,(scene,solution) in enumerate(zip(bands,solutions)):
                h=scene.design(q[:objective.n],c,False)
                gain=(scene.point.amplitude@solution['amplitude'])*np.exp(1j*scene.phase_at_time(objective.phase(q,c)))
                result.append(profile_static(np.sum(h*gain[:,None,:],axis=-1),scene.point.field.root_information,scene.point.training).ravel())
            return np.concatenate(result)
        for _ in range(4):
            direction=rng.normal(size=len(at));direction/=np.linalg.norm(direction);step=2e-5
            derivative=(prediction(at+step*direction)-prediction(at-step*direction))/(2*step)
            expected=2*float(np.vdot(derivative,derivative).real);actual=float(direction@information@direction)
            self.assertLess(abs(actual-expected)/expected,3e-5)
            real=np.r_[derivative.real,derivative.imag];coefficient=np.linalg.lstsq(real_design,real,rcond=None)[0]
            residual=real-real_design@coefficient;expected=2*float(residual@residual)
            actual=float(direction@profiled_information@direction)
            self.assertLess(abs(actual-expected)/expected,5e-4)

    def test_receiver_export_preserves_physical_ship_position(self):
        b,m=joint_fixture();q=np.linspace(-.001,.002,b.model.n)
        pc,angle=b.model.base.coefficients(q[:b.nmotion]);m.result.update(receiver_ecef_m=m.rx.tolist(),transmitters_ecef_m=m.tx.tolist())
        m.result={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in m.result.items()}
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'motion.json';b.model.export(q,path,{})
            d=json.loads(path.read_text());rx=np.asarray(d['receiver_ecef_m']);after=np.asarray(d['position_coefficients_enu_m'])
            expected=m.rx+pc@m.rotation+b.model.height_scale*q[b.nmotion]*m.rotation[2]
            np.testing.assert_allclose(rx+after@m.rotation,expected,rtol=0,atol=2e-9)
            self.assertGreater(np.linalg.norm(rx-m.rx),.001)
            self.assertFalse(d['antenna_refinement']['accepted_as_calibration'])

    def test_template_phase_origin_is_an_invertible_constant_gain(self):
        b,m=joint_fixture();q=np.linspace(-.001,.002,b.model.n);point=np.array([25.,-40.,3.])
        position,velocity,offset,*_=b.point_state(q,point)
        antenna=b.model.antennas.offsets(q[b.model.location_start:],0)
        tx,rx=m.tx[0]+antenna[0],m.rx+antenna[1];f=b.field
        raw=f.raw_trajectory(position,velocity,transmitter=tx,receiver=rx,position_offset=offset)
        _,_,g=f.path_kinematics(position,velocity,transmitter=tx,receiver=rx,position_offset=offset)
        factor,_=carrier_bin(b.t,np.sum(g*velocity,axis=1),b.coordinate['native_time_s'],b.coordinate['native_information'],f.k)
        original=f.apply(raw*factor[:,None]);rephased=b.waveform(q,0,False,point)
        reference=np.exp(1j*f.k*b.model.reference_phase(q,0))
        self.assertLess(np.linalg.norm(rephased-original*reference)/np.linalg.norm(original),1e-12)

    def test_location_and_jerk_prior_gradient(self):
        b,m=joint_fixture();model=b.model
        p=JointPrior(model,KinematicPrior(model.base,acceleration_change_mps2=.001,angular_rate_change_degps=.005))
        q=np.linspace(-.001,.002,model.n);v,g=p.evaluate(q)
        for j in [0,model.nmotion-1,model.nmotion,model.n-1]:
            step=np.eye(model.n)[j]*.002
            numeric=(p.evaluate(q+step)[0]-p.evaluate(q-step)[0])/.004
            self.assertLess(abs(numeric-g[j])/max(1,abs(g[j])),2e-6)

    def test_common_translation_is_not_false_absolute_location_information(self):
        bands=[];beta=np.arange(5)*.1;phi=np.array([.3,-.2])
        for c in range(5):
            b,m=joint_fixture(c);band=SceneBand(b,np.array([[0.,0.,0.],[25.,-30.,0.]]),amplitude_spacing=4.,adjoint_gradient=True)
            h=band.design(np.zeros(b.model.n),c,False)
            gain=(b.amplitude@np.full((b.amplitude.shape[1],2),.8))*np.exp(1j*(phi+beta[c]))
            b.field.y=np.sum(h*gain[:,None,:],axis=-1);b.refresh_data();bands.append(band)
        scene=ContinuousScene(bands,b.model,beta,fit_band_phases=True);at=np.r_[np.zeros(b.model.n),phi,np.zeros(4)]
        shifted=at.copy();delta=np.array([.5,-.3,.2]);model=b.model;basis=model.base
        shifted[:basis.np]=delta@basis.axes[0]
        shifted[basis.np:2*basis.np]=(delta@basis.axes[1])/10.
        shifted[model.nmotion]=delta[2]/model.height_scale
        shifted[model.location_start:model.n]=np.linalg.solve(model.antennas.factor,np.tile(delta,6))
        # A constant template phase reference has no absolute survey meaning.
        # Its change is exactly absorbed by the declared constant phase gauges.
        change=np.array([bands[c].point.field.k*model.reference_phase(shifted[:model.n],c) for c in range(5)])
        shifted[model.n:model.n+scene.p]-=change[3]
        for j,c in enumerate(scene.free_bands):shifted[model.n+scene.p+j]=change[3]-change[c]
        self.assertLess(scene.objective(at)[0],1e-10)
        self.assertLess(scene.objective(shifted)[0],1e-10)


if __name__=='__main__':unittest.main()
