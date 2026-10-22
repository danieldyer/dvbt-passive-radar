"""Synthetic checks for offline calibration bookkeeping (no captured data needed)."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

from geometry import C, RF, SITES, observables, ecef, enu_rotation, central_jacobian
from phase_history import interpolate
from phase_fit import projection, path_model, motion_penalty, physical_problem
from complex_likelihood import gate, profile
from polar_model import PolarTrajectory
from polar_fit import Problem
from coherent_fit import CoherentProblem, OTHER, information
from range_likelihood import interpolate_patch, OFFSETS
from range_fit import Pulse, Field, RangeProblem
from ground_model import GroundTrajectory
from terrain import geodetic, RasterTerrain
from cosited_paths import PathHistory, PathProblem
from calibration_uncertainty import coherent_fraction, phase_derivative


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(718)
        self.sites = np.array([[31634., -3030., 639.], [31732., -2815., 624.],
                               [31806., -2753., 611.]])
        self.state = np.array([[.5, .2, .08], [4.5, -3., .03], [.04, -.06, .01],
                               [.002, .003, -.001]]).ravel()

    def test_cosited_phase_history_recovery_and_heldout_isolation(self):
        seed = {'epoch':50., 'scale':15., 'path':[1600.,420.,-3.,1.]}
        model = PathHistory(seed, 3.)
        state = self.rng.normal(0, .08, model.count)
        channels, selected = [], []
        for channel in range(4):
            time = np.linspace(35.02, 64.97, 600)+channel*.00003
            other = dict(seed, path=[1600.+channel*8.,420.+channel*.2,-3.,1.])
            delta = np.polynomial.polynomial.polyval((time-50)/15, other['path'])-other['path'][0]
            k = 2*np.pi*RF[channel]/C
            amplitude = (1.+channel*.4)*np.exp(1j*.83*channel)
            y = amplitude*np.exp(-1j*k*(model.path(state, time)-delta))
            variance = np.full(len(time), .001)
            q, b = gate(time, variance, 1., 50., 30.)
            channels.append({'time':time, 'seed':other, 'seed_delta':delta,
                'k':k, 'operator':q, 'basis':b, 'variance':variance,
                'bandwidth_hz':1., 'value':y})
            selected.append(np.floor(time-35).astype(int)%2==0)
        problem = PathProblem(channels, model, selected)
        self.assertLess(np.max(abs(problem.residual(state))), 1e-10)
        trial = state+self.rng.normal(0, .015, model.count)
        numerical = central_jacobian(problem.residual, trial, 1e-5)
        np.testing.assert_allclose(problem.residual(trial, True), numerical, atol=2e-7)
        fit = problem.fit([trial])
        np.testing.assert_allclose(fit.x, state, atol=1e-8)
        modified = []
        for d, mask in zip(channels, selected):
            y = d['value'].copy()
            y[~mask] += self.rng.normal(size=(~mask).sum())*100.
            modified.append(dict(d, value=y))
        altered = PathProblem(modified, model, selected)
        np.testing.assert_allclose(altered.residual(trial), problem.residual(trial), atol=1e-10)

    def test_common_phase_uncertainty_cancels_in_coherent_power(self):
        derivative = np.eye(5)[None,:,:]
        variance = .37
        expected = (5+20*np.exp(-variance))/25
        np.testing.assert_allclose(coherent_fraction(derivative,np.eye(5)*variance),expected,atol=1e-15)
        # An arbitrarily large common phase is irrelevant to received power.
        np.testing.assert_allclose(coherent_fraction(derivative,np.ones((5,5))*100),1,atol=1e-15)
        np.testing.assert_allclose(coherent_fraction(derivative,np.zeros((5,5))),1,atol=1e-15)

    def test_fixed_world_phase_prediction_jacobian(self):
        x = np.array([[-5000.,-4000.,100.],[6000.,5000.,200.]])
        assignment = [0,0,0,0,2]
        offsets = np.zeros((5,3));offsets[:,2] = [4.,0.,3.,3.,0.]
        derivative,_ = phase_derivative(x,self.sites,assignment,offsets)
        def phase(p):
            sites = self.sites+p[:9].reshape(3,3)*[30,30,75]
            _,lengths = phase_derivative(x,sites,assignment,offsets)
            beta = np.zeros(5);beta[[0,2,3,4]] = p[9:]
            return (-2*np.pi*RF[None,:]/C*lengths.T+beta).ravel()
        numerical = central_jacobian(phase,np.zeros(13),1e-5)
        np.testing.assert_allclose(derivative.reshape(-1,13),numerical,atol=2e-5)

    def test_excess_rate_is_path_derivative(self):
        x = np.array([5000., 2000., 800.]); v = np.array([30., -20., 1.])
        h = .001
        low = observables(x-h*v, v, self.sites, np.zeros(3))[0]
        high = observables(x+h*v, v, self.sites, np.zeros(3))[0]
        rate = observables(x, v, self.sites, np.zeros(3))[1]
        np.testing.assert_allclose((high-low)/(2*h), rate, atol=1e-8)

    def test_projected_covariance_and_model_jacobian(self):
        time = np.arange(300)*.02+13.0001
        variance = self.rng.uniform(.2, 1., len(time))
        q, _ = gate(time, variance, 1., 16., 6.)
        np.testing.assert_allclose((q*variance)@q.conj().T, np.eye(len(q)), atol=2e-14)
        basis = self.rng.normal(size=(len(time), 4))*.2
        p = self.rng.normal(size=4)
        y = self.rng.normal(size=len(time))+1j*self.rng.normal(size=len(time))
        def fun(p):
            r, _, _ = profile(y, q, basis@p)
            return np.r_[r.real, r.imag]
        _, j, _, _ = profile(y, q, basis@p, basis)
        np.testing.assert_allclose(np.r_[j.real, j.imag], central_jacobian(fun, p), atol=2e-9)

    def test_exact_bistatic_polar_coordinates_and_jacobian(self):
        seed = {'epoch':50.,'path':[1600.,420.,-10.,4.,8.]}
        model = PolarTrajectory(seed)
        p = np.r_[self.rng.normal(0,.03,model.radial_count),[-2.5,.1,.03,.02,.04,.003],
                  self.rng.normal(size=9)]
        t = np.linspace(35.1,64.9,41);channel = np.arange(len(t))%5
        length,j = model.paths(p,t,channel,self.sites,True)
        numerical = central_jacobian(lambda p:model.paths(p,t,channel,self.sites,True)[0],p,1e-6)
        self.assertLess(np.max(abs(j-numerical)),2e-5)
        reference,j = model.paths(p,t,np.ones(len(t),dtype=int),self.sites,True)
        base,b = model.length_basis(t)
        np.testing.assert_allclose(reference,base+b@p[:model.radial_count],atol=2e-11)
        self.assertLess(np.max(abs(j[:,model.radial_count:])),1e-10)

    def test_configurable_licensed_antenna_assignments(self):
        seed={'epoch':50.,'path':[1600.,420.,-10.,4.,8.]}
        offsets=np.zeros((5,3));offsets[:,2]=[4.,0.,3.,3.,0.]
        model=PolarTrajectory(seed,channel_sites=[0,0,0,0,2],channel_offsets=offsets)
        p=np.r_[np.zeros(model.radial_count),[-2.5,.1,.03,.02,.04,.003],self.rng.normal(size=9)]
        time=np.linspace(36.,64.,25);channels=np.arange(25)%5
        path,j=model.paths(p,time,channels,self.sites,True)
        numerical=central_jacobian(lambda q:model.paths(q,time,channels,self.sites,True)[0],p,1e-6)
        np.testing.assert_allclose(j,numerical,atol=2e-5)
        x,_,tx=model.positions(p,time,self.sites,True)
        expected=observables(x,np.zeros_like(x),tx[model.channel_sites[channels]]+offsets[channels],np.zeros(3))[0]
        np.testing.assert_allclose(path,expected,atol=1e-10)

    def test_shared_channel_phase_model_and_full_jacobian(self):
        problems,states = [],[]
        for target in range(2):
            seed = {'epoch':50.+target*100,'path':[1600.+300*target,420.,-10.,4.,8.]}
            model = PolarTrajectory(seed)
            state = np.r_[np.zeros(model.radial_count),[-2.5+target*.3,.1,.03,.02,.04,.003]]
            channels = []
            for ch in range(5):
                time = seed['epoch']+np.linspace(-14.9,14.9,101)+ch*.0001
                path,_ = model.paths(state,time,np.full(len(time),ch),self.sites)
                delta = np.polynomial.polynomial.polyval((time-seed['epoch'])/15,seed['path'])-seed['path'][0]
                q,b = gate(time,np.ones(len(time)),.2,seed['epoch'],30.)
                k = 2*np.pi*RF[ch]/C
                a = np.exp(1j*(target*.73+ch*.31))*(1.+target*.2)*(1.+ch*.14+target*ch*.07)
                channels.append({'time':time,'seed':seed,'seed_delta':delta,'k':k,
                    'operator':q,'basis':b,'variance':np.ones(len(time)),
                    'value':a*np.exp(-1j*k*(path-delta))})
            t = np.linspace(-14.,14.,15); ch = np.arange(len(t))%5
            length,_ = model.paths(state,t+seed['epoch'],ch,self.sites)
            coarse = {'t':t,'channel':ch,'length':length,'sigma_l':np.full(len(t),3.)}
            problems.append(Problem(channels,coarse,self.sites))
            states.append(state)
        joint = CoherentProblem(problems)
        p = joint.initial(states)
        exact = joint.residual(p,include_motion=False,include_site_prior=False)
        self.assertLess(np.max(abs(exact)),1e-9)
        # Residual clutter orthogonal to each target template must not become
        # extra expected signal information merely through variable projection.
        means = joint.fitted_observations(p)
        original = joint.data
        before = information(joint, p)
        disturbed = []
        for y, mean in zip(joint.data, means):
            clutter = self.rng.normal(size=len(y))+1j*self.rng.normal(size=len(y))
            clutter -= mean*np.vdot(mean,clutter)/np.vdot(mean,mean).real
            disturbed.append(y+20*clutter)
        joint.data = disturbed
        for actual, expected in zip(joint.fitted_observations(p), means):
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        after = information(joint, p)
        np.testing.assert_allclose(after['data_posterior_covariance_normalized'],
                                   before['data_posterior_covariance_normalized'],atol=1e-10)
        joint.data = original
        p[:9] = self.rng.normal(0,.1,9)
        p[joint.beta_offset:joint.magnitude_offset] += self.rng.normal(0,.1,4)
        p[joint.magnitude_offset:] += self.rng.normal(0,.1,8)
        actual = joint.residual(p,True)
        numerical = central_jacobian(joint.residual,p,1e-5)
        self.assertLess(np.linalg.norm(actual-numerical)/np.linalg.norm(actual),2e-7)

        spectrum=np.zeros(8192);spectrum[np.arange(-3408,3409)%8192]=1.
        pulse=Pulse(np.fft.ifft(spectrum)[np.arange(-64,65)%8192])
        biases=np.array([2.,0.,-3.,-15.,-25.]);ranges=[]
        for target,pr in enumerate(problems):
            fields=[]
            for ch,d in enumerate(pr.channels):
                length,_=pr.model.paths(states[target],d['time'],np.full(len(d['time']),ch),self.sites)
                base=np.polynomial.polynomial.polyval((d['time']-d['seed']['epoch'])/15,d['seed']['path'])
                y=d['value'][:,None]*pulse(OFFSETS[None,:]-(length-base+biases[ch])[:,None]*8e6/C)
                data={'time':d['time'],'offset_samples':OFFSETS,'variance':np.ones(y.shape),'value':y}
                fields.append(Field(data,pulse,d['seed']|{'rf_hz':RF[ch]},.2))
            ranges.append(RangeProblem(pr.channels,pr.coarse,self.sites,fields))
        joint=CoherentProblem(ranges,range_bias=True);p=joint.initial(states)
        p[joint.bias_offset:]=biases[OTHER]
        p[joint.beta_offset:joint.magnitude_offset]=(OTHER-1)*.31
        for target in range(2):
            gains=1.+np.arange(5)*.14+target*np.arange(5)*.07
            p[joint.magnitude_offset+4*target:joint.magnitude_offset+4*(target+1)]=np.log(gains[OTHER]/gains[1])
        self.assertLess(np.max(abs(joint.residual(p,include_motion=False,include_site_prior=False))),1e-9)
        p[joint.bias_offset:]+=self.rng.normal(size=4)
        actual=joint.residual(p,True)
        numerical=central_jacobian(joint.residual,p,1e-5)
        self.assertLess(np.linalg.norm(actual-numerical)/np.linalg.norm(actual),2e-7)

    def test_polar_kinematics_parameter_derivative(self):
        seed = {'epoch':50.,'path':[1600.,420.,-10.,4.,8.]}
        model = PolarTrajectory(seed)
        p = np.r_[np.zeros(model.radial_count),[-2.5,.1,.03,.02,.04,.003]]
        values,j = model.kinematics(p,self.sites)
        numerical = central_jacobian(lambda q:np.concatenate([x.ravel() for x in model.kinematics(q,self.sites)[0]]),p,1e-4)
        actual = np.concatenate([x.reshape(-1,len(p)) for x in j])
        self.assertLess(np.max(abs(actual-numerical)),1e-5)

    def test_geographic_dem_pixel_centres_and_nodata(self):
        from PIL import Image,TiffImagePlugin
        with TemporaryDirectory() as directory:
            root=Path(directory)
            Image.fromarray(np.zeros((2,2),np.float32)).save(root/'us_nga_egm96_15.tif')
            tags=TiffImagePlugin.ImageFileDirectory_v2()
            tags[33922]=(0.,0.,0.,145.,-37.,0.);tags[33550]=(.01,.01,0.)
            tags[34735]=(1,1,0,2,1025,0,1,1,2048,0,1,4326)
            Image.fromarray(np.array([[10.,20.,30.],[30.,40.,50.],[60.,70.,1e30]],np.float32)).save(root/'dem.tif',tiffinfo=tags)
            terrain=RasterTerrain(root,{'receiver_ecef_m':[0,0,0],'ecef_to_enu':np.eye(3).tolist()},root/'dem.tif')
            self.assertAlmostEqual(float(terrain.elevation(-37.005,145.005)),10.,places=8)
            self.assertAlmostEqual(float(terrain.elevation(-37.010,145.010)),25.,places=8)
            self.assertTrue(np.isnan(terrain.elevation(-37.025,145.025)))
            self.assertTrue(np.isnan(terrain.elevation(-38.,145.)))

    def test_static_cpi_information_and_discontinuities(self):
        from geographic_preview import integrate
        dtype=[('channel','i4'),('bin','i8'),('segment','i8'),('information','f8'),
               ('values','c8',1024),('psf','c8',129)]
        records=np.zeros(5*92,dtype=dtype)
        for channel in range(5):
            rows=records[channel*92:(channel+1)*92]
            rows['channel']=channel;rows['bin']=np.arange(92)+100
            rows['information']=np.arange(92)+1;rows['values']=complex(channel+1,.37)
            rows['psf'][:,64]=1.08
        time=146*(30*8704/8e6)
        profiles,weights,metadata=integrate(records,time)
        np.testing.assert_allclose(profiles[:,0],np.arange(1,6)+.37j,atol=1e-7)
        np.testing.assert_allclose(weights[:,0],sum(range(1,93)))
        self.assertAlmostEqual(metadata['coherence_seconds'],3.00288)
        records['segment'][20]=1
        with self.assertRaises(ValueError): integrate(records,time)
        records['segment'][20]=0;records['bin'][20]+=1
        with self.assertRaises(ValueError): integrate(records,time)

    def test_off_gate_return_is_not_calibration_target(self):
        time = np.arange(1500)*.02
        q, _ = gate(time, np.ones(len(time)), 1., 15., 30.)
        target = (.3+.1j)*np.exp(2j*np.pi*.2*time)
        blocker = 30*np.exp(2j*np.pi*3*time)
        r, amp, _ = profile(target+blocker, q, 2*np.pi*.2*time)
        self.assertLess(abs(amp-(.3+.1j)), 1e-11)
        self.assertLess(np.max(abs(r)), 1e-10)

    def test_range_patch_matches_scalar_interpolator(self):
        profiles=self.rng.normal(size=(13,1024))+1j*self.rng.normal(size=(13,1024))
        coordinate=np.linspace(21.37,850.69,13)
        patch=interpolate_patch(profiles,coordinate)
        expected=np.array([interpolate(profiles,coordinate+offset) for offset in OFFSETS]).T
        np.testing.assert_allclose(patch,expected,atol=2e-13)

    def test_full_range_likelihood_delay_and_derivative(self):
        spectrum=np.zeros(8192)
        spectrum[np.arange(-3408,3409)%8192]=1.+.15*np.cos(np.arange(-3408,3409)/600)
        autocorrelation=np.fft.ifft(spectrum)[np.arange(-64,65)%8192]
        pulse=Pulse(autocorrelation)
        time=np.linspace(35.1,64.9,103)
        seed={'epoch':50.,'path':[1600.,420.,-10.,4.,8.],'rf_hz':184.5e6}
        base=np.polynomial.polynomial.polyval((time-50)/15,seed['path'])
        basis=np.c_[np.ones(len(time)),(time-50)/15]
        offset=np.array([7.3,.17]);path=base+basis@offset
        lag=OFFSETS[None,:]-(path-base)[:,None]*8e6/C
        k=2*np.pi*seed['rf_hz']/C
        y=(.3+.7j)*np.exp(-1j*k*(path-(base-seed['path'][0])))[:,None]*pulse(lag)
        data={'time':time,'offset_samples':OFFSETS,'variance':np.ones(y.shape),'value':y}
        field=Field(data,pulse,seed)
        error,_=field.evaluate(path)
        self.assertLess(np.max(abs(error)),1e-11)
        p=np.array([6.4,.05])
        def fun(p):
            r,_=field.evaluate(base+basis@p)
            return np.r_[r.real,r.imag]
        _,j,_=field.evaluate(base+basis@p,basis)
        np.testing.assert_allclose(np.r_[j.real,j.imag],central_jacobian(fun,p,1e-5),atol=4e-7,rtol=1e-5)
        from scipy.optimize import least_squares
        result=least_squares(fun,p,jac=lambda p:np.r_[field.evaluate(base+basis@p,basis)[1].real,
                                                       field.evaluate(base+basis@p,basis)[1].imag])
        np.testing.assert_allclose(result.x,offset,atol=1e-5)

    def test_rotation_gauge_preserves_ranges_rates_and_phase(self):
        rotation = Rotation.from_rotvec([.05, -.03, .02]).as_matrix()
        x = np.array([5000., 2000., 800.]); v = np.array([30., -20., 1.])
        a, b = observables(x, v, self.sites, np.zeros(3))
        c, d = observables(rotation@x, rotation@v, self.sites@rotation.T, np.zeros(3))
        np.testing.assert_allclose(a, c, atol=2e-11)
        np.testing.assert_allclose(b, d, atol=1e-12)
        np.testing.assert_allclose(np.exp(-2j*np.pi*RF/C*a[SITES]),
                                   np.exp(-2j*np.pi*RF/C*c[SITES]), atol=1e-10)

    def test_enu_rotation_is_rigid(self):
        q = enu_rotation(-37.81, 144.99)
        np.testing.assert_allclose(q@q.T, np.eye(3), atol=1e-15)
        np.testing.assert_allclose(ecef(0., 0., 0.), [6378137., 0., 0.], atol=1e-9)

    def test_geodetic_roundtrip_and_ground_intersection_jacobian(self):
        latitude=np.array([-37.8,0.,45.,-89.]);longitude=np.array([145.,-170.,-45.,72.]);height=np.array([45.,1000.,20.,3000.])
        a,b,c=geodetic(ecef(latitude,longitude,height))
        np.testing.assert_allclose(a,latitude,atol=1e-12)
        np.testing.assert_allclose(b,longitude,atol=1e-12)
        np.testing.assert_allclose(c,height,atol=3e-9)
        class InclinedGround:
            def clearance(self,x,derivative=False):
                g=np.broadcast_to([-.002,.003,1.],x.shape)
                h=np.sum(x*g,axis=1)+12.
                return (h,g) if derivative else h
        ground=InclinedGround();model=GroundTrajectory({'epoch':50.,'path':[1600.,420.,-10.,4.,8.]},ground)
        p=np.r_[np.zeros(model.radial_count),[-2.5,.1,.03],.2,self.rng.normal(0,.1,9)]
        time=np.linspace(35.1,64.9,21);ch=np.arange(len(time))%5
        path,j=model.paths(p,time,ch,self.sites,True)
        x,jx,_=model.positions(p,time,self.sites,True)
        np.testing.assert_allclose(ground.clearance(x),2.,atol=1e-9)
        reference,_=model.paths(p,time,np.ones(len(time),int),self.sites,True)
        np.testing.assert_allclose(reference,model.length_basis(time)[0],atol=2e-11)
        numerical=central_jacobian(lambda q:model.paths(q,time,ch,self.sites,True)[0],p,1e-5)
        self.assertLess(np.max(abs(j-numerical)),2e-6)

    def test_complex_range_interpolation_phase_and_negative_support(self):
        frequency = np.linspace(-3408/8192, 3408/8192, 93)
        coordinate = np.linspace(16.17, 870.83, len(frequency))
        ranges = np.r_[np.arange(960), np.arange(-64, 0)]
        profile = np.exp(2j*np.pi*frequency[:, None]*ranges)
        actual = interpolate(profile, coordinate)
        expected = np.exp(2j*np.pi*frequency*coordinate)
        self.assertLess(np.max(abs(actual/expected-1)), 6e-5)
        self.assertLess(np.max(abs(np.angle(actual/expected)))*180/np.pi, .004)

    def test_interpolation_rejects_missing_support(self):
        with self.assertRaises(ValueError):
            interpolate(np.ones((1, 1024), complex), np.array([923.]))

    def test_projection_jacobian_and_training_mask(self):
        n = 73
        y = self.rng.normal(size=n)+1j*self.rng.normal(size=n)
        variance = self.rng.uniform(.1, 2, n)
        basis = self.rng.normal(size=(n, 4)); p = self.rng.normal(size=4)
        train = np.arange(n)%3 != 0
        def fun(x):
            r, _, _ = projection(y, variance, basis@x, train=train)
            return np.r_[r.real, r.imag]
        _, jac, _, _ = projection(y, variance, basis@p, basis, train)
        np.testing.assert_allclose(np.r_[jac.real, jac.imag], central_jacobian(fun, p),
                                   atol=3e-8, rtol=1e-7)

    def test_path_jacobian_includes_direct_reference_distance(self):
        state = np.r_[self.state, self.rng.normal(size=9)]
        u = np.linspace(-1, 1, 37); channel = np.arange(37)%5
        _, j = path_model(state, u, channel, self.sites, 3, True)
        numerical = central_jacobian(lambda p:path_model(p, u, channel, self.sites, 3, True)[0], state, 1e-6)
        self.assertLess(np.max(abs(j-numerical)), 1e-5)

    def test_multiband_deramp_at_distinct_actual_epochs(self):
        for ch in range(5):
            time = np.linspace(35, 65, 91)+ch*.00019
            u = (time-50)/15
            path, _ = path_model(self.state, u, np.full(len(u), ch), self.sites, 3, False)
            seed = 2000.+31*(time-50)+.03*(time-50)**2
            k = 2*np.pi*RF[ch]/C
            amp = (.3+ch*.13)*np.exp(.7j*ch)
            raw = amp*np.exp(-1j*k*path)
            deramped = raw*np.exp(1j*k*(seed-2000.))
            r, recovered, _ = projection(deramped, np.ones(len(u)), -k*(path-(seed-2000.)))
            self.assertLess(np.max(abs(r)), 1e-11)
            self.assertLess(abs(recovered-amp), 1e-11)
            # A different seed's constant is exactly a nuisance phase, not a delay.
            r2, _, _ = projection(deramped, np.ones(len(u)), -k*(path-(seed-1730.)))
            self.assertLess(np.max(abs(r2)), 1e-11)

    def test_kinematic_penalty_uses_seconds(self):
        p = np.zeros((4, 3)); p[0] = [.5, .2, .08]
        velocity = np.array([30., -20., 1.]); p[1] = velocity*15/100
        scale = [100., 2., .5]
        penalty = motion_penalty(3, scale)@p.ravel()
        self.assertAlmostEqual(np.sum(penalty**2), np.sum(velocity**2)/100**2, places=12)
        self.assertLess(np.max(abs(penalty[61*3:])), 1e-12)

    def test_full_physical_likelihood_jacobian_with_kinematics(self):
        channels = []
        coarse = {'t':np.linspace(-14, 14, 30), 'channel':np.arange(30)%5,
                  'sigma_l':np.full(30, 3.)}
        coarse['length'] = path_model(self.state, coarse['t']/15, coarse['channel'], self.sites, 3, False)[0]
        for ch in range(5):
            u = np.linspace(-.98, .98, 31)+ch*.00001
            path, _ = path_model(self.state, u, np.full(len(u), ch), self.sites, 3, False)
            k = 2*np.pi*RF[ch]/C
            seed = 440*u
            channels.append({'u':u, 'time':50+15*u, 'k':k, 'seed_delta':seed,
                             'variance':np.ones(len(u)), 'value':np.exp(-1j*k*(path-seed))})
        p = np.r_[self.state, np.zeros(9)]
        fun = physical_problem(channels, coarse, self.sites, 3, True, motion=(100., 2., .5))
        j = fun(p, True); numerical = central_jacobian(fun, p, 1e-7)
        self.assertLess(np.linalg.norm(j-numerical)/np.linalg.norm(j), 1e-6)


if __name__ == '__main__':
    unittest.main()
