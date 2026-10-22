"""Road topology, short-window covariance and exact phase/Jacobian regressions."""
import unittest
import numpy as np
from geometry import central_jacobian,observables
from complex_likelihood import gate,projected_profile
from road_geometry import RoadCurve,connected_chains
from road_model import RoadTrajectory,RoadProblem
from road_search import RoadScreen,OPTIONS
import itertools
from calibration_solver import solve
from cosited_paths import PathHistory
from coherent_fit import constant_path_direction,transport_path_shift
from types import SimpleNamespace
from road_validation import subset,evaluate
from coherent_fit import CoherentProblem
from receiver_fit import ReceiverProblem,ProfiledReceiverProblem
from profiled_bands import ProfiledBandProblem
from range_fit import Field,Pulse
from geometry import RF,C
from refit_sites import select_targets


class RoadTests(unittest.TestCase):
    def setUp(self):
        self.road=RoadCurve({'enu_m':[[0,-2000,0],[300,-1950,2],[600,-1800,3],[900,-1700,4],
                                  [1200,-1650,7]],'directions':[-1,1],'posted_speed_mps':20.})
        self.seed={'epoch':50.,'scale':15.,'duration_s':14.,'path':[1500.,70.]}
        self.sites=np.array([[31000.,-3000.,700.],[31200.,-2800.,720.],[31100.,-2700.,710.]])
        self.model=RoadTrajectory(self.seed,self.road,[600,70,1,0],[0,2,1,0,2],np.zeros((5,3)))

    def test_configured_primary_backup_combinations(self):
        choices=set(itertools.product(*OPTIONS))
        self.assertEqual(len(choices),32)
        self.assertTrue(all(a[c] in (0,1) for a in choices for c in (0,2,3)))
        self.assertTrue(all(a[c] in (0,2) for a in choices for c in (1,4)))

    def test_joint_fit_uses_each_roads_own_bounds(self):
        bounds=[(np.array([-5.,-7.]),np.array([4.,10.])),
                (np.array([-1.,-2.]),np.array([3.,6.]))]
        problems=[SimpleNamespace(model=SimpleNamespace(count=2,bounds=lambda b=b:b),
            channel_data=lambda ch:np.ones(3)) for b in bounds]
        joint=CoherentProblem(problems)
        low,high=joint.bounds()
        np.testing.assert_array_equal(low[9:13],[-5,-7,-1,-2])
        np.testing.assert_array_equal(high[9:13],[4,10,3,6])

    def test_road_bound_covers_extrema_between_sampling_points(self):
        # The minimum is deliberately between the previous 121 probe points.
        centre=.007;coarse=[.05+1000*centre**2,-2000*centre,1000,0]
        model=RoadTrajectory(self.seed,self.road,coarse,[0]*5,np.zeros((5,3)))
        low,_=model.bounds();state=np.zeros(model.count);state[:model.radial_count]=low[0]
        time=self.seed['epoch']+np.array([centre])*model.half
        base,basis=model.length_basis(time)
        self.assertGreater(float((base+basis@state[:model.radial_count])[0]),0)
        model.positions(state,time,self.sites)

    def test_heldout_calibration_parameter_mapping_keeps_shared_phase(self):
        problems=[SimpleNamespace(model=SimpleNamespace(count=5),channel_data=lambda ch:np.ones(3)) for _ in range(4)]
        problem=CoherentProblem(problems,site_prior_sigma=np.array([[8,10,12],[9,11,13],[7,12,14]]))
        state=np.arange(problem.count,dtype=float)
        reduced,p=subset(problem,state,[3,0,2])
        np.testing.assert_array_equal(reduced.prior_weights,problem.prior_weights)
        np.testing.assert_array_equal(p[:9],state[:9])
        np.testing.assert_array_equal(p[reduced.beta_offset:reduced.magnitude_offset],state[problem.beta_offset:problem.magnitude_offset])
        for new,old in enumerate([3,0,2]):
            np.testing.assert_array_equal(p[9+5*new:14+5*new],state[9+5*old:14+5*old])
            np.testing.assert_array_equal(p[reduced.magnitude_offset+4*new:reduced.magnitude_offset+4*(new+1)],
                state[problem.magnitude_offset+4*old:problem.magnitude_offset+4*(old+1)])

    def test_heldout_range_origin_calibration_is_shared_and_preserved(self):
        problems=[SimpleNamespace(model=SimpleNamespace(count=5),channel_data=lambda ch:np.ones(3)) for _ in range(4)]
        problem=CoherentProblem(problems,range_bias=True)
        state=np.arange(problem.count,dtype=float)
        reduced,p=subset(problem,state,[2,0])
        self.assertTrue(reduced.range_bias)
        np.testing.assert_array_equal(p[reduced.bias_offset:],state[problem.bias_offset:])
        np.testing.assert_array_equal(p[reduced.beta_offset:reduced.magnitude_offset],
                                      state[problem.beta_offset:problem.magnitude_offset])

    def test_serialized_site_refit_subset_keeps_every_retained_parameter(self):
        problems=[SimpleNamespace(model=SimpleNamespace(count=15),channel_data=lambda ch:np.ones(3)) for _ in range(4)]
        problem=CoherentProblem(problems)
        state=np.arange(problem.count,dtype=float)
        saved={'assignment':[0,0,0,0,2], 'labels':['A','B','C','D'], 'state':state.tolist(),
               'site_prior_sigma_m':[10,10,10], 'site_positions_enu_m':np.zeros((3,3)).tolist(),
               'channel_offsets_enu_m':np.zeros((5,3)).tolist(),
               'selected_roads':[{'route':i} for i in range(4)], 'cost':12345.}
        actual=select_targets(saved,['D','A','C'])
        _,expected=subset(problem,state,[3,0,2])
        np.testing.assert_array_equal(actual['state'],expected)
        self.assertEqual(actual['selected_roads'],[{'route':i} for i in [3,0,2]])
        self.assertNotIn('cost',actual)

    def test_combined_score_cannot_hide_a_discarded_measurable_band(self):
        u=np.ones(20,complex)
        problem=SimpleNamespace(model=SimpleNamespace(count=1),
            channel_data=lambda ch:u*(-1 if ch==4 else 1),
            channel_model=lambda state,ch,*args,**kwargs:(u,None))
        joint=CoherentProblem([problem])
        state=np.zeros(joint.count)
        state[joint.magnitude_offset+3]=np.log(np.finfo(float).tiny)
        result=evaluate(joint,joint,state,problem,problem,np.zeros(1))
        self.assertLess(result['heldout_loss_db'],1.)
        self.assertAlmostEqual(abs(result['phase_contrast_degrees'][4]),180.)
        self.assertLess(result['shared_to_independent_amplitude_ratio'][4],1e-100)
        self.assertAlmostEqual(result['same_path_independent_band_energy'][4],20.)

    def test_damped_solver_agrees_with_bounded_linear_solution(self):
        rng=np.random.default_rng(361)
        a=rng.normal(size=(50,5));truth=np.array([.3,-.2,.7,-.1,.1]);y=a@truth
        def residual(x,jacobian=False):return a if jacobian else a@x-y
        fit=solve(residual,np.zeros(5),(np.full(5,-1.),np.full(5,1.)))
        self.assertTrue(fit.success)
        np.testing.assert_allclose(fit.x,truth,atol=1e-8)
        from scipy.optimize import lsq_linear
        bounds=(np.full(5,-.4),np.full(5,.4))
        constrained=solve(residual,np.zeros(5),bounds)
        reference=lsq_linear(a,y,bounds=bounds,tol=1e-12)
        np.testing.assert_allclose(constrained.x,reference.x,atol=1e-6)

    def test_damped_solver_matches_nonlinear_phase_reference(self):
        from scipy.optimize import least_squares
        rng=np.random.default_rng(617)
        t=np.linspace(-1,1,200);basis=np.c_[np.ones(len(t)),t,t*t]
        truth=np.array([.3,.7,-.15]);y=np.exp(1j*(basis@truth))
        y+=.01*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
        def residual(p,jacobian=False):
            u=np.exp(1j*(basis@p))
            v=1j*u[:,None]*basis if jacobian else u-y
            return np.r_[v.real,v.imag]
        bounds=(np.full(3,-2.),np.full(3,2.));initial=np.zeros(3)
        actual=solve(residual,initial,bounds)
        expected=least_squares(residual,initial,jac=lambda p:residual(p,True),bounds=bounds,
                               ftol=1e-12,xtol=1e-12,gtol=1e-12)
        np.testing.assert_allclose(actual.x,expected.x,atol=1e-7)
        self.assertAlmostEqual(actual.cost,expected.cost,places=11)

    def test_phase_search_sufficient_statistics_preserve_objective_gradient(self):
        rng=np.random.default_rng(524)
        base=rng.normal(size=(5,37))+1j*rng.normal(size=(5,37))
        data=rng.normal(size=(5,37))+1j*rng.normal(size=(5,37))
        slopes=rng.normal(size=(5,3));norm=np.sqrt(np.sum(abs(base)**2,axis=1))
        observed=np.sum(base.conj()*data,axis=1)/norm
        constant=np.sum(abs(data)**2)-np.sum(abs(observed)**2)
        for x in rng.normal(size=(5,3)):
            phase=np.exp(1j*(slopes@x))
            u=(base*phase[:,None]).ravel()
            du=(1j*base[:,:,None]*phase[:,None,None]*slopes[:,None,:]).reshape(-1,3)
            full,jf,_=projected_profile(data.ravel(),u,du)
            v=norm*phase
            reduced,jr,_=projected_profile(observed,v,1j*v[:,None]*slopes)
            self.assertAlmostEqual(np.vdot(full,full).real/2-np.vdot(reduced,reduced).real/2,constant,places=10)
            np.testing.assert_allclose(np.real(full.conj()@jf),np.real(reduced.conj()@jr),atol=2e-12)

    def test_short_path_spline_contains_no_unobserved_controls(self):
        model=PathHistory(self.seed,3.)
        time=np.linspace(43.001,56.999,200)
        _,basis=model.basis(time)
        singular=np.linalg.svd(basis,compute_uv=False)
        self.assertEqual(np.linalg.matrix_rank(basis),model.count)
        self.assertGreater(singular[-1],.1)

    def test_ambiguity_step_is_constant_excess_path_on_curved_road(self):
        t=np.linspace(43.01,56.99,500)
        problem=SimpleNamespace(reference_channel=1,model=self.model,sites=self.sites,
                                channels=[{'time':t} for _ in range(5)])
        p=np.zeros(self.model.count+9)
        direction=constant_path_direction(problem,p)
        _,j=self.model.paths(p,t,np.ones(len(t),int),self.sites,True)
        actual=j[:,:self.model.radial_count]@direction
        self.assertLess(np.max(abs(actual-1)),2e-4)
        # A constant distance along a curved road is not a constant path shift.
        naive=j[:,:self.model.radial_count].sum(axis=1)
        self.assertGreater(np.ptp(naive),.01)
        moved,error=transport_path_shift(problem,p,15.)
        self.assertLess(error,.001)
        original=self.model.paths(p,t,np.ones(len(t),int),self.sites,True)[0]
        shifted=self.model.paths(moved,t,np.ones(len(t),int),self.sites,True)[0]
        np.testing.assert_allclose(shifted-original,15.,atol=.001)

    def test_connectivity_and_oneway_survive_chain_orientation(self):
        def way(i,nodes,name='Test',oneway='no'):
            return {'id':i,'nodes':nodes,'tags':{'name':name,'highway':'residential','oneway':oneway}}
        ways=[way(1,[2,1],oneway='yes'),way(2,[3,2],oneway='yes'),way(3,[3,4],'Other'),
              way(4,[10,11])]
        chains=list(connected_chains(ways))
        joined=next(c for c in chains if len(c['ways'])==2)
        self.assertEqual(joined['nodes'],[1,2,3]);self.assertEqual(joined['directions'],[-1])
        self.assertEqual(len(chains),3)
        self.assertFalse(any(3 in c['nodes'] and 4 in c['nodes'] and 2 in c['nodes'] for c in chains))

    def test_short_track_gate_preserves_covariance_and_masks(self):
        t=np.linspace(43.01,56.99,700);variance=1+.3*np.sin(t)
        selected=np.floor(t-43).astype(int)%2==0
        q,_=gate(t,variance,1.,50.,14.,selected)
        np.testing.assert_allclose((q*variance)@q.conj().T,np.eye(len(q)),atol=2e-12)
        self.assertTrue(np.all(q[:,~selected]==0))
        self.assertEqual(len(q),29)

    def test_road_phase_geometry_derivative_and_rate(self):
        rng=np.random.default_rng(387)
        p=np.r_[rng.normal(0,.2,self.model.count),rng.normal(0,.03,9)]
        t=np.linspace(43.1,56.9,45);channel=np.arange(len(t))%5
        path,j=self.model.paths(p,t,channel,self.sites,True)
        numerical=central_jacobian(lambda q:self.model.paths(q,t,channel,self.sites,True)[0],p,1e-4)
        np.testing.assert_allclose(j,numerical,atol=1e-7)
        x,_,tx=self.model.positions(p,t,self.sites,True)
        v=(self.model.positions(p,t+.0001,self.sites,True)[0]-self.model.positions(p,t-.0001,self.sites,True)[0])/.0002
        _,rate=observables(x,v,tx[self.model.channel_sites[channel]],np.zeros(3))
        finite=(self.model.paths(p,t+.0001,channel,self.sites,True)[0]-self.model.paths(p,t-.0001,channel,self.sites,True)[0])/.0002
        np.testing.assert_allclose(rate,finite,atol=1e-7)

    def test_receiver_position_derivative_preserves_direct_path_reference(self):
        p=np.zeros(self.model.count+9);t=np.linspace(44,56,21);channels=np.arange(len(t))%5
        self.model.receiver_enu_m=np.array([3.,-5.,2.])
        expected=self.model.receiver_derivative(p,t,channels,self.sites,True)
        def path(receiver):
            self.model.receiver_enu_m=receiver
            return self.model.paths(p,t,channels,self.sites,True)[0]
        np.testing.assert_allclose(expected,central_jacobian(path,self.model.receiver_enu_m,.001),atol=8e-9)

    def test_shared_receiver_complex_likelihood_jacobian(self):
        rng=np.random.default_rng(915);time=np.linspace(43.1,56.9,71)
        spectrum=np.zeros(8192);spectrum[np.arange(-3408,3409)%8192]=1.
        pulse=Pulse(np.fft.ifft(spectrum)[np.arange(-64,65)%8192])
        channels=[];fields=[]
        for ch in range(5):
            centre=self.model.paths(np.zeros(self.model.count),np.array([50.]),np.array([ch]),self.sites)[0][0]
            seed=self.seed|{'path':[float(centre),70.],'rf_hz':RF[ch]}
            q,b=gate(time,np.ones(len(time)),.2,50.,14.)
            channels.append({'seed':seed,'time':time,'operator':q,'basis':b,'k':2*np.pi*RF[ch]/C,
                             'value':rng.normal(size=len(time))+1j*rng.normal(size=len(time))})
            values=rng.normal(size=(len(time),9))+1j*rng.normal(size=(len(time),9))
            fields.append(Field({'time':time,'offset_samples':np.arange(-4,5),'variance':np.ones(values.shape),
                'value':values},pulse,seed,.2))
        pr=RoadProblem(channels,{'t':np.zeros(1)},self.sites,fields,self.model)
        joint=CoherentProblem([pr],site_prior_sigma=np.array([10.,10.,10.]))
        wrapper=ReceiverProblem(joint,[15.,15.,7.5])
        state=np.r_[joint.initial([np.zeros(self.model.count)]),[.1,-.2,.15]]
        # Motion finite differences have their own larger-step regression;
        # isolate the carrier/range likelihood here to avoid differencing
        # numerical third derivatives of absolute road coordinates.
        actual=wrapper.residual(state,True,include_motion=False)
        # A step sweep separates truncation from subtraction roundoff in the
        # approximately 30 km path lengths; 1e-4 is inside that stable region.
        numerical=central_jacobian(lambda q:wrapper.residual(q,include_motion=False),state,1e-4)
        self.assertLess(np.linalg.norm(actual-numerical)/np.linalg.norm(actual),3e-7)
        np.testing.assert_array_equal(joint.problems[0].model.receiver_enu_m,np.zeros(3))
        profiled=ProfiledReceiverProblem(joint,[15.,15.,7.5])
        reduced=np.r_[ProfiledBandProblem(joint).initial(state[:-3]),state[-3:]]
        actual=profiled.residual(reduced,True,include_motion=False)
        numerical=central_jacobian(lambda q:profiled.residual(q,include_motion=False),reduced,1e-4)
        self.assertLess(np.linalg.norm(actual-numerical)/np.linalg.norm(actual),1e-6)
        expanded,_=profiled.expanded_state(reduced)
        reference=wrapper.residual(np.r_[expanded,reduced[-3:]],include_motion=False)
        np.testing.assert_allclose(profiled.residual(reduced,include_motion=False),reference,atol=1e-11)
        np.testing.assert_array_equal(joint.problems[0].model.receiver_enu_m,np.zeros(3))

        # Different receive phase centres must move both the echo and the
        # direct-reference distances. All targets share those same centres.
        offsets=np.zeros((5,3));offsets[[0,2,3,4]]=rng.normal(size=(4,3))*.2
        banded=ProfiledReceiverProblem(joint,[15.,15.,7.5],[1.,1.,.5])
        q=banded.initial(state[:-3],state[-3:]*wrapper.sigma,offsets)
        actual=banded.residual(q,True,include_motion=False)
        numeric=central_jacobian(lambda v:banded.residual(v,include_motion=False),q,1e-4)
        self.assertLess(np.linalg.norm(actual-numeric)/np.linalg.norm(actual),1e-6)
        common,restored=banded.coordinates(q)
        np.testing.assert_allclose(restored,offsets,atol=1e-15)
        expanded,_=banded.expanded_state(q)
        wrapper.joint.problems[0].model.receiver_channel_offsets_enu_m=offsets.copy()
        reference=wrapper.residual(np.r_[expanded,common/wrapper.sigma],include_motion=False)
        np.testing.assert_allclose(banded.residual(q,include_motion=False),
                                   np.r_[reference,q[banded.offset+3:]],atol=1e-11)
        np.testing.assert_array_equal(joint.problems[0].model.receiver_channel_offsets_enu_m,np.zeros((5,3)))

    def test_frequency_dependent_receiver_matches_independent_bistatic_paths(self):
        p=np.zeros(self.model.count);time=np.linspace(44,56,23)
        channels=np.arange(len(time))%5
        self.model.receiver_enu_m=np.array([3.,-5.,2.])
        offsets=np.array([[.3,-.4,.2],[0,0,0],[-.1,.2,.5],[.6,.1,-.4],[-.2,.3,.1]])
        self.model.receiver_channel_offsets_enu_m=offsets
        x,_,sites=self.model.positions(p,time,self.sites)
        transmitter=sites[self.model.channel_sites[channels]]+self.model.channel_offsets[channels]
        receivers=self.model.receiver_enu_m+offsets[channels]
        expected=np.linalg.norm(x-transmitter,axis=1)+np.linalg.norm(x-receivers,axis=1)-np.linalg.norm(transmitter-receivers,axis=1)
        actual,_=self.model.paths(p,time,channels,self.sites)
        np.testing.assert_allclose(actual,expected,atol=1e-10)
        baseline=self.model.receiver_enu_m.copy()
        def paths(shift):
            self.model.receiver_enu_m=baseline+shift
            return self.model.paths(p,time,channels,self.sites)[0]
        derivative=self.model.receiver_derivative(p,time,channels,self.sites)
        np.testing.assert_allclose(derivative,central_jacobian(paths,np.zeros(3),.001),atol=8e-9)

    def test_coarse_range_rate_jacobian(self):
        t=np.linspace(-6,6,29);channel=np.arange(len(t))%5
        obs={'t':t,'channel':channel,'description':{'start':43.,'end':57.},
             'sigma_l':np.ones(len(t)),'sigma_v':np.full(len(t),.1),
             'length':np.full(len(t),1800.),'rate':np.full(len(t),12.)}
        screen=RoadScreen(self.road,obs,self.sites[np.array([0,2,1,0,2])])
        p=np.array([600,70,1.,.2])
        np.testing.assert_allclose(screen.residual(p,True),central_jacobian(screen.residual,p,2e-4),atol=5e-8)

    def test_motion_penalty_uses_actual_short_duration(self):
        p=np.zeros(self.model.count)
        values,derivatives=self.model.kinematics(p,self.sites)
        v=(self.model.positions(p,np.array([49.999,50.001]),self.sites)[0][1]-
           self.model.positions(p,np.array([49.999,50.001]),self.sites)[0][0])/.002
        np.testing.assert_allclose(values[0][56],v,rtol=2e-5,atol=2e-5)
        numerical=central_jacobian(lambda q:self.model.kinematics(q,self.sites)[0][1].ravel(),p,1e-3)
        np.testing.assert_allclose(derivatives[1].reshape(-1,len(p)),numerical,atol=5e-8)


if __name__=='__main__':unittest.main()
