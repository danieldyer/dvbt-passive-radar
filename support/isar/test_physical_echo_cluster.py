from copy import deepcopy
import unittest

import numpy as np

from amplitude_arc import AmplitudeArc
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_echo_cluster import PhysicalEchoCluster,EnvelopeOffsetArc
from test_common_physical_arc import fixture as single_fixture


def fixture():
    source,_=single_fixture();old=source.independent.bands;g=source.geometry
    def bands():
        p=[PhysicalAmplitudeArc(b.field,b.training,b.knots,g,b.channel,b.midpoint,stationary_paths=[]) for b in old]
        e=[AmplitudeArc(b.field,b.training,b.knots,stationary_paths=[]) for b in p]
        return p,e
    p,e=bands();model=PhysicalEchoCluster(p,e,[-3.,1.2,4.])
    truth=np.zeros(model.count);truth[:g.count]=.001*np.sin(np.arange(g.count))
    truth[g.count:model.primary_count]=np.linspace(-.7,.8,5)
    for c,index in enumerate(model.indices):
        truth[index[model.primary_count:]]=np.array([
            [-3.,.4+.1*c,-.4,.06],[1.2,-.8+.2*c,.3,-.09],[4.,1.7-.2*c,-.8,.1]]).ravel()
    strengths=np.array([[.7,1.2,.4,.9,1.,.6],[.3,.5,.2,.6,.4,.7],
        [.4,.3,.7,.5,.2,.6],[.6,.3,.5,.2,.8,.4]])
    for cluster,index in zip(model.clusters,model.indices):
        b=cluster.base;signal=np.zeros(b.field.y.size,complex)
        for (raw,_),a in zip(cluster.response_blocks(truth[index]),strengths):
            signal+=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])@a
        b.field.y=signal.reshape(b.field.y.shape)
    p,e=bands()
    return PhysicalEchoCluster(p,e,[-3.,1.2,4.]),truth


class PhysicalEchoClusterTests(unittest.TestCase):
    def test_identical_declared_static_design_is_shared_by_all_echoes(self):
        original,truth=fixture();primary=[];empirical=[]
        for b in original.bands:
            f=deepcopy(b.field)
            atmid=np.interp(b.midpoint,f.data['row_time'],f.data['centre_path_m'])
            design=np.column_stack([f.stationary(atmid+d).ravel() for d in (-80.,90.)])
            static=design@np.array([40.+17j,-27.+35j])
            f.y+=static.reshape(f.y.shape)
            primary.append(PhysicalAmplitudeArc(f,b.training,b.knots,b.geometry,b.channel,b.midpoint,stationary_design=design))
            empirical.append(AmplitudeArc(f,b.training,b.knots,stationary_design=design))
        model=PhysicalEchoCluster(primary,empirical,original.origins)
        residual,_=model.evaluate(truth)
        self.assertAlmostEqual(residual@residual*model.energy,.5*truth[:model.geometry.count]@truth[:model.geometry.count],places=10)
        self.assertTrue(all(p['heldout_explained_fraction']>1-1e-12 for p in model.prediction(truth,[~b.training for b in primary])))
        empirical[0].design=empirical[0].design*1.01
        with self.assertRaises(ValueError):PhysicalEchoCluster(primary,empirical,original.origins)

    def test_envelope_control_preserves_default_and_has_exact_combined_derivative(self):
        model,truth=fixture();physical=model.bands[2];wrapped=EnvelopeOffsetArc(physical)
        core=truth[:model.primary_count];q=np.r_[core,np.zeros(5)]
        a,da=physical.response(core);b,db=wrapped.response(q)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(da,db[:,:,:len(core)])
        q[len(core)+2]=.21;direction=np.cos(np.arange(len(q)))*.01;step=5e-4
        _,jac=wrapped.response(q)
        finite=(wrapped.response(q+step*direction)[0]-wrapped.response(q-step*direction)[0])/(2*step)
        self.assertLess(np.linalg.norm(finite-jac@direction)/np.linalg.norm(finite),4e-5)

    def test_envelope_shift_does_not_apply_a_geometric_carrier_delay(self):
        model,_=fixture();f=model.bands[0].field
        dt=f.data['row_time'];delta=.13*np.sin(dt);omega=f.data['centre_omega'];bias=.8
        envelope=f.path_response(delta,omega,range_bias_m=bias)
        delayed=f.path_response(delta+bias,omega)
        np.testing.assert_allclose(envelope,delayed*np.exp(1j*f.k*bias),atol=2e-12,rtol=2e-12)

    def test_physical_truth_and_proper_complex_prior_counted_once(self):
        model,truth=fixture();r,j=model.evaluate(truth)
        expected=.5*truth[:model.geometry.count]@truth[:model.geometry.count]/model.energy
        self.assertAlmostEqual(r@r,expected,places=18)
        self.assertTrue(all(v['heldout_explained_fraction']>1-1e-12
            for v in model.prediction(truth,[~b.training for b in model.bands])))
        probe=truth+.0002*np.sin(np.arange(model.count)+.3)
        _,jac=model.evaluate(probe);direction=.01*np.cos(np.arange(model.count))
        for step in (1e-3,5e-4):
            finite=(model.evaluate(probe+step*direction)[0]-model.evaluate(probe-step*direction)[0])/(2*step)
            self.assertLess(np.linalg.norm(finite-jac@direction)/np.linalg.norm(finite),3e-5)

    def test_unused_observations_do_not_enter_fit_or_conditional_information(self):
        model,truth=fixture();before=model.evaluate(truth)[0].copy();info=model.information(truth)
        primary=[];empirical=[]
        for b in model.bands:
            f=deepcopy(b.field);f.y[~b.training]*=np.exp(.8j)
            primary.append(PhysicalAmplitudeArc(f,b.training,b.knots,b.geometry,b.channel,b.midpoint,stationary_paths=[]))
            empirical.append(AmplitudeArc(f,b.training,b.knots,stationary_paths=[]))
        changed=PhysicalEchoCluster(primary,empirical,model.origins)
        np.testing.assert_array_equal(changed.evaluate(truth)[0],before)
        np.testing.assert_allclose(changed.information(truth)['matrix'],info['matrix'],rtol=1e-12,atol=1e-12)
        self.assertTrue(all(p['heldout_error']>1 for p in changed.prediction(truth,[~b.training for b in primary])))


if __name__=='__main__':unittest.main()
