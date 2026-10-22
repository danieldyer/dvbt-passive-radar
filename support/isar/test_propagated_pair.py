import unittest

import numpy as np

from common_physical_pair import CommonPhysicalPair
from differentiable_sea import paths
from motion import ecef,enu_basis
from propagated_pair import primary_propagation
from test_common_physical_pair import fixture
from test_physical_pair import fixture as independent_fixture
from physical_amplitude_arc import PhysicalAmplitudeArc
from physical_pair import SharedPhysicalPair


def water():
    lat,lon=np.radians([-37.87,144.93]);radius=6371000.
    return dict(centre_ecef_m=ecef(lat,lon,0.)-radius*enu_basis(lat,lon)[:,2],radius_m=radius)


class PropagatedPairTests(unittest.TestCase):
    def test_explicit_background_support_survives_every_propagation_family(self):
        original,truth=independent_fixture()
        bands=[PhysicalAmplitudeArc(b.field,b.training,b.knots,b.geometry,b.channel,b.midpoint,
            stationary_paths=b.delays[::2]) for b in original.bands]
        direct=SharedPhysicalPair(bands)
        for family in ('direct','transmit','receive','both'):
            model=primary_propagation(direct,water(),family)
            for before,pair in zip(bands,model.pairs):
                for after in (pair.base,pair.secondary):
                    np.testing.assert_array_equal(after.delays,before.delays)
                    np.testing.assert_array_equal(after.design,before.design)
                    self.assertEqual(after.nuisance.rank,before.nuisance.rank)

    def test_four_ray_pair_predicts_unused_samples_and_profiles_exact_derivative(self):
        original,truth=independent_fixture()
        bands=[PhysicalAmplitudeArc(b.field,b.training,b.knots,b.geometry,b.channel,b.midpoint,
            stationary_paths=b.delays[::2]) for b in original.bands]
        model=primary_propagation(SharedPhysicalPair(bands),water(),'both')
        rng=np.random.default_rng(318)
        for pair,index in zip(model.pairs,model.indices):
            b=pair.base;signal=np.zeros(b.field.y.size,complex)
            for (raw,_),strength in zip(pair.response_blocks(truth[index]),
                    ([.7,1.2,.4,.9,1.,.6],[.2,.6,.8,.3,.5,.9])):
                signal+=b.reduced(raw[:,:,None]*b.amplitude_basis[:,None,:])@strength
            beta=(rng.normal(size=len(b.delays))+1j*rng.normal(size=len(b.delays)))*.02
            b.field.y=(signal+b.design@beta).reshape(b.field.y.shape)
        # Reconstruct to avoid retaining cached projections of the old data.
        model=primary_propagation(model,water(),'both')
        expected=truth[:model.geometry.count]@truth[:model.geometry.count]/model.energy
        self.assertAlmostEqual(model.objective(truth)[0],expected,places=12)
        prediction=model.prediction(truth,[~b.training for b in model.bands])
        self.assertTrue(all(p['heldout_explained_fraction']>1-1e-12 for p in prediction))
        x=truth+np.sin(np.arange(model.count))*.0002;_,jac=model.evaluate(x)
        direction=np.cos(np.arange(model.count))*.01
        for step in (1e-3,5e-4):
            finite=(model.evaluate(x+step*direction)[0]-model.evaluate(x-step*direction)[0])/(2*step)
            self.assertLess(np.linalg.norm(finite-jac@direction)/np.linalg.norm(finite),8e-5)

    def test_direct_limit_preserves_likelihood_bounds_and_derivative(self):
        original,truth=fixture();model=CommonPhysicalPair(primary_propagation(original.independent,water(),'direct'))
        np.testing.assert_array_equal(model.lower,original.lower)
        np.testing.assert_array_equal(model.upper,original.upper)
        x=truth+np.sin(np.arange(model.count))*.0002
        a,ja=original.evaluate(x);b,jb=model.evaluate(x)
        np.testing.assert_allclose(a,b,atol=2e-9,rtol=1e-7)
        np.testing.assert_allclose(ja,jb,atol=2e-7,rtol=1e-6)

    def test_common_phase_is_applied_to_the_physical_sum_of_rays(self):
        original,truth=fixture();surface=water()
        model=CommonPhysicalPair(primary_propagation(original.independent,surface,'transmit'))
        mapped,_=model.mapping(truth);physical=model.phase_map.physical_parameters(truth[:29]);g=model.geometry
        p,v,a,tx,rx=g.state(truth[:28])
        for b in model.bands:
            dt=b.field.data['row_time']-b.midpoint
            x=g.receiver+(p+dt[:,None]*v+.5*dt[:,None]**2*a)@g.rotation;velocity=(v+dt[:,None]*a)@g.rotation
            zero=np.zeros(x.shape+(1,));antenna_zero=np.zeros((3,1))
            length,rate,gain,*_=paths(x,velocity,g.receiver+tx[b.channel]@g.rotation,g.receiver+rx@g.rotation,
                surface['centre_ecef_m'],surface['radius_m'],zero,zero,antenna_zero,antenna_zero,b.field.metadata['rf_hz'])
            expected=sum(b.field.path_response(length[j]-b.field.data['centre_path_m'],-b.field.k*.001024*rate[j])*gain[j,:,None]
                         for j in (0,1))*np.exp(1j*physical[-1])
            np.testing.assert_allclose(b.response(mapped[:33])[0],expected,atol=2e-9,rtol=2e-9)
        # Nonzero residual checks the full profiled derivative, including
        # changing nonnegative brightness and secondary-echo coefficients.
        x=truth+np.sin(np.arange(model.count))*.0002;_,jac=model.evaluate(x)
        direction=np.cos(np.arange(model.count))*.01
        for step in (1e-3,5e-4):
            finite=(model.evaluate(x+step*direction)[0]-model.evaluate(x-step*direction)[0])/(2*step)
            self.assertLess(np.linalg.norm(finite-jac@direction)/np.linalg.norm(finite),6e-5)


if __name__=='__main__':unittest.main()
