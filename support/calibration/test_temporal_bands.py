import copy
import unittest

import numpy as np

from coherent_fit import CoherentProblem
from geometry import central_jacobian,C
from road_model import RoadProblem
from temporal_bands import TemporalBandProblem
from temporal_validation import IndependentBlocks,predict
import test_scatterer_validation as fixture_module


class TemporalBandsTest(unittest.TestCase):
    def setUp(self):
        fixture=fixture_module.RigidModelTest();fixture.setUp()
        source=fixture.problem
        channels=[{'seed':f.seed,'value':f.data['value'][:,4],
                   'operator':f.q,'time':f.time,'k':f.k} for f in source.fields]
        self.problem=RoadProblem(channels,{'t':np.zeros(1)},source.sites,source.fields,source.model)
        boundaries=np.linspace(43,57,6)
        phases=np.array([.3,1.8,-.7,2.4,-1.2])
        for ch,field in enumerate(source.fields):
            path=source.model.paths(fixture.state,field.time,np.full(len(field.time),ch),source.sites,True)[0]
            lag=field.offsets[None,:]-(path-field.seed_path)[:,None]*8e6/C
            phase=np.exp(-1j*field.k*(path-(field.seed_path-field.seed['path'][0])))
            block=np.searchsorted(boundaries,field.time,side='right')-1
            amplitude=(.8+.1*ch)*(1+.2*np.sin(block+ch))*np.exp(1j*phases[block])
            field.data={**field.data,'value':amplitude[:,None]*phase[:,None]*field.pulse(lag)}
            field.y=field.apply(field.data['value']).ravel()
        self.joint=CoherentProblem([self.problem])
        original=self.joint.initial([fixture.state[:source.model.count]],fixture.state[-9:])
        original[self.joint.beta_offset:self.joint.magnitude_offset]=0
        self.variable=TemporalBandProblem(self.joint,3.,True)
        self.state=self.variable.initial(original)

    def test_known_fading_reflectivity_is_exact_without_band_phase_freedom(self):
        residual=self.variable.residual(self.state,include_motion=False,include_site_prior=False)
        np.testing.assert_allclose(residual,0,atol=3e-14)
        constant=TemporalBandProblem(self.joint,3.,False)
        self.assertGreater(np.linalg.norm(constant.residual(self.state,include_motion=False,include_site_prior=False)),1.)
        self.assertEqual(constant.count,self.variable.count)
        for a,b in zip(constant.blocks[0],self.variable.blocks[0]):
            for x,y in zip(a,b):np.testing.assert_array_equal(x.y,y.y)

    def test_geometry_and_calibration_derivatives_through_profiled_blocks(self):
        actual=self.variable.residual(self.state,True,False,False)
        numeric=central_jacobian(lambda p:self.variable.residual(p,False,False,False),self.state,1e-4)
        self.assertLess(np.linalg.norm(actual-numeric)/np.linalg.norm(actual),5e-7)

    def test_withheld_time_samples_are_predicted_from_training_coefficients(self):
        def split(parity):
            source=copy.copy(self.problem);source.fields=[]
            for field in self.problem.fields:
                f=copy.copy(field)
                f.selected=np.floor(f.time-43).astype(int)%2==parity
                source.fields.append(f)
            return TemporalBandProblem(CoherentProblem([source]),3.,True)
        train,test=split(0),split(1)
        _,amplitudes=train.target(self.state,0)
        for block,(data,templates,_) in enumerate(test.block_models(self.state,0)):
            for ch,(y,u) in enumerate(zip(data,templates)):
                np.testing.assert_allclose(amplitudes[block,ch]*u,y,atol=2e-12)
        for fields in train.blocks[0]:
            for f in fields:self.assertTrue(np.all(f.q[:,~f.selected]==0))
        self.assertLess(predict(train,test,self.state)['prediction_error'],1e-22)

    def test_independent_control_retains_a_disagreeing_band(self):
        field=self.problem.fields[3]
        field.data={**field.data,'value':field.data['value']*np.exp(.8j)}
        field.y=field.apply(field.data['value']).ravel()
        independent=IndependentBlocks(self.joint,3.)
        shared=TemporalBandProblem(self.joint,3.)
        np.testing.assert_allclose(independent.residual(self.state,False,False,False),0,atol=3e-14)
        self.assertGreater(np.linalg.norm(shared.residual(self.state,False,False,False)),.1)
        original=[f.y.copy() for f in shared.blocks[0][0]]
        optimistic=shared.expected_signal(self.state,all_band_powers=True)
        np.testing.assert_allclose(optimistic.residual(self.state,False,False,False),0,atol=3e-14)
        _,amplitudes=optimistic.target(self.state,0)
        self.assertTrue(np.all(abs(amplitudes[:,3])>.1))
        for f,y in zip(shared.blocks[0][0],original):np.testing.assert_array_equal(f.y,y)
        actual=independent.residual(self.state,True,False,False)
        numeric=central_jacobian(lambda p:independent.residual(p,False,False,False),self.state,1e-4)
        self.assertLess(np.linalg.norm(actual-numeric)/np.linalg.norm(actual),5e-7)


if __name__=='__main__':unittest.main()
