import unittest
import numpy as np
from full_path_scene import SceneBand, RefinedScenePath,ScenePath
from test_full_path_point import problem
from scene_phase_coordinates import transform


class TestCoordinates(unittest.TestCase):
    def test_fixed_source_path_phase_pivot_is_the_same_likelihood(self):
        points=np.array([[[0.,0.,0.],[20.,-30.,10.]],[[10.,-10.,0.],[-30.,45.,10.]]])
        bands=[]
        for _ in range(5):
            point=problem();bands.append(SceneBand(point,points,[0.,8.,point.t[-1]],0.))
        likelihood=ScenePath(bands);size=likelihood.n+5+likelihood.p-1
        matrix=transform(likelihood,size);x=np.linspace(-.02,.03,size);q=np.linalg.solve(matrix,x)
        np.testing.assert_allclose(matrix@q,x,atol=2e-16)
        self.assertAlmostEqual(np.linalg.det(matrix),1.,places=12)
        np.testing.assert_array_equal(matrix[:likelihood.n],np.eye(size)[:likelihood.n])
        self.assertGreater(np.max(abs(matrix-np.eye(size))),1.)

    def test_bijection_gradient_and_linear_constraints(self):
        points=np.array([[[0.,0.,0.],[20.,-30.,10.]],[[10.,-10.,0.],[-30.,45.,10.]]])
        bands=[]
        rng=np.random.default_rng(602)
        for c in range(5):
            point=problem();point.field.y=rng.normal(size=point.field.y.shape)+1j*rng.normal(size=point.field.y.shape)
            band=SceneBand(point,points,[0.,8.,point.t[-1]],0.)
            from rigid_patch import profile_static
            for training in [True,False]:
                mask=point.training if training else ~point.training
                point.data[training]=profile_static(point.field.y,point.field.root_information,mask)*mask[:,None]
                point.energy[training]=np.vdot(point.data[training],point.data[training]).real
            bands.append(band)
        likelihood=RefinedScenePath(bands);size=likelihood.phase_end+2*likelihood.p
        matrix=transform(likelihood,size);x=rng.normal(size=size)*.01
        q=np.linalg.solve(matrix,x)
        np.testing.assert_allclose(matrix@q,x,atol=2e-16)
        self.assertAlmostEqual(np.linalg.det(matrix),1.,places=12)
        value,g=likelihood.objective(matrix@q)
        direction=rng.normal(size=size);direction/=np.linalg.norm(direction)
        step=3e-5
        finite=(likelihood.objective(matrix@(q+step*direction))[0]-likelihood.objective(matrix@(q-step*direction))[0])/(2*step)
        self.assertLess(abs(finite-g@matrix@direction),3e-5)
        constraint=rng.normal(size=(6,size))
        np.testing.assert_allclose(constraint@x,(constraint@matrix)@q,atol=1e-15)


if __name__=='__main__':unittest.main()
