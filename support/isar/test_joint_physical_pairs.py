import unittest

import numpy as np

from joint_physical_pairs import JointPhysicalPairs
from test_common_physical_pair import fixture


class JointPhysicalPairTests(unittest.TestCase):
    def test_distinct_interval_states_are_not_crossed_or_implicitly_tied(self):
        first,p=fixture();state=np.r_[np.sin(np.arange(10))*.03,-.42]
        second,q=fixture(state);model=JointPhysicalPairs([first,second])
        truth=np.r_[p[:18],p[18:],q[18:]]
        residual,_=model.evaluate(truth)
        self.assertLess(np.linalg.norm(residual[:-sum(model.prior)]),1e-10)
        prediction=model.prediction(truth,[[~b.training for b in arc.bands] for arc in model.arcs])
        self.assertTrue(all(r['heldout_explained_fraction']>1-1e-12 for arc in prediction for r in arc))
        crossed=np.r_[p[:18],q[18:],p[18:]]
        self.assertGreater(model.objective(crossed)[0],model.objective(truth)[0]+1e-3)
        x=truth+np.sin(np.arange(model.count))*.0001;_,jac=model.evaluate(x)
        direction=np.cos(np.arange(model.count)+.4)*.01;step=5e-4
        finite=(model.evaluate(x+step*direction)[0]-model.evaluate(x-step*direction)[0])/(2*step)
        self.assertLess(np.linalg.norm(finite-jac@direction)/np.linalg.norm(finite),2e-5)

    def test_shared_prior_and_full_residual_derivative(self):
        first,p=fixture();second,q=fixture();model=JointPhysicalPairs([first,second])
        truth=np.r_[p[:18],p[18:],q[18:]];x=truth+np.sin(np.arange(model.count)+.2)*.0001
        residual,jac=model.evaluate(x)
        expected=sum(arc.objective(x[index])[0]*arc.energy for arc,index in zip(model.arcs,model.indices))-x[:18]@x[:18]
        self.assertAlmostEqual(residual@residual,expected/model.energy,places=14)
        direction=np.cos(np.arange(model.count))*.01
        for step in (1e-3,5e-4):
            finite=(model.evaluate(x+step*direction)[0]-model.evaluate(x-step*direction)[0])/(2*step)
            self.assertLess(np.linalg.norm(finite-jac@direction)/np.linalg.norm(finite),2e-5)

    def test_known_truth_shared_calibration_recovery(self):
        first,p=fixture();second,q=fixture();model=JointPhysicalPairs([first,second])
        truth=np.r_[p[:18],p[18:],q[18:]];seed=truth+np.sin(np.arange(model.count))*.00005
        fitted,audit=model.fit(seed,max_evaluations=100)
        self.assertLess(audit['objective'],1e-17)
        np.testing.assert_allclose(fitted,truth,atol=3e-6,rtol=0)
        prediction=model.prediction(fitted,[[~b.training for b in arc.bands] for arc in model.arcs])
        self.assertTrue(all(r['heldout_explained_fraction']>1-1e-12 for arc in prediction for r in arc))

    def test_joint_carrier_initializers_preserve_shared_antennas_and_exact_phase(self):
        first,p=fixture();second,q=fixture();model=JointPhysicalPairs([first,second])
        independent=[first.mapping(p)[0],second.mapping(q)[0]]
        records=model.phase_initializers(independent)
        self.assertGreater(len(records),1)
        for record in records:
            parameters=record['parameters']
            self.assertTrue(np.all(parameters>=model.lower-1e-12))
            self.assertTrue(np.all(parameters<=model.upper+1e-12))
            for index in model.indices:np.testing.assert_array_equal(parameters[index][:18],parameters[:18])
        best=min(records,key=lambda r:model.objective(r['parameters'])[0])
        self.assertLess(model.objective(best['parameters'])[0],1e-15)


if __name__=='__main__':unittest.main()
