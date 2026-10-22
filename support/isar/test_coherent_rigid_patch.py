import unittest

import numpy as np

from coherent_rigid_patch import CoherentPatch
from test_rigid_patch import scenario


def problem(shift=False):
    delta=np.array([[0.,0.,0.],[.4,-.6,0.],[-.7,.3,0.]]) if shift else None
    bands=[]
    for rf,g in zip([177.5e6,191.625e6,226.5e6],[.6+.2j,1.,-.4+.8j]):
        kin,b,p=scenario(rf,g,delta);bands.append(b)
    return CoherentPatch(bands,p,reference_band=1),p


class CoherentPatchTests(unittest.TestCase):
    def test_variable_projection_includes_motion_positions_and_band_gain_derivatives(self):
        model,p=problem();q=model.initial.copy();q[:5]+=np.array([.002,-.0001,0.,.003,.0001]);q[5]+=.2
        r=model.evaluate(q)
        for j,step in [(0,1e-6),(3,1e-6),(5,1e-4),(8,1e-4),(9,1e-5),(10,1e-5)]:
            delta=np.eye(model.size)[j]*step
            finite=(model.evaluate(q+delta,derivatives=False)['residual']-model.evaluate(q-delta,derivatives=False)['residual'])/(2*step)
            self.assertLess(np.linalg.norm(finite-r['jacobian'][:,j])/np.linalg.norm(finite),3e-4)

    def test_shared_scene_gain_factorization_predicts_all_unused_bands(self):
        model,p=problem();r=model.evaluate(model.initial,derivatives=False)
        self.assertLess(r['value']/model.energy,1e-15)
        for record in model.validation(model.initial):
            self.assertLess(record['unused_rss']/record['unused_data_energy'],1e-15)

    def test_subpixel_positions_are_recovered_in_the_coherent_fit(self):
        model,p=problem(True)
        lo=np.r_[[-1.,-.1,-.01,-1.,-.02],np.full(model.nposition,-2.),np.full(model.ng,-5.)]
        hi=-lo
        found,r=model.fit(bounds=(lo,hi),max_evaluations=100)
        self.assertLess(r['objective'],1e-12)
        for record in r['validation']:
            self.assertLess(record['unused_rss']/record['unused_data_energy'],1e-11)


if __name__=='__main__':unittest.main()
