import unittest

import numpy as np

from phase_rigid_patch import PhasePatch
from test_rigid_patch import scenario


def problem():
    bands=[]
    for rf,g,a in zip([177.5e6,184.5e6,191.625e6,219.5e6,226.5e6],
            [.6+.2j,.2+.4j,-.2+.7j,1.,-.4+.8j],[[1.,.4,2.],[.7,3.,1.],[2.,1.,.5],[1.,1.,1.],[2.,.5,1.]]):
        kin,b,p=scenario(rf,g,point_amplitude=a);bands.append(b)
    return PhasePatch(bands,p),p


class PhasePatchTests(unittest.TestCase):
    def test_phase_only_scene_constraint_retains_band_dependent_scattering_amplitudes(self):
        model,p=problem();r=model.evaluate(model.initial)
        self.assertLess(abs(r['value']),1e-13)
        for c in model.validation(model.initial):self.assertLess(c['unused_rss']/c['unused_data_energy'],1e-12)

    def test_profile_gradient_includes_shared_phases_and_positive_amplitudes(self):
        model,p=problem();q=model.initial.copy();q[0]+=.002;q[3]+=.003;q[5]+=.2
        result=model.evaluate(q)
        for j,h in [(0,1e-6),(3,1e-6),(5,1e-4),(8,1e-4)]:
            delta=np.eye(model.size)[j]*h
            finite=(model.evaluate(q+delta,derivatives=False)['value']-model.evaluate(q-delta,derivatives=False)['value'])/(2*h)
            self.assertLess(abs(finite-result['gradient'][j])/max(abs(finite),1e-8),5e-4)


if __name__=='__main__':unittest.main()
