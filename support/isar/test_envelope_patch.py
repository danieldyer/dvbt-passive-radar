import copy,unittest
import numpy as np
from envelope_patch import EnvelopePatch
from rigid_patch import PatchBand
from test_rigid_patch import scenario


class EnvelopeMotionTests(unittest.TestCase):
    def test_physical_motion_gradient_with_profiled_changing_magnitudes(self):
        bands=[];phase=np.array([.3,-.7,1.2]);truth=None
        for c,rf in enumerate([177.5e6,226.5e6]):
            _,b,truth=scenario(rf);f=copy.copy(b.field);b=PatchBand(f,b.kinematics,b.points,b.training,b.tx,b.rx);bands.append(b)
        model=EnvelopePatch(bands,[-15,15],10.)
        amplitude=np.array([[1,.2,.8,.4],[.3,1,.4,.6],[.8,.7,.1,.9]])
        for c,(b,basis) in enumerate(zip(bands,model.bases)):
            h,_=b.design(truth);coeff=(basis@amplitude.T)*np.exp(1j*(phase+.2*c))[None]
            b.field.y=np.einsum('trp,tp->tr',h,coeff)
        p=np.r_[truth+np.array([.001,-.0001,0.,.002,0.]),-.2,phase+.2]
        r=model.evaluate(p);scale=np.r_[.05,.005,.0005,.05,.005,1.,1.,1.,1.]
        for j in range(model.size):
            step=1e-5*scale[j];d=np.eye(model.size)[j]*step
            numeric=(model.evaluate(p+d,derivatives=False)['value']-model.evaluate(p-d,derivatives=False)['value'])/(2*step)
            self.assertLess(abs(numeric-r['gradient'][j]),max(2e-6,abs(numeric)*2e-4))
        exact=np.r_[truth,-.2,phase+.2];fit=model.evaluate(exact)
        self.assertLess(abs(fit['value']),1e-12)
        self.assertLess(sum(v['unused_rss'] for v in model.validation(exact,fit)),1e-12)
        offgrid=EnvelopePatch(bands,[-15,15],10.,refine_positions=True)
        position=np.r_[p,[.05,-.04,-.03,.02]];actual=offgrid.evaluate(position)
        for j in range(model.size,offgrid.size):
            step=1e-5;delta=np.eye(offgrid.size)[j]*step
            numeric=(offgrid.evaluate(position+delta,derivatives=False)['value']-offgrid.evaluate(position-delta,derivatives=False)['value'])/(2*step)
            self.assertLess(abs(numeric-actual['gradient'][j]),max(2e-6,abs(numeric)*2e-4))
        # The position/phase gauge is an invertible change of optimizer
        # coordinates; it cannot change predictions or add any band phase.
        plain=EnvelopePatch(bands,[-15,15],10.,refine_positions=True,position_phase_gauge=False)
        canonical=position.copy();offset=position[offgrid.phase_end:].reshape(-1,2)
        canonical[5+len(offgrid.free)+1:offgrid.phase_end]+=np.sum(offset*offgrid.position_phase_gauge[1:],axis=1)
        expected=plain.evaluate(canonical)
        self.assertAlmostEqual(actual['value'],expected['value'],places=12)
        for left,right in zip(actual['solutions'],expected['solutions']):np.testing.assert_allclose(left['coefficient'],right['coefficient'],rtol=1e-11,atol=1e-11)


if __name__=='__main__':unittest.main()
