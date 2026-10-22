import copy,unittest
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from audit_ship_scattering_envelopes import linear_basis
from fixed_scene_reconstruction import fit_scene,score_scene


class FixedSceneReconstructionTests(unittest.TestCase):
    def fixture(self,extra_band_phase=0.):
        rng=np.random.default_rng(387);t=np.linspace(0,12,241);basis,_=linear_basis(t,[0,12],3.);basis=csr_matrix(basis)
        beta=np.array([.7,-.5,.2,0.,-1.1]);phi=np.array([.3,-.6]);selected=np.arange(len(t))%3!=2
        design=[];bands=[]
        for c in range(5):
            h=rng.normal(size=(len(t),7,2))+1j*rng.normal(size=(len(t),7,2));root=np.ones(len(t));groups=(t>6).astype(int)
            amplitude=basis@rng.uniform(.4,1.3,(basis.shape[1],2))
            y=np.sum(h*(amplitude*np.exp(1j*(beta[c]+phi+(extra_band_phase if c==2 else 0.))))[:,None,:],axis=2)
            y+=root[:,None]*(.1+.2j)*(groups[:,None]+1)
            design.append(h);bands.append(SimpleNamespace(field=SimpleNamespace(y=y,root_information=root),amplitude=basis,groups=groups))
        return design,bands,[selected.copy() for _ in bands],beta,phi

    def test_original_complex_excluded_prediction_and_phase_gauge(self):
        design,bands,selected,beta,phi=self.fixture()
        fitted=fit_scene(design,bands,selected,beta+.15,phi+np.array([.1,-.1]))
        check=score_scene(fitted,bands,[~s for s in selected])
        self.assertLess(check['cost']/check['energy'],1e-12)
        np.testing.assert_allclose(fitted.band_phase-fitted.band_phase[3],beta-beta[3],atol=1e-14,rtol=0)
        np.testing.assert_allclose(fitted.source_phase,np.r_[0.,phi[1:]-phi[0]],atol=1e-6,rtol=0)
        original=score_scene(fitted,bands,selected)
        self.assertLess(abs(original['cost']-fitted.cost)/original['energy'],1e-12)

    def test_unseen_data_cannot_change_fit_and_incompatible_band_phase_is_not_removed(self):
        design,bands,selected,beta,phi=self.fixture()
        before=fit_scene(design,bands,selected,beta,phi)
        changed=[]
        for b,mask in zip(bands,selected):
            other=copy.copy(b);other.field=copy.copy(b.field);other.field.y=b.field.y.copy();other.field.y[~mask]=1e7*(2+3j);changed.append(other)
        after=fit_scene(design,changed,selected,beta,phi)
        np.testing.assert_array_equal(before.band_phase,after.band_phase)
        for x,y in zip(before.amplitude,after.amplitude):np.testing.assert_array_equal(x,y)
        design,bands,selected,beta,phi=self.fixture(extra_band_phase=.8)
        incompatible=fit_scene(design,bands,selected,beta,phi)
        self.assertGreater(score_scene(incompatible,bands,[~s for s in selected])['cost']/incompatible.energy,.01)


if __name__=='__main__':unittest.main()
