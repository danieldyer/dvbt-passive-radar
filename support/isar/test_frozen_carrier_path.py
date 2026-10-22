import unittest
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from frozen_carrier_path import FrozenCarrier
from rigid_patch import profile_static


class FrozenCarrierTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(178);t=np.linspace(-1,1,101);basis=csr_matrix(np.c_[t,t*t,t**3])
        models=[];bands=[];truth=np.array([.12,-.08,.03]);groups=(t>=0).astype(int)
        for k in np.linspace(3.7,4.7,5):
            root=rng.uniform(.7,1.3,len(t));selected=np.arange(len(t))%3!=2
            model=rng.normal(size=(len(t),7))+1j*rng.normal(size=(len(t),7));models.append(model)
            data=model*np.exp(-1j*k*(basis@truth))[:,None]+root[:,None]*(.3+.2j)
            projected=profile_static(data,root,selected,groups)*selected[:,None]
            bands.append(SimpleNamespace(training=selected,groups=groups,b=basis,
                field=SimpleNamespace(root_information=root,k=k),data={True:projected}))
        return FrozenCarrier(bands,models),bands,models,truth

    def test_exact_projection_gradient_and_information(self):
        kernel,bands,models,_=self.fixture();at=np.array([.03,-.01,.02])
        expected=0.
        for b,model in zip(bands,models):
            prediction=model*np.exp(-1j*b.field.k*(b.b@at))[:,None]
            residual=profile_static(prediction,b.field.root_information,b.training,b.groups)*b.training[:,None]-b.data[True]
            expected+=np.vdot(residual,residual).real
        value,gradient=kernel.objective(at)
        self.assertAlmostEqual(value,expected/kernel.energy,places=14)
        for j in range(3):
            step=np.eye(3)[j]*1e-6
            numeric=(kernel.objective(at+step)[0]-kernel.objective(at-step)[0])/2e-6
            self.assertAlmostEqual(numeric,gradient[j],places=8)
        self.assertGreater(np.linalg.eigvalsh(kernel.information()).min(),0)

    def test_five_carriers_recover_one_physical_correction(self):
        kernel,bands,models,truth=self.fixture()
        result=minimize(kernel.objective,np.zeros(3),jac=True,method='BFGS',options=dict(gtol=1e-11))
        np.testing.assert_allclose(result.x,truth,atol=1e-8,rtol=0)
        # No excluded observations enter the cached sufficient statistics.
        changed=[]
        for b in bands:
            data=b.data[True].copy();data[~b.training]=1e9*(2+3j)
            changed.append(SimpleNamespace(**{**b.__dict__,'data':{True:data}}))
        other=FrozenCarrier(changed,models)
        np.testing.assert_array_equal(other.objective(truth)[1],kernel.objective(truth)[1])


if __name__=='__main__':unittest.main()
