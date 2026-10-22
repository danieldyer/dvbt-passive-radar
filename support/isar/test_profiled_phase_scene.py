import unittest
import numpy as np
from scipy.optimize import minimize
from profiled_phase_scene import ProfiledPhaseScene
from rigid_patch import profile_static


class ProfiledPhaseSceneTests(unittest.TestCase):
    def problem(self):
        rng=np.random.default_rng(1892);t=np.linspace(-1,1,151);basis=np.c_[t,t*t,t*t*t]
        x=rng.normal(size=5)+1j*rng.normal(size=5);truth=np.array([.13,-.08,.04]);ks=np.linspace(3.7,4.8,5)
        spectrum=np.array([.8,.7,1.2,1.1,1.])*np.exp(1j*np.array([.2,-.4,.8,.0,-.7]))
        design=[];data=[];roots=[];groups=[];selected=[]
        for c,k in enumerate(ks):
            h=rng.normal(size=(len(t),7,5))+1j*rng.normal(size=(len(t),7,5));root=rng.uniform(.3,2,len(t));group=(t>0).astype(int);mask=np.arange(len(t))%3!=2
            y=(h@x)*(spectrum[c]*np.exp(-1j*k*(basis@truth)))[:,None]+root[:,None]*(1+group[:,None])*(2+3j)
            design.append(h);data.append(y);roots.append(root);groups.append(group);selected.append(mask)
        model=ProfiledPhaseScene(design,data,roots,groups,selected,[basis]*5,ks,1e-7,spectrum)
        return model,truth,x

    def test_original_design_ridge_and_variable_projection_gradient(self):
        model,truth,x=self.problem();at=np.array([.02,-.03,.01]);value,g,report=model.objective(at,True)
        designs=[];data=[]
        for c,(h,y,r,group,mask,b) in enumerate(zip(model.design,model.data,model.root,model.groups,model.selected,model.basis)):
            raw=h*(model.spectrum[c]*np.exp(-1j*model.k[c]*(b@at)))[:,None,None]
            designs.append(profile_static(raw,r,mask,group)[mask].reshape(-1,h.shape[2]))
            data.append(profile_static(y,r,mask,group)[mask].ravel())
        a=np.concatenate(designs);y=np.concatenate(data)
        exact=np.linalg.lstsq(np.r_[a,np.sqrt(model.strength)*np.eye(a.shape[1])],np.r_[y,np.zeros(a.shape[1])],rcond=None)[0]
        np.testing.assert_allclose(report['coefficient'],exact,rtol=1e-12,atol=1e-12)
        step=1e-6;fd=[(model.objective(at+step*e)[0]-model.objective(at-step*e)[0])/(2*step) for e in np.eye(3)]
        np.testing.assert_allclose(g,fd,rtol=3e-8,atol=1e-9)

    def test_unknown_image_and_shared_path_recovery_predicts_unused_rows(self):
        model,truth,x=self.problem()
        result=minimize(model.objective,np.zeros(3),jac=True,method='BFGS',options=dict(gtol=1e-10,maxiter=100))
        value,g,report=model.objective(result.x,True)
        np.testing.assert_allclose(result.x,truth,atol=1e-8,rtol=0)
        np.testing.assert_allclose(report['coefficient'],x,atol=1e-8,rtol=0)
        scored=model.score(result.x,report['coefficient'],[~m for m in model.selected])
        self.assertLess(scored['cost']/scored['energy'],1e-14)
        self.assertLess(report['maximum_scene_normal_residual'],1e-12)


if __name__=='__main__':unittest.main()
