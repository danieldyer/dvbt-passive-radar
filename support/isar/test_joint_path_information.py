import unittest

import numpy as np

from joint_path_information import functional_bounds,expected_jacobian,paths


class JointPathInformationTests(unittest.TestCase):
    def test_affine_delay_restriction_matches_full_gaussian_covariance_and_raw_norms(self):
        from test_flexible_echo_cluster import cubic_scene
        from relative_scatterer_delay import RelativeScattererDelay
        model,p=cubic_scene();reduced=RelativeScattererDelay(model,1,2,[(0,1,2)])
        transform=reduced.transform;original,_=expected_jacobian(model,p)
        actual,norm=expected_jacobian(model,p,transform)
        np.testing.assert_allclose(actual,original@transform,atol=2e-12)
        expected=np.zeros(reduced.count)
        for cluster,index in zip(model.clusters,model.indices):
            b=cluster.base;local=p[index];v=cluster.evaluate(local);k=len(b.knots)
            raw=[]
            for (_,d),a in zip(cluster.response_blocks(local),v['amplitude'].reshape(cluster.components,k)):
                raw.append(b.reduced(d*(b.amplitude_basis@a)[:,None,None])[b.selected])
            derivative=np.concatenate(raw,axis=1)@transform[index]
            expected+=2*np.sum(abs(derivative)**2,axis=0)
        np.testing.assert_allclose(norm**2,expected,rtol=2e-14)
        answer=paths(model,p,components=(1,2),transform=transform)
        covariance=np.linalg.inv(actual.T@actual);t=answer['time_s'];functionals=[]
        for s in (1,2):
            g=np.zeros((len(t),model.count));g[:,model.local_path_indices[0][s]]=model.clusters[0].arcs[s].base.curve(t)
            functionals.append(g@transform)
        functionals.append(functionals[1]-functionals[0]);g=np.concatenate(functionals)
        expected=np.einsum('ni,ij,nj->n',g,covariance,g).reshape(3,len(t))
        np.testing.assert_allclose(answer['standard_deviation_by_functional_m']**2,expected,rtol=3e-7,atol=1e-15)

    def test_shared_band_assembly_matches_full_gaussian_linear_nuisance_elimination(self):
        from test_flexible_echo_cluster import cubic_scene
        model,p=cubic_scene();actual,norm=expected_jacobian(model,p)
        information=np.zeros((model.count,model.count));original_norm2=np.zeros(model.count)
        for cluster,index in zip(model.clusters,model.indices):
            b=cluster.base;local=p[index];v=cluster.evaluate(local);k=len(b.knots)
            raw=[]
            for (_,d),a in zip(cluster.response_blocks(local),v['amplitude'].reshape(cluster.components,k)):
                raw.append(b.reduced(d*(b.amplitude_basis@a)[:,None,None])[b.selected])
            raw=np.concatenate(raw,axis=1);m=np.r_[raw.real,raw.imag]
            brightness=v['design'][b.selected][:,v['amplitude']>0];static=b.design[b.selected]
            nuisance=np.c_[np.r_[brightness.real,brightness.imag],
                np.r_[static.real,static.imag],np.r_[-static.imag,static.real]]
            projected=m-nuisance@np.linalg.lstsq(nuisance,m,rcond=None)[0]
            information[np.ix_(index,index)]+=2*projected.T@projected
            original_norm2[index]+=2*np.sum(abs(raw)**2,axis=0)
        np.testing.assert_allclose(actual.T@actual,information,rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(norm**2,original_norm2,rtol=1e-14)

    def test_bounds_equal_full_joint_gaussian_covariance_without_prior_rows(self):
        rng=np.random.default_rng(814);matrix=rng.normal(size=(103,9))*np.logspace(-2,2,9)
        g=rng.normal(size=(13,4));interest=np.array([0,2,5,7])
        answer=functional_bounds(matrix,np.linalg.norm(matrix,axis=0),interest,g)
        covariance=np.linalg.solve(matrix.T@matrix,np.eye(9))[np.ix_(interest,interest)]
        expected=np.einsum('ni,ij,nj->n',g,covariance,g)
        self.assertEqual(answer['target_rank'],4)
        np.testing.assert_allclose(answer['standard_deviation']**2,expected,rtol=2e-12)

    def test_complementary_observations_restore_missing_phase_information(self):
        first=np.array([[1.,0.,0.],[0.,0.,1.],[2.,0.,.4]])
        g=np.array([[1.,0.],[0.,1.],[1.,-1.],[0.,0.]])
        one=functional_bounds(first,np.array([3.,2.,2.]),[0,1],g)
        np.testing.assert_array_equal(one['estimable'],[True,False,False,True])
        joined=np.r_[first,[[0.,1.,.2],[0.,2.,-.3]]]
        all_bands=functional_bounds(joined,np.array([3.,3.,3.]),[0,1],g)
        self.assertTrue(all_bands['estimable'].all());self.assertEqual(all_bands['target_rank'],2)
        covariance=np.linalg.inv(joined.T@joined)[:2,:2]
        np.testing.assert_allclose(all_bands['standard_deviation']**2,np.einsum('ni,ij,nj->n',g,covariance,g),atol=1e-14)

    def test_roundoff_inside_an_eliminated_direction_is_not_measured_information(self):
        rng=np.random.default_rng(15);other=rng.normal(size=51);target=other+1e-14*rng.normal(size=51)
        matrix=np.c_[target,other]
        bounds=functional_bounds(matrix,np.linalg.norm(matrix,axis=0),[0],[[1.],[0.]])
        self.assertEqual(bounds['target_rank'],0)
        self.assertFalse(bounds['estimable'][0]);self.assertTrue(np.isnan(bounds['standard_deviation'][0]))
        self.assertEqual(bounds['standard_deviation'][1],0.)


if __name__=='__main__':unittest.main()
