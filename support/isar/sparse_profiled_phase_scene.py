"""Common-path acquisition with a physical-pixel complex Laplace prior.

Unlike an L2 scene prior, sparsity can distinguish phase errors that merely
spread energy through an otherwise unrestricted Fourier image. Image
positions are not supplied. The modal operator is numerical compression;
the sparse prior remains on the full physical image coordinates.
"""
import numpy as np
from profiled_phase_scene import ProfiledPhaseScene
from complex_lasso import LowRankComplexLasso


class SparseSceneSolver:
    def __init__(self,spatial_basis,strength,discarded_operator_eigenvalue=0.,kkt_tolerance=1e-8,gap_tolerance=1e-8):
        self.spatial_basis=np.asarray(spatial_basis);self.discarded_operator_eigenvalue=discarded_operator_eigenvalue
        self.strength=strength;self.kkt_tolerance=kkt_tolerance;self.gap_tolerance=gap_tolerance;self.warm=None

    def __call__(self,gram,rhs,energy,spectrum):
        solver=LowRankComplexLasso(self.spatial_basis,gram,rhs,energy)
        physical,report=solver.solve(self.strength,self.warm,kkt_tolerance=self.kkt_tolerance,gap_tolerance=self.gap_tolerance,maximum_iterations=30000)
        if not report['numerical_convergence']:
            raise ValueError(f"Profiled sparse image missed its numerical gate: {report['maximum_scaled_kkt']}, {report['gap_relative_to_fitted_energy']}")
        self.warm=physical;coefficient=self.spatial_basis.conj().T@physical
        penalty=float(2*self.strength*np.sum(abs(physical)))
        discarded_bound=float(np.sqrt(max(0.,self.discarded_operator_eigenvalue)*max(abs(spectrum)**2))*np.linalg.norm(physical))
        information=float(np.vdot(coefficient,gram@coefficient).real)
        return coefficient,penalty,dict(physical_coefficient=physical,sparse_optimization=report,
            omitted_prediction_norm_bound=discarded_bound,
            omitted_prediction_relative_norm_bound=discarded_bound/np.sqrt(information) if information>0 else None)


class SparseProfiledPhaseScene(ProfiledPhaseScene):
    def __init__(self,*args,spatial_basis,discarded_operator_eigenvalue=0.,kkt_tolerance=1e-8,gap_tolerance=1e-8,**kwargs):
        super().__init__(*args,**kwargs)
        self.image_solver=SparseSceneSolver(spatial_basis,self.strength,discarded_operator_eigenvalue,kkt_tolerance,gap_tolerance)
