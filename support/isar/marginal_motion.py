"""Physical motion with independently marginalised complex band reflectivities.

The observation operator and stationary nuisance are unchanged. A scalar
Gaussian scene variance is fitted to each training band at every motion trial;
posterior mean coefficients predict withheld observations without refitting.
The conditional covariance Fisher matrix scales steps; it is not a measured
CRLB for the real ship or an assertion that the noise/scene model is correct.
"""
import numpy as np

from gaussian_scene import marginal, motion_information
from physical_profile import PhysicalProfile, observe_partials
from validate_motion import validate


class MarginalMotion(PhysicalProfile):
    def responses(self, parameters, *, derivatives=False):
        motion = self.expand(parameters)
        outputs = []
        for band in self.bands:
            target, partials = band.design(motion, derivatives=derivatives)
            design = band.nuisance.remove(target[band.selected])
            result = marginal(design, band.clean)
            result['target'] = target
            if derivatives:
                gradient = np.zeros(len(motion))
                observed = []
                for j, raw in enumerate(partials):
                    local = observe_partials(band, raw, False)
                    gradient += 2*np.real(result['design_gradient'][:, j].conj()@local)
                    observed.append(local[:, self.active])
                result['gradient'] = gradient[self.active]
                result['preconditioner'] = motion_information(result, np.stack(observed, axis=1))
            outputs.append(result)
        return outputs

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(self.last['parameters'], parameters):
            return self.last
        outputs = self.responses(parameters, derivatives=True)
        self.evaluations += 1
        self.last = {'parameters': np.asarray(parameters).copy(),
            'value': sum(o['value'] for o in outputs)/self.energy,
            'gradient': sum(o['gradient'] for o in outputs)/self.energy,
            'gauss_newton': sum(o['preconditioner'] for o in outputs)/self.energy,
            'scene_conditions': [o['condition'] for o in outputs],
            'scene_variances': [o['variance'] for o in outputs],
            'effective_scene_ranks': [o['effective_scene_rank'] for o in outputs],
            'variance_fits': [o['variance_fit'] for o in outputs], 'evaluations': self.evaluations}
        return self.last

    def validation(self, parameters):
        return [validate(band.field, result['target'], band.nuisance, band.selected_time,
            band.model['interval_s'], coefficients=result['coefficient'], stationary_mode='project')
            for band, result in zip(self.bands, self.responses(parameters))]
