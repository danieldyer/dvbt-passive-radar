"""Fixed stationary nuisance with explicit moving envelope-origin derivatives.

Static columns use observed absolute delay coordinates. Their unknown physical
locations are not used as calibration anchors. Only the moving road response
contains the shared hardware-origin parameter.
"""
from types import SimpleNamespace

from integrated_stationary_road import IntegratedField


class ProfiledRoadField(IntegratedField):
    def __init__(self, original, delays, binned, cutoff=1e-10):
        super().__init__(SimpleNamespace(original=original, time=original.time,
            delays=delays, cutoff=cutoff), binned)

    def template(self, path, jacobian=None, range_bias=0., bias_derivative=False):
        value, derivative = self.original.template(path, jacobian,
            range_bias=range_bias, bias_derivative=bias_derivative)
        return self.remove(value), None if derivative is None else self.remove(derivative)
