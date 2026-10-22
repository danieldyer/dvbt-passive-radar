"""Invertible optimizer coordinates for source phase/position correlations.

The physical parameter vector is x = T q. Approximate carrier-phase pivots
reduce correlation between source positions and their unknown constant phases.
This is only a coordinate change: the likelihood, bounds, physical model and
saved parameters retain their original definitions.
"""
import numpy as np

from bistatic_path_lift import lift
from motion import excess_path


def transform(likelihood, parameters):
    n, count = likelihood.n, likelihood.p
    end=n+5+count-1
    band = likelihood.bands[3]
    b = band.point
    points = getattr(likelihood,'reference_points',band.points)
    if points.ndim==3:
        windows=np.repeat(np.arange(len(points)),band.active_points.sum(axis=1))
        points=points[band.active_points]
    else:points=points.reshape(count,3);windows=np.zeros(count,int)
    pivot = np.zeros((count, parameters))
    for p, (point, window) in enumerate(zip(points, windows)):
        selected = np.flatnonzero(band.window_index == window)
        row = selected[np.argmin(abs(b.t[selected]-(b.t[selected[0]]+b.t[selected[-1]])/2))]
        sl = slice(row, row+1)
        s = lift(b.anchor[sl], b.velocity[sl], b.direction[sl], b.direction_rate[sl],
                 b.base_path[sl], b.base_rate[sl], b.motion.tx[3], b.motion.rx)
        pos = s['position_ecef_m'] + b.frame[sl]@(point-b.reference_point)
        _, gradient = excess_path(pos, b.motion.tx[3], b.motion.rx)
        path_derivative = float(np.sum(gradient*s['position_path_jacobian']))
        pivot[p, :n] = b.field.k*path_derivative*b.b[row:row+1].toarray()[0]
        if parameters>end:
            if parameters!=end+2*count:raise ValueError('Unexpected physical source-coordinate layout')
            pivot[p, end+2*p:end+2*p+2] = b.field.k*(gradient[0]@b.frame[row, :, :2])
    result = np.eye(parameters)
    result[n:n+5] += pivot[0]
    result[n+5:end] += pivot[1:]-pivot[0]
    # Phase columns are unchanged; only physical path and source-coordinate
    # columns feed the phase rows. This is triangular with unit diagonal.
    return result
