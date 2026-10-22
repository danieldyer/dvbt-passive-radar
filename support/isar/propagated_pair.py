"""Substitute a declared physical sea response for the primary echo only."""
from physical_pair import SharedPhysicalPair
from sea_physical_arc import SeaPhysicalArc


def primary_propagation(independent,water,family):
    """Keep data, secondary echoes, priors, masks and search bounds identical.

    Reflected primary rays have spherical specular paths, spreading factors and
    HH Fresnel coefficients. They share the primary's physical point and real
    brightness; no independent reflected-path phase or strength is fitted.
    The water geometry and dielectric assumptions remain declared inputs.
    """
    bands=[SeaPhysicalArc(b.field,b.training,b.knots,b.geometry,b.channel,b.midpoint,
                          water=water,family=family,stationary_paths=b.delays) for b in independent.bands]
    result=SharedPhysicalPair(bands)
    result.lower=independent.lower.copy();result.upper=independent.upper.copy()
    return result
