"""Local Cartesian AIS reference which preserves interpolation knots.

This is a numerical representation of the existing AIS prior, not a fitted
radar trajectory or a claim that AIS interpolated positions are exact.
Each original Hermite interval has its own short Cartesian polynomial.
"""
import numpy as np

from motion import ecef


class LocalAisReference:
    def __init__(self,track,utc,rotation,receiver,*,half_width_s=4.):
        epoch=utc-track.epoch
        if epoch-half_width_s<track.curve.x[0] or epoch+half_width_s>track.curve.x[-1]:
            raise ValueError('AIS extrapolation is not permitted')
        interior=track.curve.x[(track.curve.x>epoch-half_width_s)&(track.curve.x<epoch+half_width_s)]-epoch
        self.edges=np.r_[-half_width_s,interior,half_width_s]
        self.centres=(self.edges[1:]+self.edges[:-1])/2
        self.scales=np.diff(self.edges)/2
        self.coefficients=[];self.maximum_error_m=0.
        def position(offset):
            lat,lon=track.curve(epoch+offset).T
            return (ecef(lat,lon,track.reference_height_ellipsoid_m)-receiver)@rotation.T
        for centre,scale in zip(self.centres,self.scales):
            grid=np.linspace(-1.,1.,33)
            coefficient=np.polynomial.polynomial.polyfit(grid,position(centre+scale*grid),5)
            probe=np.linspace(-.997,.997,101)
            error=np.linalg.norm(np.polynomial.polynomial.polyval(probe,coefficient).T-position(centre+scale*probe),axis=1).max()
            self.maximum_error_m=max(self.maximum_error_m,float(error))
            if error>1e-5:raise ValueError('Piecewise Cartesian AIS representation exceeds 10 micrometres')
            self.coefficients.append(coefficient)

    def __call__(self,offset,derivative=0):
        value=np.asarray(offset)
        if value.ndim or not np.isfinite(value) or not 0<=derivative<=3:
            raise ValueError('A finite scalar time and derivative order 0 to 3 are required')
        real=float(value.real)
        if real<self.edges[0] or real>self.edges[-1]:raise ValueError('Outside the local AIS reference')
        index=min(np.searchsorted(self.edges,real,side='right')-1,len(self.scales)-1)
        coefficient=np.polynomial.polynomial.polyder(self.coefficients[index],derivative)
        return np.polynomial.polynomial.polyval((value-self.centres[index])/self.scales[index],coefficient)/self.scales[index]**derivative

    def sample(self,offset,derivative=0):
        """Vectorized evaluation, retaining the same piecewise knot convention."""
        value=np.asarray(offset)
        if not np.isfinite(value).all() or not 0<=derivative<=3:
            raise ValueError('Finite times and derivative order 0 to 3 are required')
        flat=value.ravel();real=flat.real
        if np.any((real<self.edges[0])|(real>self.edges[-1])):
            raise ValueError('Outside the local AIS reference')
        indices=np.minimum(np.searchsorted(self.edges,real,side='right')-1,len(self.scales)-1)
        result=np.empty((len(flat),3),dtype=np.result_type(value,float))
        for index in np.unique(indices):
            use=indices==index
            coefficient=np.polynomial.polynomial.polyder(self.coefficients[index],derivative)
            u=(flat[use]-self.centres[index])/self.scales[index]
            result[use]=np.polynomial.polynomial.polyval(u,coefficient).T/self.scales[index]**derivative
        return result.reshape(value.shape+(3,))
