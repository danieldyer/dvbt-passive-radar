"""Exact bistatic coordinates: excess length plus receive bearing.

This lets a continuous radial history follow observed manoeuvres without using
large Cartesian accelerations to manufacture high-order range curvature.
Range, carrier phase, velocity and Doppler still come from the same 3-D curve.
"""
import numpy as np
from numpy.polynomial import legendre as leg
from scipy.interpolate import BSpline

from geometry import SITES, unit


class PolarTrajectory:
    def __init__(self, seed, reference_channel=1, angular_degree=2, knot_interval=1.5,
                 channel_sites=None, channel_offsets=None):
        self.seed = seed
        self.channel_sites = np.array(SITES if channel_sites is None else channel_sites)
        self.channel_offsets = np.zeros((5,3)) if channel_offsets is None else np.asarray(channel_offsets)
        self.reference_site = self.channel_sites[reference_channel]
        self.reference_offset = self.channel_offsets[reference_channel]
        self.angular_degree = angular_degree
        self.half = seed.get('duration_s',30.)/2
        knots = np.r_[np.full(4,-self.half),np.arange(-self.half+knot_interval,self.half,knot_interval),np.full(4,self.half)]
        self.radial_count = len(knots)-4
        self.spline = BSpline(knots,np.eye(self.radial_count),3,extrapolate=True)
        self.count = self.radial_count+2*(angular_degree+1)

    def length_basis(self, time):
        t = np.asarray(time)-self.seed['epoch']
        base = np.polynomial.polynomial.polyval(t/self.seed.get('scale',15.),self.seed['path'])
        return base,self.spline(t)

    def positions(self, p, time, sites, free_sites=False):
        t = (np.asarray(time)-self.seed['epoch'])/self.half
        base,b = self.length_basis(time)
        length = base+b@p[:self.radial_count]
        angular = leg.legvander(t,self.angular_degree)
        m = self.angular_degree+1
        theta = angular@p[self.radial_count:self.radial_count+m]
        elevation = angular@p[self.radial_count+m:self.count]
        direction = np.c_[np.cos(elevation)*np.cos(theta),np.cos(elevation)*np.sin(theta),np.sin(elevation)]
        dtheta = np.c_[-np.cos(elevation)*np.sin(theta),np.cos(elevation)*np.cos(theta),np.zeros(len(t))]
        delevation = np.c_[-np.sin(elevation)*np.cos(theta),-np.sin(elevation)*np.sin(theta),np.cos(elevation)]
        tx = sites.copy()
        if free_sites:
            tx += p[self.count:].reshape(3,3)*[30,30,75]
        ref = tx[self.reference_site]+self.reference_offset; baseline = np.linalg.norm(ref)
        s = length+baseline
        denominator = s-direction@ref
        radius = length*(length+2*baseline)/(2*denominator)
        x = radius[:,None]*direction
        j = np.zeros((len(t),3,len(p)))
        drdl = (s-radius)/denominator
        j[:,:,:self.radial_count] = direction[:,:,None]*drdl[:,None,None]*b[:,None,:]
        for derivative,offset in [(dtheta,self.radial_count),(delevation,self.radial_count+m)]:
            dr = radius*(derivative@ref)/denominator
            dx = radius[:,None]*derivative+dr[:,None]*direction
            j[:,:,offset:offset+m] = dx[:,:,None]*angular[:,None,:]
        if free_sites:
            drdt = ((length-radius)[:,None]*unit(ref)+radius[:,None]*direction)/denominator[:,None]
            first = self.count+3*self.reference_site
            j[:,:,first:first+3] = direction[:,:,None]*drdt[:,None,:]*np.array([30,30,75])
        return x,j,tx

    def paths(self, p, time, channel, sites, free_sites=False):
        x,jx,tx = self.positions(p,time,sites,free_sites)
        transmitter = tx[self.channel_sites[channel]]+self.channel_offsets[channel]
        a,b = x-transmitter,x
        path = np.linalg.norm(a,axis=1)+np.linalg.norm(b,axis=1)-np.linalg.norm(transmitter,axis=1)
        j = np.einsum('ni,nij->nj',unit(a)+unit(b),jx)
        if free_sites:
            direct = np.zeros((len(time),3,3))
            direct[np.arange(len(time)),self.channel_sites[channel]] = (-unit(a)-unit(transmitter))*[30,30,75]
            j[:,self.count:] += direct.reshape(len(time),9)
        return path,j

    def kinematics(self, p, sites, free_sites=False):
        # The same deterministic differentiation is used for values and their
        # analytic parameter Jacobians. End effects are omitted from the penalty.
        time = self.seed['epoch']+np.linspace(-self.half,self.half,121)
        step = 2*self.half/120
        x,j,_ = self.positions(p,time,sites,free_sites)
        values,derivatives = [],[]
        for _ in range(3):
            x = np.gradient(x,step,axis=0,edge_order=2)
            j = np.gradient(j,step,axis=0,edge_order=2)
            values.append(x[4:-4]);derivatives.append(j[4:-4])
        return values,derivatives

    def bounds(self, free_sites=False):
        lo,hi = np.full(self.count,-1.),np.full(self.count,1.)
        lo[:self.radial_count],hi[:self.radial_count] = -100.,100.
        lo[self.radial_count],hi[self.radial_count] = -2*np.pi,2*np.pi
        lo[self.radial_count+self.angular_degree+1],hi[self.radial_count+self.angular_degree+1] = -.15,1.3
        if free_sites:
            lo,hi = np.r_[lo,np.full(9,-3.)],np.r_[hi,np.full(9,3.)]
        return lo,hi
