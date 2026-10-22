"""Geometric bounds for identifying the same two points across transmitters."""
import numpy as np


def relative_path(point,reference,transmitters,receiver):
    p,q,t,r=map(np.asarray,(point,reference,transmitters,receiver))
    return np.linalg.norm(p-t,axis=1)-np.linalg.norm(q-t,axis=1)+np.linalg.norm(p-r)-np.linalg.norm(q-r)


def relative_delay_bound(centre,sites,*,scene_radius_m,point_separation_m,phase_centre_radius_m=0.):
    """Upper bound on any two transmitters' inter-point range difference.

    Integrate the Hessian of distance over the point and transmitter segments.
    Its spectral norm is 1/r, so |double difference| <= L*B/r_min.
    Points lie in a convex ball about centre; each transmitter may lie in a
    phase_centre_radius_m ball about one site. A common receiver cancels exactly.
    """
    c,t=np.asarray(centre,float),np.asarray(sites,float)
    if (c.shape!=(3,) or t.ndim!=2 or t.shape[1]!=3 or not len(t) or
        scene_radius_m<0 or point_separation_m<0 or phase_centre_radius_m<0):
        raise ValueError('Finite 3D geometry and nonnegative radii are required')
    if not np.isfinite(c).all() or not np.isfinite(t).all():raise ValueError('Nonfinite geometry')
    minimum=np.inf;baseline=0.
    for a in t:
        for b in t:
            d=b-a;norm=d@d;u=np.clip((c-a)@d/norm,0,1) if norm>0 else 0.
            minimum=min(minimum,np.linalg.norm(c-a-u*d));baseline=max(baseline,np.sqrt(norm))
    minimum-=scene_radius_m+phase_centre_radius_m
    if minimum<=0:raise ValueError('Scene/transmitter uncertainty intersects; far-field bound unavailable')
    baseline+=2*phase_centre_radius_m
    return dict(maximum_range_difference_m=float(point_separation_m*baseline/minimum),
        maximum_transmitter_baseline_m=float(baseline),minimum_propagation_distance_m=float(minimum),
        scene_radius_m=float(scene_radius_m),point_separation_m=float(point_separation_m),
        phase_centre_radius_m=float(phase_centre_radius_m))


def relative_rate_bound(centre,sites,*,scene_radius_m,maximum_point_speed_mps,phase_centre_radius_m=0.):
    """Bound interband path-rate differences for one physical moving point.

    The receiver leg cancels. Integrating the unit-direction derivative over
    the transmitter segment gives |Ldot_a-Ldot_b| <= |v| B/r_min.
    """
    if not np.isfinite(maximum_point_speed_mps) or maximum_point_speed_mps<0:
        raise ValueError('A finite nonnegative speed bound is required')
    bound=relative_delay_bound(centre,sites,scene_radius_m=scene_radius_m,
        point_separation_m=1.,phase_centre_radius_m=phase_centre_radius_m)
    return dict(maximum_rate_difference_mps=maximum_point_speed_mps*bound['maximum_range_difference_m'],
        maximum_point_speed_mps=maximum_point_speed_mps,
        maximum_transmitter_baseline_m=bound['maximum_transmitter_baseline_m'],
        minimum_propagation_distance_m=bound['minimum_propagation_distance_m'],
        scene_radius_m=scene_radius_m,phase_centre_radius_m=phase_centre_radius_m)


def correlated_pair_bounds(centre,transmitters,ecef_to_enu,offset_factor,*,scene_radius_m,
                           maximum_speed_mps,maximum_acceleration_mps2,parameter_radius=3.):
    """Necessary receiver-independent bounds for the same moving point.

    ENU transmitter offsets are F q with |q_j| <= parameter_radius. Using
    F_a-F_b preserves shared-site correlations instead of adding two unrelated
    survey balls. The point stays in the declared scene ball throughout.
    For D=|x-T_a|-|x-T_b|, |D'| <= v B/r and
    |D''| <= a B/r + 3 v² B/r². The second inequality follows by integrating
    the distance Hessian along the transmitter segment; its spatial derivative
    has operator norm at most 3/r². The common receiver term cancels exactly.
    These are conservative feasibility bounds, not statistical error bars.
    """
    c,t,r,f=map(lambda a:np.asarray(a,float),(centre,transmitters,ecef_to_enu,offset_factor))
    limits=np.asarray([scene_radius_m,maximum_speed_mps,maximum_acceleration_mps2,parameter_radius])
    if (c.shape!=(3,) or t.ndim!=2 or t.shape[1]!=3 or r.shape!=(3,3) or
            f.ndim!=2 or f.shape[0]!=3*len(t) or not np.isfinite(limits).all() or
            np.any(limits<0) or any(not np.isfinite(a).all() for a in (c,t,r,f))):
        raise ValueError('Finite geometry, matching ENU factor and nonnegative bounds required')
    np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-10,rtol=0)
    factors=f.reshape(len(t),3,-1)
    radii=parameter_radius*np.linalg.norm(np.sum(abs(factors),axis=2),axis=1)
    shape=(len(t),len(t));baseline=np.zeros(shape);distance=np.zeros(shape)
    for i,a in enumerate(t):
        for j,b in enumerate(t):
            d=b-a;norm=d@d;u=np.clip((c-a)@d/norm,0,1) if norm>0 else 0.
            distance[i,j]=np.linalg.norm(c-a-u*d)-scene_radius_m-max(radii[i],radii[j])
            delta=(a-b)@r.T
            baseline[i,j]=np.linalg.norm(abs(delta)+parameter_radius*np.sum(abs(factors[i]-factors[j]),axis=1))
    if np.any(distance<=0):raise ValueError('The allowed transmitter segment intersects the scene')
    rate=maximum_speed_mps*baseline/distance
    acceleration=maximum_acceleration_mps2*baseline/distance+3*maximum_speed_mps**2*baseline/distance**2
    return dict(maximum_transmitter_baseline_m=baseline,minimum_propagation_distance_m=distance,
        maximum_rate_difference_mps=rate,maximum_acceleration_difference_mps2=acceleration)
