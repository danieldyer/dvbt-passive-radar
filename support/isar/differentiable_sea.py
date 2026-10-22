"""Analytic endpoint derivatives of the existing spherical four-ray reference.

The specular point moves according to the two-dimensional Fermat Hessian on
the fixed sphere. Differentiating that constraint preserves reflected path,
path rate, divergence and HH Fresnel phase without finite-difference fitting.
The water surface and dielectric parameters are declared, not inferred here.
"""
import numpy as np

from spherical_sea import reflected_leg,horizontal_reflection


def dot(vector,jacobian):
    return np.einsum('ni,nip->np',vector,jacobian)


def direct(position,velocity,antenna,dx,dv,da):
    vector=position-antenna
    length=np.linalg.norm(vector,axis=1);unit=vector/length[:,None]
    change=dx-da
    derivative=dot(unit,change)
    dunit=(change-unit[:,:,None]*derivative[:,None,:])/length[:,None,None]
    rate=np.sum(unit*velocity,axis=1)
    drate=dot(velocity,dunit)+dot(unit,dv)
    return length,derivative,rate,drate


def reflected(position,velocity,antenna,centre,radius,dx,dv,da,frequency_hz,
              *,relative_permittivity=80.,conductivity_s_per_m=4.):
    ray=reflected_leg(position,velocity,antenna,centre,radius)
    point=ray['specular_ecef_m'];normal=(point-centre)/radius
    la,lb=ray['antenna_segment_m'],ray['target_segment_m']
    ua=(antenna-point)/la[:,None];ub=(position-point)/lb[:,None]
    projector=lambda u,d:(np.eye(3)[None]-u[:,:,None]*u[:,None,:])/d[:,None,None]
    pa,pb=projector(ua,la),projector(ub,lb)
    tangent=ua-np.sum(ua*normal,axis=1)[:,None]*normal
    tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    basis=np.stack([tangent,np.cross(normal,tangent)],axis=2)
    hessian=np.einsum('nia,nij,njb->nab',basis,pa+pb,basis)
    hessian+=np.eye(2)[None]*(np.sum((ua+ub)*normal,axis=1)/radius)[:,None,None]
    rhs=np.einsum('nij,njp->nip',pb,dx)+np.einsum('nij,jp->nip',pa,da)
    local=np.linalg.solve(hessian,np.einsum('nia,nip->nap',basis,rhs))
    ds=np.einsum('nia,nap->nip',basis,local)
    dua=np.einsum('nij,njp->nip',pa,da[None]-ds)
    dub=np.einsum('nij,njp->nip',pb,dx-ds)
    dla=dot(ua,da[None]-ds);dlb=dot(ub,dx-ds)
    # Fermat stationarity cancels the tangential specular-point term in dL.
    dlength=dot(ua,np.broadcast_to(da,dx.shape))+dot(ub,dx)
    drate=dot(velocity,dub)+dot(ub,dv)
    sine=ray['sin_grazing']
    dsine=.5*(dot(normal,dua+dub)+dot(ua+ub,ds/radius))
    distance=la*lb/(la+lb)
    ddistance=(lb[:,None]**2*dla+la[:,None]**2*dlb)/(la+lb)[:,None]**2
    u=2*distance/(radius*sine);v=2*distance*sine/radius
    du=2/radius*(ddistance/sine[:,None]-distance[:,None]*dsine/sine[:,None]**2)
    dvfactor=2/radius*(ddistance*sine[:,None]+distance[:,None]*dsine)
    divergence=ray['divergence']
    ddivergence=-.5*divergence[:,None]*(du/(1+u[:,None])+dvfactor/(1+v[:,None]))
    gamma=horizontal_reflection(sine,frequency_hz,relative_permittivity=relative_permittivity,
                               conductivity_s_per_m=conductivity_s_per_m)
    epsilon=relative_permittivity-1j*conductivity_s_per_m/(2*np.pi*frequency_hz*8.8541878128e-12)
    root=np.sqrt(epsilon-1+sine*sine)
    dgamma=(2*(epsilon-1)/(root*(sine+root)**2))[:,None]*dsine
    gain=divergence*gamma
    dgain=ddivergence*gamma[:,None]+divergence[:,None]*dgamma
    return ray['length_m'],dlength,ray['rate_mps'],drate,gain,dgain


def paths(position,velocity,transmitter,receiver,centre,radius,dx,dv,dtx,drx,frequency_hz):
    """Four excess paths, rates and relative fields, plus all parameter Jacobians."""
    antennas=[transmitter,receiver];derivatives=[dtx,drx]
    straight=[direct(position,velocity,a,dx,dv,j) for a,j in zip(antennas,derivatives)]
    reflected_rays=[reflected(position,velocity,a,centre,radius,dx,dv,j,frequency_hz)
                    for a,j in zip(antennas,derivatives)]
    baseline=np.linalg.norm(transmitter-receiver)
    dbaseline=((transmitter-receiver)/baseline)@(dtx-drx)
    length=[];rate=[];gain=[];dlength=[];drate=[];dgain=[]
    for flags in ((False,False),(True,False),(False,True),(True,True)):
        l=np.full(len(position),-baseline);dl=np.broadcast_to(-dbaseline,(len(position),dx.shape[2])).copy()
        v=np.zeros(len(position));dj=np.zeros_like(dl)
        a=np.ones(len(position),complex);da=np.zeros_like(dl,dtype=complex)
        for flag,ref,direct_ray in zip(flags,reflected_rays,straight):
            selected=ref if flag else direct_ray
            l+=selected[0];dl+=selected[1];v+=selected[2];dj+=selected[3]
            if flag:
                ratio=direct_ray[0]/ref[0]
                dratio=direct_ray[1]/ref[0][:,None]-direct_ray[0][:,None]*ref[1]/ref[0][:,None]**2
                weight=ratio*ref[4];dweight=dratio*ref[4][:,None]+ratio[:,None]*ref[5]
                da=da*weight[:,None]+a[:,None]*dweight;a*=weight
        length.append(l);dlength.append(dl);rate.append(v);drate.append(dj);gain.append(a);dgain.append(da)
    return tuple(np.array(x) for x in (length,rate,gain,dlength,drate,dgain))
