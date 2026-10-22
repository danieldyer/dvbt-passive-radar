"""Recorded-kernel cross-check of factored joint antenna derivatives."""
import argparse,json,time
from pathlib import Path
import numpy as np
from antenna_motion import Antennas
from continuous_ship_motion import ContinuousShipMotion
from full_path_point import carrier_bin
from full_ship_field import load
from joint_continuous_ship import JointCoordinates,JointPointBand
from bistatic_path_lift import curvature


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['motion','field','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();motion=ContinuousShipMotion(a.motion);fields,coordinates,_=load(a.field)
    full=fields[4];field=full.windowed(142.,172.);indices=np.searchsorted(full.time,field.time)
    coordinate={key:coordinates[4][key].reshape(-1,16)[indices].ravel() for key in ['native_time_s','native_information']}
    coordinate['training']=np.ones(len(field.time),bool)
    model=JointCoordinates(motion,Antennas(motion.rotation,['o','o','o','o','r']));b=JointPointBand(field,coordinate,model,4)
    q=np.zeros(model.n);q[model.nmotion]=.02;q[model.location_start:]=np.linspace(-.02,.03,model.antennas.count)
    point=np.array([25.,-40.,15.]);started=time.monotonic();value=b.waveform(q,4,True,point);factored_seconds=time.monotonic()-started
    position,velocity,offset,dp,dv=b.point_state(q,point);antenna=model.antennas.offsets(q[model.location_start:],4)
    tx,rx=motion.tx[4]+antenna[0],motion.rx+antenna[1];kw=dict(transmitter=tx,receiver=rx,position_offset=offset)
    started=time.monotonic();raw,jac=field.raw_trajectory(position,velocity,position_jacobian=dp,velocity_jacobian=dv,antenna_derivatives=True,**kw)
    _,_,g=field.path_kinematics(position,velocity,**kw)
    factor,dfactor=carrier_bin(b.t,(g*velocity).sum(axis=1),coordinate['native_time_s'],coordinate['native_information'],field.k)
    delta,dref=model.reference_phase(q,4,True);phase=np.exp(1j*field.k*delta);speed=[]
    for j in range(7):speed.append(np.sum(g*dv[:,:,j]+curvature(position,dp[:,:,j],tx,rx)*velocity,axis=1))
    for antenna in [tx,rx]:
        ray=position-antenna;distance=np.linalg.norm(ray,axis=1);unit=ray/distance[:,None]
        speed.extend((-(velocity-unit*np.sum(unit*velocity,axis=1)[:,None])/distance[:,None]).T)
    reference=[field.apply(raw*factor[:,None])*phase]
    for j,ds in enumerate(speed):
        derivative=field.apply(jac[:,:,j]*factor[:,None]+raw*(dfactor*ds)[:,None])*phase
        if j>=7:derivative+=reference[0]*(1j*field.k*dref[j-7])
        reference.append(derivative)
    unfactored_seconds=time.monotonic()-started
    errors=[float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30)) for x,y in zip(value,reference)]
    if max(errors)>1e-9:raise AssertionError('Factoring range whitening changes the physical waveform derivative')
    rng=np.random.default_rng(37);checks=[]
    for _ in range(5):
        direction=rng.normal(size=len(q));direction/=np.linalg.norm(direction);step=2e-5
        numeric=(b.waveform(q+step*direction,4,False,point)-b.waveform(q-step*direction,4,False,point))/(2*step)
        expected=sum(v*(basis@direction)[:,None] for v,basis in zip(value[1:],b.motion_bases))
        error=float(np.linalg.norm(numeric-expected)/np.linalg.norm(expected));checks.append(error)
        if error>2e-4:raise AssertionError('Recorded-kernel joint derivative differs from finite differences')
    report=dict(complete=True,channel='ABC',interval_s=[142.,172.],native_16_symbol_bins=len(field.time),
        relative_errors_against_unfactored_original_operator=errors,finite_difference_direction_errors=checks,
        factored_seconds=factored_seconds,unfactored_seconds=unfactored_seconds,scope=__doc__)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
