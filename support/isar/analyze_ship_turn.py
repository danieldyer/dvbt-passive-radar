"""Quantify the heading hypothesis suggested, but not proved, by AIS course."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv
from flash_siting_report import EPOCH
from motion import path_difference
from report_measured_scatterer_phases import serial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/data/isar_20261018/turn-control'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    time=np.linspace(142.,172.,601);utc=EPOCH+time;mid=157.
    course=np.interp(utc,track.seconds,np.unwrap(track.course));reference=np.interp(EPOCH+mid,track.seconds,np.unwrap(track.course))
    index=np.searchsorted(track.seconds,EPOCH+mid)-1
    rate=np.degrees(np.unwrap(track.course)[index+1]-np.unwrap(track.course)[index])/(track.seconds[index+1]-track.seconds[index])
    data=track.sample(utc,max_gap_s=400.)
    geo=json.loads(Path('docs/data/injection_20261018/separated-500m/plan.json').read_text())
    rx=np.asarray(geo['receiver_ecef_m']);tx=np.asarray(geo['transmitters_ecef_m']);rotation=np.asarray(geo['ecef_to_enu'])
    velocity=data['velocity_ecef_mps']@rotation.T
    derived=np.unwrap(np.arctan2(velocity[:,0],velocity[:,1]));derived-=2*np.pi*np.round((derived[len(time)//2]-reference)/(2*np.pi))
    arm=150*np.array([np.sin(reference),np.cos(reference),0.])
    angle=-(course-reference);c,s=np.cos(angle),np.sin(angle)
    rotated=np.c_[c*arm[0]-s*arm[1],s*arm[0]+c*arm[1],np.zeros(len(time))]
    fixed=data['position_ecef_m']+arm@rotation;offset=(rotated-arm)@rotation
    differences=np.array([path_difference(fixed,offset,t,rx) for t in tx])
    loss=[-10*np.log10(abs(np.mean(np.exp(-2j*np.pi*f/299792458.*d)))**2) for f,d in zip(geo['rf_hz'],differences)]
    result=dict(scope=__doc__,interval_s=[142,172],midpoint_s=mid,
        reported_true_heading_available=bool(np.isfinite(track.heading[index:index+2]).any()),
        bracketing_utc_seconds=track.seconds[index:index+2],bracketing_course_deg=np.degrees(track.course[index:index+2]),
        course_rate_hypothesis_degps=rate,course_change_over_aperture_deg=np.degrees(course[-1]-course[0]),
        physical_heading_established=False,arm_length_m=150.,
        fixed_heading_vs_course_proxy_excess_path_span_m=np.ptp(differences,axis=1),
        hypothetical_phase_only_coherence_loss_db=loss)
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True,layout='constrained')
    axes[0].plot(time,np.degrees(course-reference),label='Linear interpolation of reported COG')
    axes[0].plot(time,np.degrees(derived-reference),label='Direction of Hermite position derivative')
    axes[0].axhline(0,color='#aaa',lw=.7);axes[0].set(ylabel='Course relative to midpoint COG (°)',title='Course observations suggest a turn; true heading is not reported')
    axes[0].legend()
    for c,label in [(2,'Nine geometry'),(4,'ABC geometry')]:axes[1].plot(time,differences[c],label=label)
    axes[1].set(xlabel='Capture time (s)',ylabel='Excess path difference (m)',title='150 m arm: course-following rotation versus fixed N/E orientation')
    axes[1].legend();fig.savefig(out/'course-and-phase-sensitivity.png',dpi=160);plt.close(fig)
    (out/'turn-hypothesis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(serial(result)),flush=True)


if __name__=='__main__':main()
