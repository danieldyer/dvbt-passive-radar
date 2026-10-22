"""Geometric tests of the flash's sea and finite-specular-surface hypotheses.

Uses revised physical antenna priors, fixed terrain datum and interpolated AIS
position. These are predictions/sensitivity tests, not a calibration result.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv
from compare_ais_phase import NAMES, COLOURS
from flash_siting_report import EPOCH
from make_sea_dictionary import water_sphere
from motion import C, ecef, enu_basis
from specular_facet import response
from spherical_sea import geometry, horizontal_reflection, reflected_leg
from terrain import RasterTerrain, geodetic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,default=Path('docs/data/calibration_20261016_sites'))
    p.add_argument('--output',type=Path,default=Path('docs/data/isar_20261016/flash-reflection'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    prior=json.loads((args.geometry/'geometry-prior.json').read_text())
    terrain_directory=Path('build/calibration-20261013')
    terrain=RasterTerrain(terrain_directory,prior,terrain_directory/'ga-footprint-dem.tif')
    track=read_csv('docs/data/isar_20261013/rio-delta-ais.csv',reference_height_ellipsoid_m=5.)
    state=track.sample([EPOCH+164.5],max_gap_s=180)
    water=water_sphere(track,{'interval_s':[164.,165.],'nominal_start_utc':'2026-10-13T08:33:36.107619Z'},terrain)
    lat,lon=state['latitude'][0],state['longitude'][0]
    origin=ecef(lat,lon,water['tangent_geoid_m']);basis=enu_basis(lat,lon)
    rx=np.array(prior['receiver_ecef_m']);rotation=np.array(prior['ecef_to_enu'])
    tx=rx+np.array(prior['enu_transmitters_m'])@rotation
    labels=[s['id'].title() for s in prior['transmitters']]
    centre=np.array(water['centre_ecef_m']);radius=water['radius_m']
    local_tx=(tx-origin)@basis;local_rx=(rx-origin)@basis
    velocity=state['velocity_ecef_mps'][0]
    local_velocity=velocity@basis
    cog=float(np.degrees(np.arctan2(local_velocity[0],local_velocity[1]))%360)
    result={'scope':__doc__,'time_s':164.5,'physical_geometry':str(args.geometry),
        'water':water,'ais_latitude_longitude_deg':np.degrees([lat,lon]).tolist(),
        'ais_interpolated_course_deg':cog,'ais_gap_s':float(state['bracketing_interval_s'][0]),
        'relative_permittivity':80.,'conductivity_s_per_m':4.,
        'accepted_calibration':False,'ship_sea_paths':[],'sea_only_paths':[],'specular_side':[]}
    # Curved-surface geometric paths at declared heights, all three sites.
    for site,antenna in enumerate(tx):
        only=reflected_leg(rx[None,:],np.zeros((1,3)),antenna,centre,radius)
        la,lo,_=geodetic(only['specular_ecef_m'][0])
        result['sea_only_paths'].append({'site':labels[site],'latitude_deg':float(la),'longitude_deg':float(lo),
            'terrain_egm96_m':float(terrain.elevation(la,lo)),
            'excess_path_m':float(only['length_m'][0]-np.linalg.norm(antenna-rx)),
            'distance_from_ship_m':float(np.linalg.norm(only['specular_ecef_m'][0]-origin))})
        for height in (5.,20.,40.,60.):
            position=origin[None,:]+height*basis[:,2]
            paths=geometry(position,velocity[None,:],antenna,rx,centre,radius)
            entry={'site':labels[site],'scatterer_height_above_water_m':height,
                'extra_paths_m':(paths['excess_path_m'][:,0]-paths['excess_path_m'][0,0]).tolist(),
                'extra_path_rates_mps':(paths['path_rate_mps'][:,0]-paths['path_rate_mps'][0,0]).tolist(),
                'legs':[]}
            for leg,ray in enumerate(paths['reflected_legs']):
                la,lo,_=geodetic(ray['specular_ecef_m'][0])
                reflection=horizontal_reflection(ray['sin_grazing'],200e6,relative_permittivity=80.,conductivity_s_per_m=4.)[0]
                entry['legs'].append({'leg':'transmit' if leg==0 else 'receive',
                    'grazing_deg':float(np.degrees(np.arcsin(ray['sin_grazing'][0]))),
                    'fresnel_power_db':float(20*np.log10(abs(reflection))),
                    'fresnel_phase_deg':float(np.degrees(np.angle(reflection))),
                    'divergence':float(ray['divergence'][0]),'latitude_deg':float(la),'longitude_deg':float(lo),
                    'terrain_egm96_m':float(terrain.elevation(la,lo)),
                    'maximum_specular_law_error':float(ray['specular_sine_residual'].max())})
            result['ship_sea_paths'].append(entry)
        point=np.array([0.,0.,20.])
        toward=[(a-point)/np.linalg.norm(a-point) for a in (local_tx[site],local_rx)]
        normal=sum(toward);normal/=np.linalg.norm(normal)
        side_heading=(np.degrees(np.arctan2(normal[0],normal[1]))-90)%360
        result['specular_side'].append({'site':labels[site],'horizontal_normal_azimuth_deg':float((side_heading+90)%360),
            'side_axis_heading_deg':float(side_heading),'normal_tilt_from_horizontal_deg':float(np.degrees(np.arcsin(normal[2]))),
            'side_minus_interpolated_course_deg':float(np.degrees(np.angle(np.exp(1j*np.radians(side_heading-cog)))))})
    # Display finite-water path locations against the cached terrain, not an
    # invented ocean mask. Zero DEM is only a provisional water classification.
    east=np.arange(-1500.,39001.,80.);north=np.arange(-2500.,11001.,80.)
    x,y=np.meshgrid(east,north);z=-(x*x+y*y)/(radius+np.sqrt(radius*radius-x*x-y*y))
    xyz=origin+np.stack([x,y,z],axis=-1)@basis.T
    la,lo,_=geodetic(xyz);ground=terrain.elevation(la,lo)
    ocean=np.isfinite(ground)&(abs(ground)<=.5)
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for ax in axes:
        ax.contourf(x/1000,y/1000,np.where(ocean,0.,1.),[-.5,.5,1.5],colors=['#e5f3f8','#dddcd8'])
        ax.plot(0,0,'*',color='#b51d4c',ms=12,label='Ship AIS position')
        ax.plot(local_rx[0]/1000,local_rx[1]/1000,'kx',ms=8,label='Receiver')
        ax.set(aspect='equal',xlabel='East of ship (km)',ylabel='North of ship (km)')
    for site,(antenna,name) in enumerate(zip(local_tx,labels)):
        axes[0].plot(antenna[0]/1000,antenna[1]/1000,'^',label=name)
        for end in (antenna,local_rx):
            axes[0].plot([0,end[0]/1000],[0,end[1]/1000],color='#888888',lw=.7)
        r=result['sea_only_paths'][site]
        q=(ecef(np.radians(r['latitude_deg']),np.radians(r['longitude_deg']),0)-origin)@basis
        axes[0].plot(q[0]/1000,q[1]/1000,'o',mfc='none',mec='black',ms=9,label='Sea-only specular centre' if site==0 else None)
    for r in result['ship_sea_paths']:
        if r['site']!='Ornata': continue
        for leg,marker in zip(r['legs'],['o','s']):
            q=(ecef(np.radians(leg['latitude_deg']),np.radians(leg['longitude_deg']),0)-origin)@basis
            axes[1].plot(q[0]/1000,q[1]/1000,marker,label=f"{leg['leg']} leg, {r['scatterer_height_above_water_m']:g} m")
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8,loc='lower right')
    axes[0].set_title('Smooth-water reflection without a ship falls inland')
    axes[1].set(xlim=(-.5,6.),ylim=(-.5,8.),title='Sea–ship paths: Ornata, four declared heights')
    fig.suptitle('Revised priors at the flash · north up · grey: nonzero or unavailable DEM\nMarkers are geometric specular points, not complete reflecting footprints')
    fig.savefig(args.output/'01-reflection-geometry.png',dpi=170);plt.close(fig)
    heights=np.linspace(1.,60.,180)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    frequency=np.array([177.5,184.5,191.625,219.5,226.5])*1e6
    for site,name in enumerate(labels):
        position=origin+heights[:,None]*basis[:,2]
        paths=geometry(position,np.broadcast_to(velocity,position.shape),tx[site],rx,centre,radius)
        extra=paths['excess_path_m']-paths['excess_path_m'][0]
        axes[0].plot(heights,extra[3],label=name)
        if site!=0:continue
        for leg,ray in enumerate(paths['reflected_legs']):
            reflection=horizontal_reflection(ray['sin_grazing'],200e6,relative_permittivity=80.,conductivity_s_per_m=4.)
            axes[1].plot(heights,20*np.log10(abs(reflection)*ray['divergence']),label=['Transmit leg','Receive leg'][leg])
        for ch,f in enumerate(frequency):
            gt,gr=[horizontal_reflection(ray['sin_grazing'],f,relative_permittivity=80.,conductivity_s_per_m=4.) for ray in paths['reflected_legs']]
            total=np.sum(paths['relative_spreading']*np.array([np.ones(len(heights)),gt,gr,gt*gr])*np.exp(-2j*np.pi*f/C*extra),axis=0)
            axes[2].plot(heights,20*np.log10(abs(total)),color=COLOURS[ch],label=NAMES[ch])
    for ax in axes:ax.set_xlabel('Hypothesized scatterer height above water (m)');ax.legend(fontsize=8)
    axes[0].set_ylabel('Both-bounce extra path (m)')
    axes[1].set_ylabel('Ideal reflected/direct leg power ratio (dB)')
    axes[2].set(ylabel='Coherent four-path / direct-only power (dB)',ylim=(-40,12.5))
    fig.suptitle('HH smooth infinite-water sensitivity · not fitted echo strengths\nSpreading and spherical divergence included; shoreline, blockage and ray-dependent ship scattering omitted')
    fig.savefig(args.output/'02-sea-path-scales.png',dpi=170);plt.close(fig)
    # A finite plate can have carrier-dependent phase despite a perfectly
    # deterministic physical shape. Keep exact spherical path curvature.
    reference_heading=result['specular_side'][0]['side_axis_heading_deg']
    angles=np.linspace(-1.5,1.5,601)
    heading=np.radians(reference_heading+angles)
    along=np.column_stack([np.sin(heading),np.cos(heading),np.zeros(len(angles))])
    centres=np.tile([0.,0.,20.],(len(angles),1));up=np.array([0.,0.,1.])
    finite=response(centres,local_tx[0],local_rx,along,up,286.53,30.,frequency)
    dense=response(centres[::30],local_tx[0],local_rx,along[::30],up,286.53,30.,frequency,order=(128,24))
    result['facet']={'length_m':286.53,'height_m':30.,'centre_height_above_water_m':20.,
        'reference_side_heading_deg':reference_heading,'heading_offset_deg':angles.tolist(),
        'relative_field_real':finite['relative_field'].real.tolist(),'relative_field_imag':finite['relative_field'].imag.tolist(),
        'group_path_offset_m':finite['path_moment_m'].real.tolist(),
        'quadrature_maximum_absolute_difference':float(np.max(abs(finite['relative_field'][::30]-dense['relative_field']))),
        'scope':'Uniform rectangular PEC physical-optics current; direct path only. Shape sensitivity, not a fit. Height and length hypotheses, not recovered ship structure.'}
    fig,axes=plt.subplots(3,1,figsize=(12,11),layout='constrained',sharex=True)
    for ch in range(5):
        field=finite['relative_field'][:,ch]
        axes[0].plot(angles,20*np.log10(abs(field)),color=COLOURS[ch],label=NAMES[ch])
        axes[1].plot(angles,np.degrees(np.angle(field)),color=COLOURS[ch])
        axes[2].plot(angles,finite['path_moment_m'][:,ch].real,color=COLOURS[ch])
    axes[0].set(ylabel='Field / area-normalized point power (dB)',ylim=(-45,0));axes[0].legend(ncol=5)
    axes[1].set(ylabel='Phase relative to facet centre (degrees)',ylim=(-185,185))
    axes[2].set(ylabel='Group excess path relative to centre (m)',xlabel='Fixed side-axis offset from bistatic specular heading (degrees)',ylim=(-15,15))
    fig.suptitle('Finite 286.5 m × 30 m vertical facet, revised Ornata/receiver geometry\nExact spherical phase; no far-field sinc approximation and no measured data fitted')
    fig.savefig(args.output/'03-finite-facet.png',dpi=170);plt.close(fig)
    result['source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [args.geometry/'geometry-prior.json',Path('docs/data/isar_20261013/rio-delta-ais.csv'),Path(__file__),Path(__file__).with_name('specular_facet.py')]}
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    sea_rows=''.join(f"<tr><td>{r['site']}</td><td>{r['excess_path_m']:.3f} m</td><td>{r['terrain_egm96_m']:.1f} m</td><td>{r['distance_from_ship_m']/1000:.2f} km</td></tr>" for r in result['sea_only_paths'])
    side_rows=''.join(f"<tr><td>{r['site']}</td><td>{r['side_axis_heading_deg']:.2f}°</td><td>{r['normal_tilt_from_horizontal_deg']:.2f}°</td><td>{r['side_minus_interpolated_course_deg']:+.2f}°</td></tr>" for r in result['specular_side'])
    maxdelay=max(r['extra_paths_m'][3] for r in result['ship_sea_paths'])
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Flash: sea reflection and finite-facet tests</title>
<style>body{{font:16px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#18232d}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccd4d9;text-align:left}}.note{{background:#fff1d3;padding:16px}}</style>
<h1>Flash: horizontal polarisation, sea paths and a finite ship side</h1>
<p class="note">Horizontal polarisation makes a strong sea-reflected <em>leg</em> plausible. It does not establish that the observed flash comes from a smooth ocean patch. The smooth-sea specular point for Tx→sea→Rx lies inland in this geometry. Ship–sea paths are plausible, but their geometric delays remain unresolved at DVB-T bandwidth. No new antenna calibration has been accepted.</p>
<p>Calculation at 164.5 s uses the user’s revised sites and receiver, the existing one-arcsecond SRTM-derived EGM96 DEM, and interpolated AIS position. The bracketing AIS gap is {result['ais_gap_s']:g} s. The receiver is 10 m above ground. A fixed osculating sphere represents mean water level; tide, waves, refraction, urban blockage and finite shoreline propagation are not measured here.</p>
<h2>1. What horizontal polarisation changes</h2>
<p>The TE Fresnel coefficient is Γ=(sinα−√(ε−cos²α))/(sinα+√(ε−cos²α)), for grazing angle α and exp(+iωt), exp(−ikL) convention. At shallow grazing it tends to −1: large reflected field, with nearly a half-cycle phase reversal. The illustrative water values ε′=80 and conductivity 4 S/m are assumptions, not measurements of the bay. Earth divergence lowers the reflected ray strength, and direct plus reflected fields can reinforce or cancel.</p>
<p><a href="https://www.itu.int/rec/R-REC-P.527/en">ITU-R P.527</a> describes the electrical parameters of seawater. The four-path construction is established in <a href="https://www.jpier.org/ac_api/download.php?id=11101108">Chen, Zhang and Yin (2012)</a>. Their microwave scattering simulation is not itself a validated quantitative VHF model for this capture.</p>
<img src="01-reflection-geometry.png" alt="North-up antenna, ship and reflection-point geometry">
<table><tr><th>Site</th><th>Sea-only excess path</th><th>DEM elevation at specular centre</th><th>Distance from ship</th></tr>{sea_rows}</table>
<p>A perfectly smooth, stationary sea surface cannot send a strong specular return from the ship’s position to this receiver without another interaction. Tx→sea→ship→Rx, Tx→ship→sea→Rx and the path reflected on both legs are different mechanisms, and remain relevant. Water waves or a wake require a rough/dynamic surface model.</p>
<h2>2. Path delays and coherent interference</h2>
<img src="02-sea-path-scales.png" alt="Predicted sea-path delays, reflection strength and coherent interference versus height">
<p>Across all three sites and 5–60 m reflector heights, the largest calculated two-bounce extra path is {maxdelay:.3f} m. One 8 MS/s range sample is {C/8e6:.2f} m; the useful DVB-T range resolution is coarser still. These rays primarily change the complex echo within a range cell. They cannot be identified with a separate return hundreds of metres behind the ship from geometric delay alone. Destructive interference can bias an apparent phase/group-delay estimate; that is not a physically longer ray.</p>
<p>Some receiver-side reflection centres lie on land, and even a centre on water does not guarantee that the reflecting Fresnel zone is water. The infinite-water interference curves show the mechanism and sensitivity only. An earlier <a href="../../isar_20261015/sea-validation.html">30 s sea-path control</a> used the older geometry and did not recover stable common band phases; it does not settle this short flash under revised priors.</p>
<h2>3. A long ship side is a distinct, testable hypothesis</h2>
<p>The specular surface normal bisects directions from the target towards the transmitter and receiver. Its horizontal tangent is close to the AIS course ({cog:.2f}°), although COG is not true heading and the actual heading was not reported. These predictions do not fit radar phase.</p>
<table><tr><th>Site</th><th>Required side-axis heading</th><th>Normal tilt above horizontal</th><th>Side heading minus interpolated COG</th></tr>{side_rows}</table>
<img src="03-finite-facet.png" alt="Physical-optics finite-facet amplitude, phase and group path across the five carriers">
<p>The aperture integral retains exact spherical path length over the finite rectangle, HH PEC current projection and spreading. It is a deterministic, frequency-dependent complex response even for a perfectly fixed surface. The 64×12 and 128×24 quadratures agree to {result['facet']['quadrature_maximum_absolute_difference']:.2g} in normalized complex field. Independent tests check the point limit, far-field rectangular transform, frequency derivative and rigid-coordinate invariance. The approximation omits edge diffraction, corrugations, containers, sea interactions and absolute RCS.</p>
<p>The distinction matters: a specular point sliding tangentially over an infinite smooth surface does not contribute an arbitrary extra first-order path rate. Fermat stationarity makes that tangential term zero. A finite aperture, interference, rotation or changing illuminated support must actually predict any additional phase variation. The general physical-optics treatment is discussed in <a href="https://www.mdpi.com/2072-4292/18/8/1219">Di Martino et al. (2026)</a>; the exact spherical-phase numerical integral here is our diagnostic extension.</p>
<h2>4. Measured-data check</h2>
<p>The <a href="../flash-pair-wide/comparison.png">one-versus-two-echo comparison</a> uses unchanged samples and the same conditional covariance. Each component has one constant phase and a quadratic path driving both delay migration and RF phase. Only real nonnegative strengths vary per training bin; unused 125 ms blocks are predicted using training strengths. It tests whether a single fitted phase trajectory is being biased by unresolved signal, not whether the two fitted components are proven physical points. Boundary hits, range-width sensitivity and residual error must be resolved before using those trajectories to constrain antennas.</p>
<p><a href="analysis.json">All numerical predictions and source hashes</a>. The single-point <a href="../transmitter-prior-refit/index.html">old/new-prior comparison</a> remains available. These results motivate testing the finite-facet response against the full complex flash rather than treating one fitted phase centre as a surveyed reflector.</p></html>'''
    (args.output/'index.html').write_text(page)
    print(json.dumps({'sea_only':result['sea_only_paths'],'side':result['specular_side'],
        'max_ship_sea_extra_path_m':maxdelay,'quadrature_error':result['facet']['quadrature_maximum_absolute_difference']},indent=2),flush=True)


if __name__=='__main__':main()
