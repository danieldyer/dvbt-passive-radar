"""Uncalibrated independent-channel geographic diagnostic; never a phase fit.

Projects the static, three-second matched-filter sufficient statistics onto
the DEM. Independent complex channel gains are profiled out, so the statistic
is a sum of channel likelihoods. It does not claim coherent spatial focusing.
"""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from geographic_io import read_profiles
from geometry import C,RF,ecef,enu_rotation
from terrain import RasterTerrain


def scene(terrain_directory,output,dem,nx=1280,ny=1024):
    prior=json.loads((output/'geometry-prior.json').read_text())
    terrain=RasterTerrain(terrain_directory,prior,dem)
    rx=prior['receiver'];lat,lon=rx['latitude_deg'],rx['longitude_deg']
    receiver=ecef(lat,lon,terrain.elevation(lat,lon)+terrain.undulation(lat,lon)+rx['antenna_agl_m'])
    rotation=enu_rotation(lat,lon)
    acma=json.loads((output/'acma-transmitters.json').read_text())
    transmitters=[]
    for rf in RF:
        device=next(d for d in acma['devices'] if float(d['FREQUENCY'])==rf and d['STATION_TYPE']=='Main')
        site=acma['sites'][device['SITE_ID']];la,lo=float(site['LATITUDE']),float(site['LONGITUDE'])
        tx=ecef(la,lo,terrain.elevation(la,lo)+terrain.undulation(la,lo)+float(device['HEIGHT']))
        transmitters.append(rotation@(tx-receiver))
    longitude=np.linspace(144.755,145.595,nx);latitude=np.linspace(-38.095,-37.535,ny)
    lo,la=np.meshgrid(longitude,latitude);height=terrain.elevation(la,lo)
    valid=np.isfinite(height)
    point=ecef(la,lo,np.where(valid,height,0)+terrain.undulation(la,lo))
    x=(point-receiver)@rotation.T
    geometry=np.zeros((nx*ny,5,4),np.float32)
    ranges=[]
    for ch,tx in enumerate(transmitters):
        length=np.linalg.norm(x-tx,axis=-1)+np.linalg.norm(x,axis=-1)-np.linalg.norm(tx)
        ranges.append(length)
        geometry[:,ch,0]=(length*8e6/C).ravel()
        phase=np.remainder(2*np.pi*RF[ch]/C*length,2*np.pi)
        geometry[:,ch,1]=np.cos(phase).ravel();geometry[:,ch,2]=np.sin(phase).ravel()
    common=valid&np.all((np.array(ranges)>=600)&(np.array(ranges)<=30000),axis=0)
    geometry[:,:,3]=common.ravel()[:,None]
    return geometry,longitude,latitude,height,common


def integrate(records,centre,blocks=92):
    dt=30*8704/8e6;first=int(centre/dt)-blocks//2
    profiles=np.zeros((5,1024),np.complex64);weights=np.zeros((5,4),np.float32)
    report=[]
    for channel in range(5):
        keep=(records['channel']==channel)&(records['bin']>=first)&(records['bin']<first+blocks)
        rows=records[keep]
        if len(rows)!=blocks or len(np.unique(rows['bin']))!=blocks or len(np.unique(rows['segment']))!=1:
            raise ValueError('Requested CPI is incomplete or crosses a discontinuity')
        w=rows['information'];information=w.sum()
        if information<=0:
            raise ValueError('No valid reconstructed reference in CPI')
        profiles[channel]=np.sum(rows['values']*w[:,None],axis=0)/information
        pulse=float(np.sum(rows['psf'][:,64].real*w)/information)
        weights[channel]=[information,1,0,pulse]
        report.append({'channel':channel,'valid_blocks':int(np.count_nonzero(w)),
                       'inverse_covariance_scale':float(information),'mean_template_peak':pulse})
    return profiles,weights,{'start_seconds':first*dt,'end_seconds':(first+blocks)*dt,
                            'coherence_seconds':blocks*dt,'channels':report}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['templates','terrain','dem','output','work','executable']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--centre',type=float,default=50.)
    args=p.parse_args();args.work.mkdir(parents=True,exist_ok=True)
    geometry,longitude,latitude,height,mask=scene(args.terrain,args.output,args.dem)
    path=args.work/'main-geographic-geometry.bin';geometry.tofile(path)
    _,records=read_profiles(args.templates)
    profiles,weights,description=integrate(records,args.centre)
    process=subprocess.run([str(args.executable),str(path),'5'],
        input=profiles.tobytes()+weights.tobytes(),stdout=subprocess.PIPE,check=True)
    fields=np.frombuffer(process.stdout,np.float32).reshape(len(latitude),len(longitude),4)
    power=fields[:,:,2];db=10*np.log10(np.maximum(power,np.finfo(np.float32).tiny))
    lower=float(np.percentile(db[mask],1));upper=float(db[mask].max())
    fig,ax=plt.subplots(figsize=(14,10))
    fig.subplots_adjust(left=.08,right=.90,bottom=.12,top=.91)
    ax.set_facecolor('#11151d')
    image=ax.imshow(np.where(mask,db,np.nan),origin='lower',extent=[longitude[0],longitude[-1],latitude[0],latitude[-1]],
                    cmap='inferno',vmin=lower,vmax=upper,interpolation='nearest',aspect=1/np.cos(np.deg2rad(-37.8)))
    ax.contour(longitude,latitude,height,levels=[10,50,100,200,300,400,500,600,800],colors='#c3c6ca',linewidths=.35,alpha=.55)
    rx=json.loads((args.output/'geometry-prior.json').read_text())['receiver']
    ax.plot(rx['longitude_deg'],rx['latitude_deg'],'+',color='white',markersize=9)
    ax.annotate('Receiver',(rx['longitude_deg'],rx['latitude_deg']),xytext=(5,6),textcoords='offset points',color='white')
    ax.plot(145.34661,-37.837476,'^',color='white',markersize=6)
    ax.annotate('Mt Dandenong',(145.34661,-37.837476),xytext=(5,6),textcoords='offset points',color='white')
    ax.set(xlabel='Longitude (degrees east)',ylabel='Latitude (degrees north)',
           title=f'Geographic diagnostic · {description["start_seconds"]:.2f}–{description["end_seconds"]:.2f} s\n'
                 'Independent channel likelihoods · licensed main sites · UNCALIBRATED')
    fig.colorbar(image,ax=ax,label='Summed independent-channel matched-filter statistic (dB)')
    fig.text(.08,.025,'Terrain: Geoscience Australia 1 arcsecond SRTM-derived DEM, EGM96. Grey lines: terrain contours.\n'
             'Static 3 s integration; moving echoes attenuated. Elliptical ambiguity ridges are not resolved geographic targets.',fontsize=9)
    fig.savefig(args.output/'geographic-uncalibrated.png',dpi=150);plt.close(fig)
    description.update({'status':'uncalibrated independent-response geographic diagnostic; not coherent image formation',
        'range_support_m':[600,30000],'colour_scale_db':[lower,upper],
        'terrain_source':'https://services.ga.gov.au/gis/rest/services/DEM_SRTM_1Second_2024/MapServer',
        'terrain_vertical_datum':'EGM96','terrain_native_dted2':False})
    (args.output/'geographic-uncalibrated.json').write_text(json.dumps(description,indent=2))
    print(json.dumps(description,indent=2))


if __name__=='__main__':
    main()
