"""Geodetic height bookkeeping shared by calibration and geographic support tools."""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def geodetic(x):
    x=np.asarray(x);rho=np.hypot(x[...,0],x[...,1]);z=x[...,2]
    e2=6.6943799901413165e-3;a=6378137.
    latitude=np.arctan2(z,rho*(1-e2))
    for _ in range(7):
        n=a/np.sqrt(1-e2*np.sin(latitude)**2)
        latitude=np.arctan2(z+e2*n*np.sin(latitude),rho)
    n=a/np.sqrt(1-e2*np.sin(latitude)**2)
    height=rho*np.cos(latitude)+z*np.sin(latitude)-n*(1-e2*np.sin(latitude)**2)
    return np.rad2deg(latitude),np.rad2deg(np.arctan2(x[...,1],x[...,0])),height


class Terrain:
    def __init__(self,directory,geometry):
        self.directory=Path(directory)
        self.receiver=np.asarray(geometry['receiver_ecef_m'])
        self.rotation=np.asarray(geometry['ecef_to_enu'])
        self.geoid=np.asarray(Image.open(self.directory/'us_nga_egm96_15.tif'))
        self.tiles={}

    def elevation(self,latitude,longitude):
        lat,lon=np.broadcast_arrays(latitude,longitude)
        shape=lat.shape;lat,lon=lat.ravel(),lon.ravel()
        ilat,ilon=np.floor(lat).astype(int),np.floor(lon).astype(int)
        values=np.empty(len(lat))
        for la,lo in np.unique(np.c_[ilat,ilon],axis=0):
            key=(la,lo)
            if key not in self.tiles:
                name=f"{'N' if la>=0 else 'S'}{abs(la):02d}{'E' if lo>=0 else 'W'}{abs(lo):03d}.npy"
                self.tiles[key]=np.load(self.directory/name,mmap_mode='r')
            tile=self.tiles[key];n=tile.shape[0]-1
            keep=(ilat==la)&(ilon==lo)
            values[keep]=map_coordinates(tile,[(la+1-lat[keep])*n,(lon[keep]-lo)*n],order=1,output=np.float64)
        if np.any(values<-500):
            raise ValueError('Void in terrain support')
        return values.reshape(shape)

    def undulation(self,latitude,longitude):
        lat,lon=np.broadcast_arrays(latitude,longitude)
        return map_coordinates(self.geoid,[(90-lat.ravel())/.25,(lon.ravel()+180)/.25],
                               order=1,output=np.float64).reshape(lat.shape)

    def clearance(self,positions,derivative=False):
        x=np.asarray(positions)
        latitude,longitude,height=geodetic(self.receiver+x@self.rotation)
        result=height-self.undulation(latitude,longitude)-self.elevation(latitude,longitude)
        if not derivative:
            return result
        gradient=np.empty_like(x)
        for axis in range(3):
            delta=np.zeros(3);delta[axis]=.5
            gradient[:,axis]=self.clearance(x+delta)-self.clearance(x-delta)
        return result,gradient


class RasterTerrain(Terrain):
    """North-up, geographic PixelIsArea DEM with an explicitly EGM96 datum."""
    def __init__(self,directory,geometry,path):
        super().__init__(directory,geometry)
        image=Image.open(path)
        keys=image.tag_v2[34735]
        tags={keys[i]:keys[i+3] for i in range(4,len(keys),4)}
        if tags.get(2048)!=4326 or tags.get(1025)!=1:
            raise ValueError('DEM must declare WGS84 and PixelIsArea')
        tie=image.tag_v2[33922];scale=image.tag_v2[33550]
        self.west,self.north=tie[3],tie[4];self.dx,self.dy=scale[:2]
        self.values=np.asarray(image).astype(float)
        self.values[(self.values < -500)|(self.values > 10000)]=np.nan

    def elevation(self,latitude,longitude):
        lat,lon=np.broadcast_arrays(latitude,longitude)
        # The GeoTIFF tie point is the outer corner of the first pixel.
        rows=(self.north-lat.ravel())/self.dy-.5
        columns=(lon.ravel()-self.west)/self.dx-.5
        ny,nx=self.values.shape
        # Decimal-degree roundoff may put an exact edge centre just outside.
        valid=(rows>=-1e-8)&(rows<=ny-1+1e-8)&(columns>=-1e-8)&(columns<=nx-1+1e-8)
        values=map_coordinates(self.values,[np.clip(rows,0,ny-1),np.clip(columns,0,nx-1)],
                               order=1,mode='constant',cval=np.nan,prefilter=False)
        return np.where(valid,values,np.nan).reshape(lat.shape)
