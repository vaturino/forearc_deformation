#!/usr/bin/env python3

import numpy as np
import os
import netCDF4 as nc
import glob
import math
from scipy.special import erfc
from numba import jit
import time

####### define numba function to track where is slab and weak zone in the whole array and return the index of the array and mark it in 3-d array using these index later.
@jit(nopython=True)
def calculate_slabs(lon_t, lat_t, depth_t, glon,glat,gdepth, dx,dz):
  slab=[]
  for j in range(len(lon_t)):
      ix = [idx for idx, val in enumerate(glon) if lon_t[j] >= np.round((val-(dx/2)),2) and lon_t[j] < np.round((val+(dx/2)),2)]
      iy = [idx for idx, val in enumerate(glat) if lat_t[j] >= np.round((val-(dx/2)),2) and lat_t[j] < np.round((val+(dx/2)),2)]
      iz = [idx for idx, val in enumerate(gdepth) if depth_t[j] >= np.round((val-(dz/2)),2) and depth_t[j] < np.round((val+(dz/2)),2)]
    
      dnorm = 1 - (j / (len(lon_t) - 1 ))
      if len(ix)==1 and len(iy)==1 and len(iz)==1:
          slab.append((ix,iy,iz,dnorm))

  return slab

@jit(nopython=True)
def calculate_composi(lon_t, glon, glat, gdepth, lat_t, depth_t,dx,dz):
  composi=[]
  for j in range(len(lon_t)):
      ix = [idx for idx, val in enumerate(glon) if lon_t[j] >= np.round((val-(dx/2)),2) and lon_t[j] < np.round((val+(dx/2)),2)]
      iy = [idx for idx, val in enumerate(glat) if lat_t[j] >= np.round((val-(dx/2)),2) and lat_t[j] < np.round((val+(dx/2)),2)]
      iz = [idx for idx, val in enumerate(gdepth) if depth_t[j] >= np.round((val-(dz/2)),2) and depth_t[j] < np.round((val+(dz/2)),2)]
          
      if len(ix)==1 and len(iy)==1 and len(iz)==1:
          composi.append((ix,iy,iz))
  return composi       

##############################################
T1=time.time()
#set up grids
dx = 0.2 ; dz = 10

# glon = np.linspace(150,360,1801 ,endpoint=True , dtype=np.float32)
# glat = np.linspace(-90, 90, 901, endpoint=True, dtype=np.float32)
glon = np.linspace(0, 360,1801,endpoint=True, dtype=np.float32)
glat = np.linspace(-90, 90, 901, endpoint=True, dtype=np.float32)
gdepth = np.linspace(0, 1100,111,endpoint=True, dtype=np.float32)

# define parameters
temp_ref = 1573 # Kelvin    Mantle interior temperature
temp_surf = 273 # surface temperature
W = 30 # km    width of plate boundary zone, the 30km wide weaker zone mentioned in Sam's paper

#initialize arrays
slabs = np.zeros((len(glat),len(glon),len(gdepth)), dtype=bool)  #slabs = slab.astype(int)
slab_temp = np.full((len(glat),len(glon),len(gdepth)), np.nan, dtype=np.float32) #slab_temp = np.full((len(glat),len(glon),len(gdepth)),0, dtype=np.float32)
Composi = np.full((len(glat),len(glon),len(gdepth)),0, dtype=np.uint8)

# Read in and iterate slab-by-slab
directory = '/home/vturino/PhD/projects/forearc_deformation/slab2/sam/'
c="dep"
pattern = os.path.join(directory, f"*{c}*grd")
fs = [os.path.basename(file) for file in glob.glob(pattern)]
name = "sam"

for filename in fs:
    slabname = filename.split('_')[0]
    print(slabname)
    fnhead = filename.split('dep')[0]
    fntail = filename.split('dep')[1]
    dep = nc.Dataset('%s/%s' %(directory, filename))
    thk = nc.Dataset('%s/%sthk%s' %(directory,fnhead, fntail))
    str = nc.Dataset('%s/%sstr%s' %(directory,fnhead, fntail))
    dip = nc.Dataset('%s/%sdip%s' %(directory,fnhead, fntail))
 
    lon = dep.variables['x'][:]
    lat = dep.variables['y'][:]
    dep = -dep.variables['z'][:]
    thk = thk.variables['z'][:]
    str = str.variables['z'][:]
    dip = dip.variables['z'][:]

    # print(np.min(lon), np.max(lon), np.min(lat), np.max(lat), np.min(dep), np.max(dep))

    lat, lon  = np.meshgrid(lat, lon,indexing='ij')


    m = ~np.isnan(dep)
  
    lon = lon[m]; lat = lat[m]; thk = thk[m]; str = str[m]; dip = dip[m]; dep = dep[m]


# Define 3-D slab below slab surface, point-by-point
    for i in range(len(lon)): 

        # Compute spacing of latitude and longitude at point (function of depth and latitude)
        Dx = (2 * math.pi * (6371 - dep[i]) * math.cos(math.radians(lat[i]))) / 360
        Dy = (2 * math.pi * (6371 - dep[i])) / 360        

        # Compute normal vector from strike and dip
        normx = math.cos(math.radians(str[i])) * math.sin(math.radians(dip[i]))
        normy = -math.sin(math.radians(str[i]))* math.sin(math.radians(dip[i]))
        normz = math.cos(math.radians(dip[i]))

        #  Construct transect of 25 points normal to slab top (enough points to be finer than grid spacing defined above)
        lon_t = np.linspace(lon[i] - normx * thk[i] / Dx, lon[i], 25, endpoint=True)#.reshape(1,-1)
        lat_t = np.linspace(lat[i] - normy * thk[i] / Dy, lat[i], 25, endpoint=True)#.reshape(1,-1)
        depth_t = np.linspace(dep[i] + normz * thk[i], dep[i], 25, endpoint=True)#.reshape(1,-1)

        lon_t = lon_t[depth_t >= 0]
        lat_t = lat_t[depth_t >= 0]
        depth_t = depth_t[depth_t >= 0]       
        


        ########## snap each transect point to nearest grid point and mark slab location
        slabindex=calculate_slabs(lon_t,lat_t,depth_t,glon,glat,gdepth,dx,dz) 
        for indices in slabindex:
            ix,iy,iz,dnorm = indices
            slabs[iy,ix,iz] = 1
            slab_temp[iy,ix,iz] = temp_ref + (temp_surf - temp_ref) * erfc (1.16 * dnorm)

        # similarly, define the plate boundary zone above slab surface (weak zone)  using a transect of 10 points
        lon_t = np.linspace(lon[i] - normx*(-W)/Dx, lon[i],10, endpoint= True)
        lat_t = np.linspace(lat[i] - normy*(-W)/Dy, lat[i],10, endpoint= True)
        depth_t = np.linspace(dep[i] + normz*(-W), dep[i], 10, endpoint= True)

        lon_t = lon_t[depth_t>=0]
        lat_t = lat_t[depth_t>=0]
        depth_t = depth_t[depth_t>=0]
        ########################################## snap to nearest grid point, mark down weak zone composition
        composi_index=calculate_composi(lon_t, glon, glat, gdepth, lat_t, depth_t,dx,dz)         ########################## use numba function
        for indices in composi_index:
            ix,iy,iz = indices
            Composi[iy,ix,iz] = 2

Composi[slabs]=0

ds = nc.Dataset('../slab_geometries/sam_geometry.nc', 'w', format="NETCDF4")
glon1 = ds.createDimension('glon1', len(glon))
glat1 = ds.createDimension('glat1', len(glat))
gdepth1 = ds.createDimension('gdepth1', len(gdepth))

glon1 = ds.createVariable('glon1',np.float32,('glon1',))
glon1[:] = glon

glat1 = ds.createVariable('glat1', np.float32,('glat1',))
glat1[:] = glat

gdepth1 = ds.createVariable('gdepth1', np.float32 ,('gdepth1',))
gdepth1[:] = gdepth

slab1  = ds.createVariable('slabs1', 'i1' ,('glat1','glon1','gdepth1'))
slab1[:] = slabs

Composi1  = ds.createVariable('Composi1', np.uint8 ,('glat1','glon1','gdepth1'))
Composi1[:] = Composi

slab_temp1  = ds.createVariable('slab_temp1', np.float32 ,('glat1','glon1','gdepth1'))
slab_temp1[:] = slab_temp

print(np.sum(slabs))
print(np.sum(Composi))


ds.close()
T2=time.time()
T3=(T2-T1)
print(T3)