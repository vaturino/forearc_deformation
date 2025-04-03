#!/usr/bin/env python3

import netCDF4 as nc
import numpy as np
from scipy.interpolate import interpn
from scipy.interpolate import griddata
from scipy.special import erfc
from scipy.interpolate import RegularGridInterpolator
from numba import jit
import time
import scipy.io
import math
#from skimage.morphology import dilation
from scipy.ndimage import generate_binary_structure, binary_dilation

T1=time.time()
print(T1)
# Make gridded vertical plate boundaries in 2-d from Bird2003 dataset
# specify parameters
W = 15  # half-width of plate boundaries zone (km)
dx = 0.2  # grid spacing (degree)   # keep same spacing with slab geometry

# Read in Bird-2003 plate boundaries --- collected by Sam
data = scipy.io.loadmat('/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/plbd_0_360.mat')
lon = data['lon'][:,-1]  # 6870x1
lat = data['lat'][:,-1]  # 6870x1  


lon = np.interp(np.linspace(0,1, len(lon)*5), np.linspace(0,1, len(lon)),lon )  # 30240  ###
lat = np.interp(np.linspace(0,1, len(lat)*5),np.linspace(0,1, len(lat)), lat)  # 30240
#print(lon.shape, lat.shape)
lat = lat[~np.isnan(lon)]
lon = lon[~np.isnan(lon)]


# initialize arrays
glon1 = np.arange(0, 360+dx, dx , dtype=np.float32)   #1801
glat1 = np.arange(-90, 90+dx, dx, dtype=np.float32)   #901
C = np.zeros((len(glat1),len(glon1)),dtype=np.uint8)  # 901x1801

glat, glon = np.meshgrid(glat1, glon1,indexing = 'ij')


dlat = 6371*1000 * (np.pi) /180    # is a float   1. 11194 
dlon = 6371*1000 * (np.pi) /180 * np.cos(np.radians(lat))  #  30240  ##matlab 30900
#print(dlon[1])

for i in range(len(lon)):
    d = np.sqrt(((lon[i] - glon) * dlon[i])**2 + ((lat[i] - glat) * dlat)**2) / 1000 
    C[d <= W] = 1


#print(np.sum(C))   #  21494   matlab ---20578


## add in slab plate boundaries
slab2_file = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry.nc",'r')
depth  = slab2_file.variables['gdepth1'][:]
slab_C = slab2_file.variables['Composi1'][:]
slab   = slab2_file.variables['slabs1'][:]
slab2_file.close()

C = C[:,:,np.newaxis]
C = np.tile(C, (1,1,len(depth)))
C = np.maximum(C, slab_C)
C[slab]=0


# get rid of vertical plate boundaries where they should be slab-normal
structuring_element = np.ones((3, 3))
m = binary_dilation(C[:,:,0] == 2, structure=structuring_element).astype(int)


for i in range(C.shape[2]):
    C_= C[:,:,i].copy()
    C_[np.logical_and(m==1,C_==1)] =0
    C[:,:,i] = C_
C[slab]=0


# Add lithosphere
lith = nc.Dataset("mean_no_slabs.l.grd",'r')
lon = lith.variables['lon'][:]
lat = lith.variables['lat'][:]
lthick = lith.variables['z'][:]

newlith=interpn((lat,lon), lthick, (glat, glon), method='linear')

for i in range(1, len(depth)):
    C_ = C[:,:,i]
    C_[newlith <= depth[i-1]]=0
    C[:,:,i] =C_


C_max = np.max(np.max(C, axis=0), axis=0)
depth = depth[C_max > 0]

#gdepth = np.concatenate((depth, [depth[-1] + 1, 2900]))
gdepth = np.concatenate((depth, [240,250,260,270,280,290,300, 2900]))
print(gdepth)
C = C[:, :, :len(gdepth)]  

C[C==2] = 1

# Add lithosphere
lith = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/mean_no_slabs.l.grd",'r')
lon = lith.variables['lon'][:]
lat = lith.variables['lat'][:]
lthick = lith.variables['z'][:]

newlith=interpn((lat,lon), lthick, (glat, glon), method='linear')

dz=10
continents=np.zeros_like(C)

#### add continents
for i in range(len(glat1)):
   for j in range(len(glon1)):
      if newlith[i,j]>=170:
       # print(i,j,newlith[i,j])
        iz = [idx for idx, val in enumerate(gdepth) if newlith[i,j] >= np.round((val-(dz/2))) and newlith[i,j] < np.round((val+(dz/2)))]
        #ix = [idx for idx, val in enumerate(glon) if lon_t[j] >= np.round((val-(dx/2)),2) and lon_t[j] < np.round((val+(dx/2)),2)]
        iz = iz[0]+1
        continents[i,j,:iz]=1
    

glon = np.arange(0, 360+dx, dx , dtype=np.float32)   #1801
glat = np.arange(-90, 90+dx, dx, dtype=np.float32)  

ds = nc.Dataset('/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry_continents.nc', 'w', format="NETCDF4")
glon_di = ds.createDimension('glon_di', len(glon1))
glat_di = ds.createDimension('glat_di', len(glat1))
gdepth_di = ds.createDimension('gdepth_di', len(gdepth))

glon_var = ds.createVariable('glon_var',np.float32,('glon_di',))
glon_var[:] = glon1

glat_var = ds.createVariable('glat_var', np.float32,('glat_di',))
glat_var[:] = glat1

gdepth_var = ds.createVariable('gdepth_var', np.float32,('gdepth_di',))
gdepth_var[:] = gdepth

C_var = ds.createVariable('C', 'i1' ,('glat_di','glon_di','gdepth_di'))
C_var[:] =  C

continents_var = ds.createVariable('continents', 'i1' ,('glat_di','glon_di','gdepth_di'))
continents_var[:] =  continents

ds.close()

print(gdepth)



fname = '/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry_continents.txt'
glon = np.arange(0, 360+dx, dx , dtype=np.float32)   #1801
glat = np.arange(-90, 90+dx, dx, dtype=np.float32)   #901
glatm, glonm, gdepthm,= np.meshgrid(glat, glon, gdepth, indexing = 'ij')
# convert into spherical coordinates
r = (6371-gdepthm)*1000 
phi = glonm *np.pi /180
theta = (90 - glatm) * np.pi / 180
print(phi.shape, theta.shape)

r = r[:,:-1,:]
phi = phi[:,:-1,:]
theta = theta[:,:-1,:]

with open (fname, 'w') as fid:
    fid.write('# POINTS: ' + str(r.shape[2]) + ' ' + str(r.shape[1]) + ' ' + str(r.shape[0]) + '\n')
    fid.write('# Columns: r phi theta plbd continents\n')
    fid.write('# Bird plate boundaries, width of boundary zone 30 km (Composition 1) \n')
    fid.write('# Dipping boundaries above slabs, following SLAB2 slab top \n')
    fid.write('# Plate boundaries go as deep as lithosphere, from Steinberger and Becker 2018 \n')
    fid.write('# Rest of mantle is background orrrr I can continents information\n')
    
    if theta[1, 0, 0] > theta[0, 0, 0]:
        ii = range(r.shape[0])
    else:
        ii = range(r.shape[0]-1, -1, -1)


    if phi[0, 1, 0] > phi[0, 0, 0]:
        jj = range(r.shape[1])
    else:
        jj = range(r.shape[1]-1, -1, -1)

    if r[0, 0, 1] > r[0, 0, 0]:
        kk = range(r.shape[2])
    else:
        kk = range(r.shape[2]-1, -1, -1)

    for i in ii:  # Vary colatitude
        for j in jj:  # Vary longitude
            for k in kk:  # Vary radius (increasing)
                fid.write(f"%.0f %.4f %.4f %i %i \n" % (r[i,j,k], phi[i,j,k], theta[i,j,k], C[i,j,k], continents[i,j,k]))
         
T2=time.time()
print(T2-T1)
