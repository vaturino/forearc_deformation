#!/usr/bin/env python3

import netCDF4 as nc
import numpy as np
from scipy.interpolate import interpn
from scipy.interpolate import griddata
from scipy.special import erfc
from scipy.interpolate import RegularGridInterpolator
from numba import jit
import time
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)


T1=time.time()
#Define parameters
vs_to_density_scaling = 0.2
vp_to_density_scaling = 0.4
rho_ref = 3300
temp_ref = 1573
temp_surf = 273
expansivity = 3e-5

min_lat = -60
max_lat = 20
min_lon = 220
max_lon = 320


# Define structured grid, non-uniform resolution, refined where slabs
dx = 0.2 #horizontal resolution away from slabs
dz = 10 #vertical resolution for upper mantle
slab2_file = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry.nc",'r')    ### Your slab geometry nc file
glon = slab2_file.variables['glon1'][:]   # 0.2 degree
glat = slab2_file.variables['glat1'][:]   # 0.2 degree
slab = slab2_file.variables['slabs1'][:]
slab2_file.close()

slabs = np.sum(slab, axis=2).astype(bool)
slabs_lon = np.sum(slabs, axis=0).astype(bool) # Longitude where have slabs   maybe need to transpose  1801,1 -> 1,1801
slabs_lat = np.sum(slabs, axis=1).astype(bool) # Latitudes where have slabs


#combine global with high-resolution(0.2 degrees) where slabs
glon1 = np.concatenate((glon[slabs_lon], np.arange(min_lon, max_lon+dx, dx)))
glon1 = np.sort(np.unique(np.round(glon1, decimals=1)))   #1382

glat1 = np.concatenate((glat[slabs_lat], np.arange(min_lat, max_lat+dx, dx)))
glat1 = np.sort(np.unique(np.round(glat1, decimals=1)))   # 665   # slab resolution is 0.2 degree, so one decimal number will be fine
slabs_lon=slabs_lat=None


# use depth from tomography model for lower mantle
#tomogrpahy_file = nc.Dataset("/home/tzhao/Tao_script/model_prepare/update/tomography_background/Thrastarson2024/bssa-2023273_supplement_reveal.nc/BSSA-2023273_Supplement_REVEAL_changelon_dlnvs.nc",'r') #S362ANI_with_taokai2018.nc",'r')
tomogrpahy_file = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/t_savani-dlnvs-add.nc",'r')   # tomography model input for mantle temperature calculation
depth = tomogrpahy_file.variables['dep'][:]
lon = tomogrpahy_file.variables['lon'][:]
lat = tomogrpahy_file.variables['lat'][:]
tomogrpahy_file.close()
lon = np.round(lon, decimals=2) ;lat = np.round(lat, decimals=2)

#read savani datafile, ###############################using TX2019 depth   
gdepth1 = np.concatenate((np.arange(0, 670, 10), depth[(depth > 660)])).astype(np.float32)

glatm, glonm, gdepthm = np.meshgrid(glat1, glon1, gdepth1, indexing='ij')
print(f"Depth range: {gdepth1.min()} km to {gdepth1.max()} km")


# Use tomography below certain depth, with a smooth transition
Dc = 300 #threshold
sw = 100 # smoothing width

# interpolate onto grid
tomogrpahy_file = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/t_savani-dlnvs-add.nc",'r')   # tomography model input for mantle temperature calculation
#tomogrpahy_file = nc.Dataset("/home/tzhao/Tao_script/model_prepare/update/tomography_background/Thrastarson2024/bssa-2023273_supplement_reveal.nc/BSSA-2023273_Supplement_REVEAL_changelon_dlnvs.nc",'r') #S362ANI_with_taokai2018.nc",'r')
depth = tomogrpahy_file.variables['dep'][:]
lon = tomogrpahy_file.variables['lon'][:]
lat = tomogrpahy_file.variables['lat'][:]
dlnvs = tomogrpahy_file.variables['dlnvs'][:]
dlnvs = np.transpose(dlnvs, (1,2,0))
tomogrpahy_file.close()


print("start dlnvs")
newdlnvs = interpn((lat,lon, depth), dlnvs, (glatm,glonm, gdepthm), method='linear',bounds_error=False)


print("calculate temp")
temp = np.float32(temp_ref + (-1./expansivity) * vs_to_density_scaling * newdlnvs/100)


W = (gdepth1 - Dc + sw/2) /sw
W[W<0]=0
W[W>1]=1
W1 = W[np.newaxis, np.newaxis,:]
W1 = np.tile(W1, (len(glat1),len(glon1),1))
temp = (W1 * temp + (1-W1)* temp_ref)
#print(temp)

# Add lithosphere
lith = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/mean_no_slabs.l.grd",'r')   # lithosphere data input  # attached
lon = lith.variables['lon'][:]
lat = lith.variables['lat'][:]
lthick = lith.variables['z'][:]
glatm, glonm= np.meshgrid(glat1,glon1, indexing = 'ij')

newlith=interpn((lat,lon), lthick, (glatm, glonm), method='linear')

for i in range(temp.shape[2]):
    d= gdepthm[0,0,i] 
    if d<Dc:
        temp[:,:,i]= temp_ref + (temp_surf - temp_ref) *erfc(d/(0.8621 * newlith))

slab2_file = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry.nc",'r')   # your slab geometry input nc file

glon = slab2_file.variables['glon1'][:]
glat = slab2_file.variables['glat1'][:]
gdepth = slab2_file.variables['gdepth1'][:]
slab_temp= slab2_file.variables['slab_temp1'][:]
slab2_file.close()

glatm, glonm,gdepthm = np.meshgrid( glat1, glon1, gdepth1, indexing='ij')

slab_temp[slab_temp==0] = np.nan


print("start new slab_temp")
newslab_temp= interpn((glat,glon,gdepth), slab_temp, (glatm,glonm,gdepthm), method='nearest', bounds_error=False, fill_value=np.nan)


temp = np.fmin(temp, newslab_temp)


newfn = "/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/temp_with_slab2supp_modified_and_SAVANI.nc"
ds = nc.Dataset(newfn, 'w', format='NETCDF4')
lon = ds.createDimension('lon', len(glon1))
lat = ds.createDimension('lat', len(glat1))
dep = ds.createDimension('dep', len(gdepth1))  

lons = ds.createVariable('lon','f4',('lon',))
lons.units = 'degree_east_resolution_1_degree_from_0_to_360'
lats = ds.createVariable('lat','f4',('lat',))
lats.units = 'degree_north_resolution_0.5_degree_from_-90_to_90'
deps = ds.createVariable('dep','f4',('dep',))
deps.units = 'km, from 0-2900 km'
vartemp = ds.createVariable('finaltemp', 'f4', ('lat','lon','dep'))

lons[:] = glon1
lats[:] = glat1
deps[:] = gdepth1
vartemp[:]=temp
ds.close()


fname = '/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/temp_with_slab2supp_modified_and_SAVANI.txt'

glatm, glonm,gdepthm,= np.meshgrid(glat1, glon1, gdepth1, indexing = 'ij')
# convert into spherical coordinates
r = (6371-gdepthm)*1000 
phi = glonm *np.pi /180
theta = (90 - glatm) * np.pi / 180
print(phi.shape, theta.shape, temp.shape)

with open (fname, 'w') as f:
    f.write('# POINTS: ' + str(r.shape[2]) + ' ' + str(r.shape[1]) + ' ' + str(r.shape[0]) + '\n')
    f.write("# Columns: r phi theta temperature\n")
    f.write("# Savani (dvs) for mantle below 300 km depth, 100 km smoothing zone\n")
    f.write("# Steinberger and Becker 2018 lithosphere (mean_no_slabs model), with half-space cooling profile\n")
    f.write("# SLAB2 slab structure, half-space cooling temperature profile\n")
    f.write("# Non-uniform mesh - 0.2 degrees where slabs, 1 degree elsewhere\n")
    f.write("# 10 km depth resolution in UM, 150 km in LM\n")

    
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
                f.write(f"%.0f %.3f %.3f %.1f \n" % (r[i,j,k], phi[i,j,k], theta[i,j,k],temp[i,j,k]))
                        
T2=time.time()
T3=(T2-T1)
print(T3)
