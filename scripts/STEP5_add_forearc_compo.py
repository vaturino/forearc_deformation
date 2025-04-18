#!/usr/bin/env python3

import netCDF4 as nc
import numpy as np
from scipy.interpolate import interp1d, interpn,griddata
from shapely.geometry import Point, Polygon
import pandas as pd
import time
import scipy.io
from scipy.ndimage import generate_binary_structure, binary_dilation, binary_erosion
from scipy.ndimage import label
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


min_lat = -60
max_lat = 20
min_lon = 240
max_lon = 320

T1 = time.time()
print(T1)

# Make gridded vertical plate boundaries in 2-d from Bird2003 dataset
# specify parameters
W = 15  # half-width of plate boundaries zone (km)
dx = 0.1  # grid spacing (degree)   # keep same spacing with slab geometry
depth_limit = 100  # Max depth for Bound in km

# Initialize the gridded arrays with appropriate longitude and latitude ranges
glon1 = np.linspace(min_lon, max_lon, int((max_lon - min_lon) / dx) + 1, endpoint=True, dtype=np.float32)  # longitude range from min_lon to max_lon
glat1 = np.linspace(min_lat, max_lat, int((max_lat - min_lat) / dx) + 1, endpoint=True, dtype=np.float32)  # latitude range from min_lat to max_lat
Bound = np.zeros((len(glat1), len(glon1)), dtype=np.uint8)
Slab = np.zeros((len(glat1), len(glon1)), dtype=np.uint8)
Crust = np.zeros((len(glat1), len(glon1)), dtype=np.uint8)
OP = np.zeros((len(glat1), len(glon1)), dtype=np.uint8)

# Add in slab plate boundaries
slab2_file = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry_continents.nc", 'r')
depth = slab2_file.variables['gdepth_var'][:]
slab_C = slab2_file.variables['C_crust'][:]  # Crust (Composi1) from sam_geometry
slab = slab2_file.variables['C_slab'][:]  # Slab (slabs1) from sam_geometry
plbd = slab2_file.variables['C_bound'][:]
op = slab2_file.variables['C_OP'][:]
slab2_file.close()

# Broadcast the data
Slab = np.broadcast_to(slab, (len(glat1), len(glon1), slab.shape[-1]))  # Keep slab's last dimension
Crust = np.broadcast_to(slab_C, (len(glat1), len(glon1), slab_C.shape[-1]))  # Match depth dimension
Bound = np.broadcast_to(plbd, (len(glat1), len(glon1), plbd.shape[-1]))  # Match depth dimension
OP = np.broadcast_to(op, (len(glat1), len(glon1), op.shape[-1]))  # Match depth dimension
C_farc = np.zeros_like(OP)  # Forearc regions

# Initialize compositional arrays
C_bound = np.zeros_like(Bound)  # Bound compositional field
C_slab = np.zeros_like(Slab)    # Slab compositional field
C_crust = np.zeros_like(Crust)  # Crust compositional field
C_op = np.zeros_like(OP)        # OP compositional field

C_bound[Bound != 0] = 1  # Assign Bound regions
C_slab[Slab != 0] = 2   # Assign Slab regions (from slabs1)
C_crust[Crust != 0] = 3  # Assign Crust regions (from Composi1)
C_op[OP != 0] = 4        # Assign OP regions (from Composi1)


# Loop over latitude
for i_lat in range(C_op.shape[0]):  # 801 latitudes
    # Get the indices (j) where OP exists at surface (depth = 0)
    op_indices = np.where(C_op[i_lat, :, 0] == 4)[0]
    
    if op_indices.size > 0:
        lon_op = glon1[op_indices]
        min_lon = lon_op.min()
        max_lon = min_lon + 2.0  # 2 degrees to the east

        # Get longitude indices within that range
        farc_indices = np.where((glon1 >= min_lon) & (glon1 <= max_lon))[0]

        # For all depths, assign composition 5 where C_op is non-zero
        for j in farc_indices:
            farc_mask = C_op[i_lat, j, :] != 0
            C_farc[i_lat, j, farc_mask] = 5

# just values below lat = 13
C_farc[glat1 >= 13, :, :] = 0
C_farc[glat1 < -51.5, :, :] = 0

# Define the latitude and longitude masks for the specified ranges
mask_lat = np.logical_and(glat1 > -46, glat1 < -45)
mask_lon = np.logical_and(glon1 > 286.3, glon1 < 287.7)

# Assign composition 5 for the specified latitude and longitude ranges
C_farc[np.ix_(mask_lat, mask_lon, np.arange(C_farc.shape[2]))] = 5


C_op[C_farc != 0] = 0  # Assign composition 5 to OP where C_farc is non-zero

C = C_bound + C_slab + C_crust + C_op + C_farc  # Combine all compositional fields




gdepth = depth

# Save compositional fields as separate arrays
ds = nc.Dataset('/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry_continents_farc.nc', 'w', format="NETCDF4")
glon_di = ds.createDimension('glon_di', len(glon1))
glat_di = ds.createDimension('glat_di', len(glat1))
gdepth_di = ds.createDimension('gdepth_di', len(depth))

glon_var = ds.createVariable('glon_var', np.float32, ('glon_di',))
glon_var[:] = glon1

glat_var = ds.createVariable('glat_var', np.float32, ('glat_di',))
glat_var[:] = glat1

gdepth_var = ds.createVariable('gdepth_var', np.float32, ('gdepth_di',))
gdepth_var[:] = gdepth

C_bound_var = ds.createVariable('C_bound', 'i1', ('glat_di', 'glon_di', 'gdepth_di'))
C_bound_var[:] = C_bound

C_slab_var = ds.createVariable('C_slab', 'i1', ('glat_di', 'glon_di', 'gdepth_di'))
C_slab_var[:] = C_slab

C_crust_var = ds.createVariable('C_crust', 'i1', ('glat_di', 'glon_di', 'gdepth_di'))
C_crust_var[:] = C_crust

C_OP_var = ds.createVariable('C_OP', 'i1', ('glat_di', 'glon_di', 'gdepth_di'))
C_OP_var[:] = C_op

C_farc_var = ds.createVariable('C_farc', 'i1', ('glat_di', 'glon_di', 'gdepth_di'))
C_farc_var[:] = C_farc

# continents_var = ds.createVariable('continents', 'i1', ('glat_di', 'glon_di', 'gdepth_di'))
# continents_var[:] = continents

ds.close()

print("saved nc file")

fname = '/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry_continents_farc.txt'
glon = np.linspace(min_lon, max_lon, int((max_lon - min_lon) / dx) + 1, endpoint=True, dtype=np.float32)
glat = np.linspace(min_lat, max_lat, int((max_lat - min_lat) / dx) + 1, endpoint=True, dtype=np.float32)
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
    fid.write('# Columns: r phi theta plbd crust slab op farc\n')
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
                fid.write("%.0f %.4f %.4f %i %i %i %i %i \n" % (
                    r[i,j,k], phi[i,j,k], theta[i,j,k],
                    C_bound[i,j,k], C_crust[i,j,k], C_slab[i,j,k], C_op[i,j,k], C_farc[i,j,k]
                ))
         
T2=time.time()
print(T2-T1)