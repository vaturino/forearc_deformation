#!/usr/bin/env python

import numpy as np
import os
import netCDF4 as nc
import glob
import math
from scipy.interpolate import interpn
from numba import jit
import time

@jit(nopython=True)
def calculate_slabs(lon_t, lat_t, depth_t, glon, glat, gdepth, dx, dz):
    slab = []
    for j in range(len(lon_t)):
        ix = [idx for idx, val in enumerate(glon) if lon_t[j] >= np.round((val - (dx / 2)), 2) and lon_t[j] < np.round((val + (dx / 2)), 2)]
        iy = [idx for idx, val in enumerate(glat) if lat_t[j] >= np.round((val - (dx / 2)), 2) and lat_t[j] < np.round((val + (dx / 2)), 2)]
        iz = [idx for idx, val in enumerate(gdepth) if depth_t[j] >= np.round((val - (dz / 2)), 2) and depth_t[j] < np.round((val + (dz / 2)), 2)]

        dnorm = 1 - (j / (len(lon_t) - 1))
        if len(ix) == 1 and len(iy) == 1 and len(iz) == 1:
            slab.append((ix, iy, iz, dnorm))

    return slab

@jit(nopython=True)
def calculate_composi(lon_t, glon, glat, gdepth, lat_t, depth_t, dx, dz):
    composi = []
    for j in range(len(lon_t)):
        ix = [idx for idx, val in enumerate(glon) if lon_t[j] >= np.round((val - (dx / 2)), 2) and lon_t[j] < np.round((val + (dx / 2)), 2)]
        iy = [idx for idx, val in enumerate(glat) if lat_t[j] >= np.round((val - (dx / 2)), 2) and lat_t[j] < np.round((val + (dx / 2)), 2)]
        iz = [idx for idx, val in enumerate(gdepth) if depth_t[j] >= np.round((val - (dz / 2)), 2) and depth_t[j] < np.round((val + (dz / 2)), 2)]

        if len(ix) == 1 and len(iy) == 1 and len(iz) == 1:
            composi.append((ix, iy, iz))
    return composi

@jit(nopython=True)
def calculate_lithosphere(lon_t, lat_t, depth_t, glon, glat, gdepth, dx, dz, litho_depth):
    lithosphere = []
    for j in range(len(lon_t)):
        ix = [idx for idx, val in enumerate(glon) if lon_t[j] >= np.round((val - (dx / 2)), 2) and lon_t[j] < np.round((val + (dx / 2)), 2)]
        iy = [idx for idx, val in enumerate(glat) if lat_t[j] >= np.round((val - (dx / 2)), 2) and lat_t[j] < np.round((val + (dx / 2)), 2)]
        iz = [idx for idx, val in enumerate(gdepth) if depth_t[j] >= np.round((val - (dz / 2)), 2) and depth_t[j] < np.round((val + (dz / 2)), 2)]

        # Check if depth is within the lithosphere range
        if len(ix) == 1 and len(iy) == 1 and len(iz) == 1:
            if 0 <= gdepth[iz[0]] <= litho_depth[iy[0], ix[0]]:  # Depth is within the lithosphere
                lithosphere.append((ix, iy, iz))
    return lithosphere




##############################################
T1 = time.time()

# Define grid resolution to match the global script
dx = 0.2  # Longitude grid resolution
dz = 10   # Depth grid resolution
litho_depth = 100  # Lithosphere depth threshold (100 km)

# Define regional grid setup
glon = np.linspace(250, 320, int((320 - 250) / dx) + 1, endpoint=True, dtype=np.float32)  # Longitude from 100 to 150
glat = np.linspace(-60, 20, int((20 + 60) / dx) + 1, endpoint=True, dtype=np.float32)      # Latitude from 10 to 30
gdepth = np.linspace(0, 2900, 291, endpoint=True, dtype=np.float32)  # Depth from 0 to 2900 km

# Define parameters
temp_ref = 1573  # Kelvin Mantle interior temperature
temp_surf = 273  # Surface temperature
W = 30  # km width of plate boundary zone

# Initialize arrays
slabs = np.zeros((len(glat), len(glon), len(gdepth)), dtype=bool)
slab_temp = np.full((len(glat), len(glon), len(gdepth)), np.nan, dtype=np.float32)
Composi = np.full((len(glat), len(glon), len(gdepth)), 0, dtype=np.uint8)
Litho = np.full((len(glat), len(glon), len(gdepth)), 0, dtype=np.uint8)  # Lithosphere array

# Read in and interpolate lithosphere thickness data
lith = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/mean_no_slabs.l.grd",'r')   # Lithosphere data input
lon_lith = lith.variables['lon'][:]
lat_lith = lith.variables['lat'][:]
lthick = lith.variables['z'][:]  # Lithosphere thickness data
glatm, glonm = np.meshgrid(glat, glon, indexing='ij')  # Regional grid for interpolation

# Interpolate lithosphere thickness to regional grid
new_lith = interpn((lat_lith, lon_lith), lthick, (glatm, glonm), method='linear')

# Calculate lithosphere depth for each grid point
lithosphere_depth = np.zeros_like(new_lith)
for i in range(len(glat)):
    for j in range(len(glon)):
        lithosphere_depth[i, j] = new_lith[i, j]

# Process the slab, compositional, and lithosphere data
directory = '/home/vturino/PhD/projects/forearc_deformation/slab2/sam/'
c = "dep"
pattern = os.path.join(directory, f"*{c}*grd")
fs = [os.path.basename(file) for file in glob.glob(pattern)]
name = "sam"

for filename in fs:
    slabname = filename.split('_')[0]
    print(slabname)
    fnhead = filename.split('dep')[0]
    fntail = filename.split('dep')[1]
    dep = nc.Dataset(f'{directory}/{filename}')
    thk = nc.Dataset(f'{directory}/{fnhead}thk{fntail}')
    str = nc.Dataset(f'{directory}/{fnhead}str{fntail}')
    dip = nc.Dataset(f'{directory}/{fnhead}dip{fntail}')

    lon = dep.variables['x'][:]
    lat = dep.variables['y'][:]
    dep = -dep.variables['z'][:]
    thk = thk.variables['z'][:]
    str = str.variables['z'][:]
    dip = dip.variables['z'][:]

    lat, lon = np.meshgrid(lat, lon, indexing='ij')

    m = ~np.isnan(dep)

    lon = lon[m]
    lat = lat[m]
    thk = thk[m]
    str = str[m]
    dip = dip[m]
    dep = dep[m]

    for i in range(len(lon)):
        Dx = (2 * math.pi * (6371 - dep[i]) * math.cos(math.radians(lat[i]))) / 360
        Dy = (2 * math.pi * (6371 - dep[i])) / 360

        normx = math.cos(math.radians(str[i])) * math.sin(math.radians(dip[i]))
        normy = -math.sin(math.radians(str[i])) * math.sin(math.radians(dip[i]))
        normz = math.cos(math.radians(dip[i]))

        lon_t = np.linspace(lon[i] - normx * thk[i] / Dx, lon[i], 25, endpoint=True)
        lat_t = np.linspace(lat[i] - normy * thk[i] / Dy, lat[i], 25, endpoint=True)
        depth_t = np.linspace(dep[i] + normz * thk[i], dep[i], 25, endpoint=True)

        lon_t = lon_t[depth_t >= 0]
        lat_t = lat_t[depth_t >= 0]
        depth_t = depth_t[depth_t >= 0]

        slabindex = calculate_slabs(lon_t, lat_t, depth_t, glon, glat, gdepth, dx, dz)
        for indices in slabindex:
            ix, iy, iz, dnorm = indices
            slabs[iy, ix, iz] = 1

        lon_t = np.linspace(lon[i] - normx * (-W) / Dx, lon[i], 10, endpoint=True)
        lat_t = np.linspace(lat[i] - normy * (-W) / Dy, lat[i], 10, endpoint=True)
        depth_t = np.linspace(dep[i] + normz * (-W), dep[i], 10, endpoint=True)

        lon_t = lon_t[depth_t >= 0]
        lat_t = lat_t[depth_t >= 0]
        depth_t = depth_t[depth_t >= 0]

        composi_index = calculate_composi(lon_t, glon, glat, gdepth, lat_t, depth_t, dx, dz)
        for indices in composi_index:
            ix, iy, iz = indices
            Composi[iy, ix, iz] = 2

        # Use lithosphere depth from interpolated data
        litho_index = calculate_lithosphere(lon_t, lat_t, depth_t, glon, glat, gdepth, dx, dz, lithosphere_depth)
        for indices in litho_index:
            ix, iy, iz = indices
            Litho[iy, ix, iz] = 3  # Assign lithosphere as 3

    # Save the data to a NetCDF file
    ds = nc.Dataset('../slab_geometries/sam_geometry_with_litho.nc', 'w', format="NETCDF4")
    glon1 = ds.createDimension('glon1', len(glon))
    glat1 = ds.createDimension('glat1', len(glat))
    gdepth1 = ds.createDimension('gdepth1', len(gdepth))

    glon1 = ds.createVariable('glon1', np.float32, ('glon1',))
    glon1[:] = glon

    glat1 = ds.createVariable('glat1', np.float32, ('glat1',))
    glat1[:] = glat

    gdepth1 = ds.createVariable('gdepth1', np.float32, ('gdepth1',))
    gdepth1[:] = gdepth

    slab1 = ds.createVariable('slabs1', 'i1', ('glat1', 'glon1', 'gdepth1'))
    slab1[:] = slabs

    Composi1 = ds.createVariable('Composi1', np.uint8, ('glat1', 'glon1', 'gdepth1'))
    Composi1[:] = Composi

    Litho1 = ds.createVariable('Litho1', np.uint8, ('glat1', 'glon1', 'gdepth1'))
    Litho1[:] = Litho

    slab_temp1 = ds.createVariable('slab_temp1', np.float32, ('glat1', 'glon1', 'gdepth1'))
    slab_temp1[:] = slab_temp

    ds.close()

T2 = time.time()
T3 = (T2 - T1)
print(T3)
