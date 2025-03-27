#!/usr/bin/env python3

import numpy as np
import os
import netCDF4 as nc
import glob
import math
from scipy.special import erfc
from numba import jit
import time

# Define numba function to track slab, lithosphere, and weak zone positions
@jit(nopython=True)
def calculate_positions(lon_t, lat_t, depth_t, glon, glat, gdepth, dx, dz):
    positions = []
    for j in range(len(lon_t)):
        ix = [idx for idx, val in enumerate(glon) if lon_t[j] >= np.round((val - (dx/2)), 2) and lon_t[j] < np.round((val + (dx/2)), 2)]
        iy = [idx for idx, val in enumerate(glat) if lat_t[j] >= np.round((val - (dx/2)), 2) and lat_t[j] < np.round((val + (dx/2)), 2)]
        iz = [idx for idx, val in enumerate(gdepth) if depth_t[j] >= np.round((val - (dz/2)), 2) and depth_t[j] < np.round((val + (dz/2)), 2)]

        if len(ix) == 1 and len(iy) == 1 and len(iz) == 1:
            positions.append((ix, iy, iz))
    return positions

# Set up grids
dx, dz = 0.2, 10
glon = np.linspace(0, 360, 1801, endpoint=True, dtype=np.float32)
glat = np.linspace(-90, 90, 901, endpoint=True, dtype=np.float32)
gdepth = np.linspace(0, 1100, 111, endpoint=True, dtype=np.float32)

# Parameters
temp_ref = 1573  # Mantle interior temperature (Kelvin)
temp_surf = 273  # Surface temperature (Kelvin)
W = 30  # Weak zone width (km)
lith_thick = 100  # Lithosphere thickness (km)

# Initialize arrays
slabs = np.zeros((len(glat), len(glon), len(gdepth)), dtype=bool)
lithosphere = np.zeros((len(glat), len(glon), len(gdepth)), dtype=bool)
weak_zone = np.full((len(glat), len(glon), len(gdepth)), 0, dtype=np.uint8)
temperature = np.full((len(glat), len(glon), len(gdepth)), np.nan, dtype=np.float32)

# Read and process slab data
directory = '/home/vturino/PhD/projects/forearc_deformation/slab2/sam/'
c = "dep"
pattern = os.path.join(directory, f"*{c}*grd")
fs = [os.path.basename(file) for file in glob.glob(pattern)]
name = "sam"

for filename in fs:
    slabname = filename.split('_')[0]
    print(f"Processing {slabname}...")
    
    fnhead, fntail = filename.split('dep')[0], filename.split('dep')[1]
    dep = nc.Dataset(f'{directory}/{filename}').variables['z'][:]
    thk = nc.Dataset(f'{directory}/{fnhead}thk{fntail}').variables['z'][:]
    str = nc.Dataset(f'{directory}/{fnhead}str{fntail}').variables['z'][:]
    dip = nc.Dataset(f'{directory}/{fnhead}dip{fntail}').variables['z'][:]

    lon = nc.Dataset(f'{directory}/{filename}').variables['x'][:]
    lat = nc.Dataset(f'{directory}/{filename}').variables['y'][:]
    lat, lon = np.meshgrid(lat, lon, indexing='ij')
    
    dep, thk, str, dip = -dep, thk, str, dip
    mask = ~np.isnan(dep)
    lon, lat, dep, thk, str, dip = lon[mask], lat[mask], dep[mask], thk[mask], str[mask], dip[mask]

    for i in range(len(lon)):
        Dx = 2 * np.pi * (6371 - dep[i]) * np.cos(np.radians(lat[i])) / 360
        Dy = 2 * np.pi * (6371 - dep[i]) / 360
        normx = np.cos(np.radians(str[i])) * np.sin(np.radians(dip[i]))
        normy = -np.sin(np.radians(str[i])) * np.sin(np.radians(dip[i]))
        normz = np.cos(np.radians(dip[i]))

        # Compute slab position
        lon_t = np.linspace(lon[i] - normx * thk[i] / Dx, lon[i], 25)
        lat_t = np.linspace(lat[i] - normy * thk[i] / Dy, lat[i], 25)
        depth_t = np.linspace(dep[i] + normz * thk[i], dep[i], 25)

        slab_positions = calculate_positions(lon_t, lat_t, depth_t, glon, glat, gdepth, dx, dz)
        for ix, iy, iz in slab_positions:
            slabs[iy, ix, iz] = 1

        # Compute lithosphere position
        lith_depth = dep[i] - lith_thick
        lon_t = np.linspace(lon[i], lon[i], 10)
        lat_t = np.linspace(lat[i], lat[i], 10)
        depth_t = np.linspace(lith_depth, dep[i], 10)

        lith_positions = calculate_positions(lon_t, lat_t, depth_t, glon, glat, gdepth, dx, dz)
        for ix, iy, iz in lith_positions:
            lithosphere[iy, ix, iz] = 1

        # Compute weak zone position (above both slab and lithosphere)
        lon_t = np.linspace(lon[i], lon[i] - normx * W / Dx, 10)
        lat_t = np.linspace(lat[i], lat[i] - normy * W / Dy, 10)
        depth_t = np.linspace(dep[i], dep[i] + normz * W, 10)

        weak_positions = calculate_positions(lon_t, lat_t, depth_t, glon, glat, gdepth, dx, dz)
        for ix, iy, iz in weak_positions:
            weak_zone[iy, ix, iz] = 2

        # Compute temperature from weak zone surface to slab/lithosphere
        for iz in range(len(gdepth)):
            if slabs[iy, ix, iz] or lithosphere[iy, ix, iz]:
                depth_ratio = (gdepth[iz] - dep[i]) / (lith_depth - dep[i])
                temperature[iy, ix, iz] = temp_ref + (temp_surf - temp_ref) * erfc(1.16 * depth_ratio)

# Save results to NetCDF
ds = nc.Dataset(f'../slab_geometries/{name}_geometry.nc', 'w', format="NETCDF4")
ds.createDimension('glon', len(glon))
ds.createDimension('glat', len(glat))
ds.createDimension('gdepth', len(gdepth))

glon1 = ds.createVariable('glon',np.float32,('glon',))
glon1[:] = glon

glat1 = ds.createVariable('glat', np.float32,('glat',))
glat1[:] = glat

gdepth1 = ds.createVariable('gdepth', np.float32 ,('gdepth',))
gdepth1[:] = gdepth

ds.createVariable('slabs', 'i1', ('glat', 'glon', 'gdepth'))[:] = slabs
ds.createVariable('lithosphere', 'i1', ('glat', 'glon', 'gdepth'))[:] = lithosphere
ds.createVariable('weak_zone', np.uint8 , ('glat', 'glon', 'gdepth'))[:] = weak_zone
ds.createVariable('temperature', np.float32, ('glat', 'glon', 'gdepth'))[:] = temperature

ds.close()

print("Processing complete!")
