#!/usr/bin/env python

import numpy as np
import os
import netCDF4 as nc
import glob
import math
from scipy.special import erfc
from numba import jit
import time
import scipy.ndimage as ndimage
# Function to check if a point is inside the high-resolution region
def is_high_resolution(lon, lat, depth):
    return (min_lat_high <= lat <= max_lat_high) and \
           (min_lon_high <= lon <= max_lon_high) and \
           (min_depth_high <= depth <= max_depth_high)



@jit(nopython=True)
def calculate_slabs(lon_t, lat_t, depth_t, glon, glat, gdepth, dx, dz):
    slab = []
    
    # Loop over all target points
    for j in range(len(lon_t)):
        # Find the index of the closest point for longitude, latitude, and depth
        ix = np.abs(glon - lon_t[j]).argmin()  # Find the index of the closest longitude
        iy = np.abs(glat - lat_t[j]).argmin()  # Find the index of the closest latitude
        iz = np.abs(gdepth - depth_t[j]).argmin()  # Find the index of the closest depth

        # Check if the point is within the allowed tolerance in each direction
        if (np.abs(glon[ix] - lon_t[j]) <= dx / 2 and
            np.abs(glat[iy] - lat_t[j]) <= dx / 2 and
            np.abs(gdepth[iz] - depth_t[j]) <= dz / 2):
            
            # Normalize based on the index (you can modify this logic as needed)
            dnorm = 1 - (j / (len(lon_t) - 1))
            
            # Append the result as a tuple of indices and normalization factor
            slab.append((ix, iy, iz, dnorm))
    
    return slab

@jit(nopython=True)
def calculate_composi(lon_t, lat_t, depth_t, glon, glat, gdepth, dx, dz):
    composi = []
    # Loop over all target points
    for j in range(len(lon_t)):
        # Find the index of the closest point for longitude, latitude, and depth
        ix = np.abs(glon - lon_t[j]).argmin()  # Find the index of the closest longitude
        iy = np.abs(glat - lat_t[j]).argmin()  # Find the index of the closest latitude
        iz = np.abs(gdepth - depth_t[j]).argmin()  # Find the index of the closest depth

        # Check if the point is within the allowed tolerance in each direction
        if (np.abs(glon[ix] - lon_t[j]) <= dx / 2 and
            np.abs(glat[iy] - lat_t[j]) <= dx / 2 and
            np.abs(gdepth[iz] - depth_t[j]) <= dz / 2):
            
            # Normalize based on the index (you can modify this logic as needed)
            dnorm = 1 - (j / (len(lon_t) - 1))
            
            # Append the result as a tuple of indices and normalization factor
            composi.append((ix, iy, iz))
    
    return composi



##############################################
T1 = time.time()


# Define grid resolution to match the global script
dx = 0.2  # Longitude grid resolution
dz = 10   # Depth grid resolution
dx_high = 0.1  # refined horizontal resolution
dz_high = 5    # refined vertical resolution

# Domain limits
min_lat = -60
max_lat = 20
min_lon = 240
max_lon = 320
max_depth = 2900
# High-resolution domain limits
min_lat_high = -50
max_lat_high = 10
min_lon_high = 277
max_lon_high = 300
min_depth_high = 0
max_depth_high = 700

# Non uniform latitude
lat_low_1 = np.arange(min_lat, min_lat_high, dx)
lat_high = np.arange(min_lat_high, max_lat_high, dx_high)
lat_low_2 = np.arange(max_lat_high, max_lat + dx, dx)  # Include max_lat
glat = np.unique(np.concatenate((lat_low_1, lat_high, lat_low_2)))
np.sort(glat)

# Non uniform longitude
lon_low_1 = np.arange(min_lon, min_lon_high, dx)
lon_high = np.arange(min_lon_high, max_lon_high, dx_high)
lon_low_2 = np.arange(max_lon_high, max_lon + dx, dx)  # Include max_lon
glon = np.unique(np.concatenate((lon_low_1, lon_high, lon_low_2)))
np.sort(glon)

# Non uniform depth
depth_low_1 = np.arange(max_depth_high, max_depth, dz)
depth_high = np.arange(min_depth_high, max_depth_high, dz_high)
gdepth = np.unique(np.concatenate((depth_low_1, depth_high)))
gdepth = np.sort(gdepth)

# glonm, glatm, gdepthm = np.meshgrid(glon, glat, gdepth, indexing='ij')


# import matplotlib.pyplot as plt

# # Select mid-points (you can adjust these)
# depth_idx = 125  # Middle of depth (out of 250)
# lat_idx = 320    # Middle of lat (out of 641)
# lon_idx = 172    # Middle of lon (out of 345)

# # 1. Lon–Lat @ fixed depth
# lat_slice = glatm[:, :, depth_idx]
# lon_slice = glonm[:, :, depth_idx]

# # 2. Lat–Depth @ fixed lon
# lat_depth = glatm[:, lon_idx, :]
# depth_slice_lat = gdepthm[:, lon_idx, :]

# # 3. Lon–Depth @ fixed lat
# lon_depth = glonm[lat_idx, :, :]
# depth_slice_lon = gdepthm[lat_idx, :, :]

# # Plotting
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# # 1. Lon–Lat slice (horizontal)
# axs[0].scatter(lon_slice, lat_slice, s=0.1, color='black')
# axs[0].set_title(f"Lon–Lat @ depth = {gdepthm[0, 0, depth_idx]:.1f} km")
# axs[0].set_xlabel("Longitude")
# axs[0].set_ylabel("Latitude")
# axs[0].axis('equal')
# axs[0].grid(True)

# # 2. Lat–Depth slice (vertical, fixed lon)
# axs[1].scatter(lat_depth, -depth_slice_lat, s=0.1, color='blue')
# axs[1].set_title(f"Lat–Depth @ lon = {glonm[0, lon_idx, 0]:.2f}°")
# axs[1].set_xlabel("Latitude")
# axs[1].set_ylabel("Depth (km)")
# axs[1].grid(True)

# # 3. Lon–Depth slice (vertical, fixed lat)
# axs[2].scatter(lon_depth, -depth_slice_lon, s=0.1, color='red')
# axs[2].set_title(f"Lon–Depth @ lat = {glatm[lat_idx, 0, 0]:.2f}°")
# axs[2].set_xlabel("Longitude")
# axs[2].set_ylabel("Depth (km)")
# axs[2].grid(True)

# plt.tight_layout()
# plt.show()
# exit()



# Define parameters
temp_ref = 1573  # Kelvin Mantle interior temperature
temp_surf = 273  # Surface temperature
crust_thickness = 15  # km crust thickness

# Initialize arrays
slabs = np.zeros((len(glat), len(glon), len(gdepth)), dtype=bool)
slab_temp = np.full((len(glat), len(glon), len(gdepth)), np.nan, dtype=np.float32)
Composi = np.full((len(glat), len(glon), len(gdepth)), 0, dtype=np.uint8)


# Read in and iterate slab-by-slab
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

        ddx = 0.5
        ddz = 0.5

         # Check if the current lat, lon, and depth are inside the high-resolution region
        if is_high_resolution(lon[i], lat[i], dep[i]):
            # Use high-res spacing
            ddx = dx_high
            ddz = dz_high
        else:
            # Use default resolution
            ddx = dx
            ddz = dz

        normx = math.cos(math.radians(str[i])) * math.sin(math.radians(dip[i]))
        normy = -math.sin(math.radians(str[i])) * math.sin(math.radians(dip[i]))
        normz = math.cos(math.radians(dip[i]))

        lon_t = np.linspace(lon[i] - normx * thk[i] / Dx, lon[i], 25, endpoint=True)
        lat_t = np.linspace(lat[i] - normy * thk[i] / Dy, lat[i], 25, endpoint=True)
        depth_t = np.linspace(dep[i] + normz * thk[i], dep[i], 25, endpoint=True)

        lon_t = lon_t[depth_t >= 0]
        lat_t = lat_t[depth_t >= 0]
        depth_t = depth_t[depth_t >= 0]

        slabindex = calculate_slabs(lon_t, lat_t, depth_t, glon, glat, gdepth, ddx, ddz)
        for indices in slabindex:
            ix, iy, iz, dnorm = indices
            slabs[iy, ix, iz] = 1

        lon_t = np.linspace(lon[i] - normx * (-crust_thickness) / Dx, lon[i], 10, endpoint=True)
        lat_t = np.linspace(lat[i] - normy * (-crust_thickness) / Dy, lat[i], 10, endpoint=True)
        depth_t = np.linspace(dep[i] + normz * (-crust_thickness), dep[i], 10, endpoint=True)

        lon_t = lon_t[depth_t >= 0]
        lat_t = lat_t[depth_t >= 0]
        depth_t = depth_t[depth_t >= 0]

        composi_index = calculate_composi(lon_t, lat_t, depth_t, glon, glat, gdepth, ddx, ddz)
        for indices in composi_index:
            ix, iy, iz = indices
            Composi[iy, ix, iz] = 2

        # Ensure proper depth ranges for the temperature calculation
        if composi_index and slabindex:
            top_composi_depth = depth_t[0]
            bottom_slab_depth = depth_t[-1]
            for indices in composi_index:
                ix, iy, iz = indices
                dnorm = (gdepth[iz] - top_composi_depth) / (bottom_slab_depth - top_composi_depth)
                # Avoid division by zero or NaN issues
                if not np.isnan(dnorm) and bottom_slab_depth != top_composi_depth:
                    slab_temp[iy, ix, iz] = temp_ref + (temp_surf - temp_ref) * erfc(1.16 * dnorm)

    # Update temperature for the entire slab and compositional region
    for iy in range(len(glat)):
        for ix in range(len(glon)):
            slab_indices = np.where(slabs[iy, ix, :] == 1)[0]
            composi_indices = np.where(Composi[iy, ix, :] == 2)[0]
            if len(slab_indices) > 0 and len(composi_indices) > 0:
                top_composi_depth = gdepth[composi_indices[0]]
                bottom_slab_depth = gdepth[slab_indices[-1]]
                for iz in range(composi_indices[0], slab_indices[-1] + 1):
                    if iz >= 0 and iz < len(gdepth):  # Ensure index is within the valid depth range
                        dnorm = (gdepth[iz] - top_composi_depth) / (bottom_slab_depth - top_composi_depth)
                        if not np.isnan(dnorm) and bottom_slab_depth != top_composi_depth:
                            slab_temp[iy, ix, iz] = temp_ref + (temp_surf - temp_ref) * erfc(1.16 * dnorm)







# slabs = ndimage.binary_propagation(slabs, structure=np.ones((3, 3, 3)), border_value=1)
# slabs = slabs.astype(np.uint8)
# Composi = ndimage.binary_propagation(Composi, structure=np.ones((3, 3, 3)), border_value=1)
# Composi = Composi.astype(np.uint8)
# Composi[slabs ==1] = 0
# Composi[Composi ==1] = 2


# Save the data to a NetCDF file
ds = nc.Dataset('../slab_geometries/sam_geometry.nc', 'w', format="NETCDF4")
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

slab_temp1 = ds.createVariable('slab_temp1', np.float32, ('glat1', 'glon1', 'gdepth1'))
slab_temp1[:] = slab_temp

ds.close()

T2 = time.time()
T3 = (T2 - T1)
print(T3)


