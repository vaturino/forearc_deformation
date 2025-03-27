#!/usr/bin/env python3

import numpy as np
import netCDF4 as nc
from scipy.special import erfc

# Load positions from the first script
ds = nc.Dataset('../slab_geometries/sam_positions.nc', 'r')

glon = ds.variables['glon'][:]
glat = ds.variables['glat'][:]
gdepth = ds.variables['gdepth'][:]

slabs = ds.variables['slabs'][:]
lithosphere = ds.variables['lithosphere'][:]
weak_zone = ds.variables['weak_zone'][:]

ds.close()

# Parameters
temp_ref = 1573  # Mantle interior temperature (Kelvin)
temp_surf = 273  # Surface temperature (Kelvin)

# Initialize temperature array
temperature = np.full((len(glat), len(glon), len(gdepth)), np.nan, dtype=np.float32)

# Calculate temperature for the regions
for iy in range(len(glat)):
    for ix in range(len(glon)):
        for iz in range(len(gdepth)):
            if slabs[iy, ix, iz] or lithosphere[iy, ix, iz]:
                # Half-space cooling law temperature calculation
                depth_ratio = gdepth[iz]  # Depth is in km
                temperature[iy, ix, iz] = temp_surf + (temp_ref - temp_surf) * erfc(depth_ratio / (2 * np.sqrt(2 * 1e6)))

# Save results to NetCDF
ds = nc.Dataset('../slab_geometries/sam_geometry_with_temperature.nc', 'w', format="NETCDF4")
ds.createDimension('glon', len(glon))
ds.createDimension('glat', len(glat))
ds.createDimension('gdepth', len(gdepth))

glon1 = ds.createVariable('glon', np.float32, ('glon',))
glon1[:] = glon

glat1 = ds.createVariable('glat', np.float32, ('glat',))
glat1[:] = glat

gdepth1 = ds.createVariable('gdepth', np.float32, ('gdepth',))
gdepth1[:] = gdepth

ds.createVariable('slabs', 'i1', ('glat', 'glon', 'gdepth'))[:] = slabs
ds.createVariable('lithosphere', 'i1', ('glat', 'glon', 'gdepth'))[:] = lithosphere
ds.createVariable('weak_zone', np.uint8, ('glat', 'glon', 'gdepth'))[:] = weak_zone
ds.createVariable('temperature', np.float32, ('glat', 'glon', 'gdepth'))[:] = temperature

ds.close()

print("Temperature calculation complete!")
