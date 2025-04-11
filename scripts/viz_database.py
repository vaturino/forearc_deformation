#! /usr/bin/env python

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

# Load LithoRef 
lith = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/LithoRef_model/lithospheric-thickness-griddata.nc", "r")

# Extract data
lon = (lith.variables["lon"][:] + 360) % 360  # Wrap longitudes to [0, 360]
lat = lith.variables["lat"][:]
thick = lith.variables["z"][:]

# Remove duplicate longitudes and sort them
lon_sorted, lon_unique_idx = np.unique(lon, return_index=True)
thick_unique_lon = thick[:, lon_unique_idx]  # Keep only unique longitude columns

# Sort latitudes and reorder thickness data accordingly
lat_sorted = np.sort(lat)
lat_sort_idx = np.argsort(lat)
thick_sorted = thick_unique_lon[lat_sort_idx, :]

# Ensure latitudes and longitudes are strictly increasing
assert np.all(np.diff(lon_sorted) > 0), "Longitude values are not strictly ascending"
assert np.all(np.diff(lat_sorted) > 0), "Latitude values are not strictly ascending"

# Interpolation using RegularGridInterpolator
interp_func = RegularGridInterpolator(
    (lat_sorted, lon_sorted), thick_sorted, bounds_error=False, fill_value=np.nan
)

# Define the grid with 0.2 degree resolution
grid_lon = np.arange(0, 360, 0.2)  # 0.2-degree resolution for longitude
grid_lat = np.arange(-90, 90, 0.2)  # 0.2-degree resolution for latitude
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# Interpolate the lithospheric thickness data onto this grid
grid_points = np.column_stack((grid_lat.ravel(), grid_lon.ravel()))
thick_grid = interp_func(grid_points).reshape(grid_lat.shape)

# Apply a Gaussian filter to smooth the data (optional)
smoothed_thick_grid = gaussian_filter(thick_grid, sigma=10)

# Increase the figure dpi to improve the resolution of the plot
plt.figure(figsize=(12, 8))
plt.imshow(smoothed_thick_grid, extent=(grid_lon.min(), grid_lon.max(), grid_lat.min(), grid_lat.max()), origin='lower', interpolation='bicubic')
plt.colorbar(label='Lithospheric Thickness (km)')
plt.title('Lithospheric Thickness at 0.2° Resolution (Smoothed)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
