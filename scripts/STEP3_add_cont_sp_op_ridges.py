#!/usr/bin/env python3

import netCDF4 as nc
import numpy as np
from scipy.interpolate import interp1d, interpn
import time
import scipy.io
from scipy.ndimage import generate_binary_structure, binary_dilation
from scipy.ndimage import label
import matplotlib.pyplot as plt


min_lat = -60
max_lat = 20
min_lon = 220
max_lon = 320

T1 = time.time()
print(T1)

# Make gridded vertical plate boundaries in 2-d from Bird2003 dataset
# specify parameters
W = 15  # half-width of plate boundaries zone (km)
dx = 0.2  # grid spacing (degree)   # keep same spacing with slab geometry
depth_limit = 50  # Max depth for Bound in km

# Read in Bird-2003 plate boundaries --- collected by Sam
data = scipy.io.loadmat('/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/plbd_0_360.mat')
lon = data['lon'][:, -1]  # 6870x1
lat = data['lat'][:, -1]  # 6870x1  

# Apply longitude and latitude cut (250° to 320° and -60° to 20°)
lon_mask = (lon >= min_lon) & (lon <= max_lon)  # Mask lon between min_lon and max_lon
lat_mask = (lat >= min_lat) & (lat <= max_lat)  # Mask lat between min_lat and max_lat

# Apply mask to lon and lat to keep only data within the desired region
lon_filtered = lon[lon_mask]
lat_filtered = lat[lat_mask]

print("Filtered lon range:", lon_filtered.min(), lon_filtered.max())
print("Filtered lat range:", lat_filtered.min(), lat_filtered.max())

# Interpolate lon and lat for finer resolution
new_lon = np.linspace(min_lon, max_lon, 30240)  # Desired range from min_lon to max_lon with 30240 points
new_lat = np.linspace(min_lat, max_lat, 30240)  # Desired range from min_lat to max_lat with 30240 points


# Interpolate longitude and latitude separately (preserving the range)
lon_interp_func = interp1d(lon_filtered, lon_filtered, kind='linear', fill_value="extrapolate", bounds_error=False)
lat_interp_func = interp1d(lat_filtered, lat_filtered, kind='linear', fill_value="extrapolate", bounds_error=False)

# Interpolating the filtered longitude and latitude data
lon_new = lon_interp_func(new_lon)
lat_new = lat_interp_func(new_lat)

# Initialize the gridded arrays with appropriate longitude and latitude ranges
glon1 = np.arange(min_lon, max_lon + dx, dx, dtype=np.float32)  # longitude range from min_lon to max_lon
glat1 = np.arange(min_lat, max_lat + dx, dx, dtype=np.float32)  # latitude range from min_lat to max_lat
Bound = np.zeros((len(glat1), len(glon1)), dtype=np.uint8)
Slab = np.zeros((len(glat1), len(glon1)), dtype=np.uint8)
Crust = np.zeros((len(glat1), len(glon1)), dtype=np.uint8)

glat, glon = np.meshgrid(glat1, glon1, indexing='ij')

dlat = 6371 * 1000 * (np.pi) / 180    
dlon = 6371 * 1000 * (np.pi) / 180 * np.cos(np.radians(lat_new))  

print("Created grid")

# Iterate through lon, lat and update Bound array
for i in range(len(lon)):
    d = np.sqrt(((lon[i] - glon) * dlon[i])**2 + ((lat[i] - glat) * dlat)**2) / 1000 
    Bound[d <= W] = 1  # Assign plate boundaries where the distance is within the boundary width

# Apply binary dilation to ensure continuity of plate boundaries (closing small gaps)
struct = generate_binary_structure(2, 2)  # Structuring element for dilation
Bound = binary_dilation(Bound, structure=struct, iterations=2).astype(np.uint8)  # Dilation to remove gaps

print("Added plate boundaries with dilation")

# Add in slab plate boundaries
slab2_file = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry.nc", 'r')
depth = slab2_file.variables['gdepth1'][:]
slab_C = slab2_file.variables['Composi1'][:]  # Crust (Composi1) from sam_geometry
slab = slab2_file.variables['slabs1'][:]  # Slab (slabs1) from sam_geometry
slab2_file.close()

# Extend gdepth as in the parent script by appending additional depth values
gdepth = np.concatenate((depth, [240, 250, 260, 270, 280, 290, 300, 2900]))  # Extension

print("read in slab geometry")

# Ensure correct broadcasting instead of tiling
Slab = np.broadcast_to(slab, (len(glat1), len(glon1), slab.shape[-1]))  # Keep slab's last dimension
Crust = np.broadcast_to(slab_C, (len(glat1), len(glon1), slab_C.shape[-1]))  # Match depth dimension

depth_mask = depth <= depth_limit  # Boolean mask for depth constraint
# Create writable Bound array and enforce depth constraint
Bound = np.broadcast_to(Bound[:, :, np.newaxis], (len(glat1), len(glon1), len(depth))).copy()
Bound[:, :, ~depth_mask] = 0  # Remove Bound below 100 km

# Initialize compositional arrays
C_bound = np.zeros_like(Bound)  # Bound compositional field
C_slab = np.zeros_like(Slab)    # Slab compositional field
C_crust = np.zeros_like(Crust)  # Crust compositional field

# Assign Bound, Slab, Crust as per the conditions
C_bound[Bound == 1] = 1  # Assign Bound regions
C_slab[Slab == 1] = 2   # Assign Slab regions (from slabs1)
C_crust[Crust != 0] = 3  # Assign Crust regions (from Composi1)

# Set plate boundary to 0 where Slab or Crust is not 0
C_bound[(C_slab != 0) | (C_crust != 0)] = 0  # 

print("assigned compositional fields")

# Add lithosphere based on depth
lith = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/mean_no_slabs.l.grd", 'r')
lon_lith = lith.variables['lon'][:]
lat_lith = lith.variables['lat'][:]
lthick = lith.variables['z'][:]

newlith = interpn((lat_lith, lon_lith), lthick, (glat, glon), method='linear')

# Initialize the continents array
dz = 10
continents = np.zeros_like(C_slab)

# Add continents based on lithosphere thickness
for i in range(len(glat1)):
    for j in range(len(glon1)):
        if newlith[i, j] >= 170:  # Threshold for continents (e.g., 170 km lithosphere thickness)
            # Find the depth index range where lithosphere thickness matches
            iz = [idx for idx, val in enumerate(depth) if newlith[i, j] >= np.round((val - (dz / 2))) and newlith[i, j] < np.round((val + (dz / 2)))]
            iz = iz[0] + 1  # First valid index, plus 1 for depth step
            continents[i, j, :iz] = 1  # Assign 1 for continents up to depth index iz

print("added continents")

# Step 1: Identify where C_crust is 3 (surface crust)
crust_surface = C_crust[:, :, 0] == 3

# Step 2: Find the latitudes of the surface crust
latitudes_with_crust = glat1[crust_surface.any(axis=1)]

# Step 3: Calculate the center latitude (mean of these latitudes)
center_latitude = np.mean(latitudes_with_crust)

#Step 4: find the closest latitude to the center latitude
closest_lat_index = (np.abs(glat1 - center_latitude)).argmin()
# Step 1: Identify where C_crust is 3 (surface crust) in the row at closest latitude
surface_crust_indices = np.where(C_crust[closest_lat_index, :, 0] == 3)[0]

# Step 2: Find the bounds i, j of the surface crust
i, j = surface_crust_indices.min(), surface_crust_indices.max()

# Step 3: Extract the row values between i and j (inclusive)
row_values = C_crust[closest_lat_index, i:j+1, 0]  # Values of the row at closest latitude within bounds



#Step 5: find the middle index i of C_crust[closest_lat_index, :, 0]
middle_index = (i + j) // 2  # Middle index in the original longitude array
centroid_lon = glon1[middle_index]  # Longitude at the middle index

print("Compo at center latitude:", C_crust[closest_lat_index, middle_index, 0])


# Define the target longitudes (±3 degrees from centroid longitude)
deg_offset = 3
target_lon_left = centroid_lon - deg_offset
target_lon_right = centroid_lon + deg_offset 

# Find the closest indices within the longitude array
index_left = np.abs(glon1 - target_lon_left).argmin()
index_right = np.abs(glon1 - target_lon_right).argmin()

# Find and print the depth at which C_crust == 3 for the offset points
offset_points = [index_left, index_right]
offset_lons = [glon1[index_left], glon1[index_right]]

correct_side = None  # Variable to store the side where crust should be extended

for idx, lon_idx in enumerate(offset_points):
    depths_with_crust = np.where(C_crust[closest_lat_index, lon_idx, :] == 3)[0]
    side = "left" if idx == 0 else "right"
    
    if depths_with_crust.size > 0:
        depth_at_crust = depth[depths_with_crust]
    else:
        correct_side = side  # Store the side where crust is missing

print(f"The correct side to extend the crust is: {correct_side}")



#Step6: Set up an array of the last points of C_crust on the correct side
coords = []

# Iterate over all latitudes
for i in range(len(glat1)):
    # Depending on the side, iterate over longitudes in the correct direction
    if correct_side == "left":
        # Check from left to right (as before)
        for j in range(len(glon1)):
            if C_crust[i, j, 0] == 3:  # Check if C_crust is 3 at the specific point
                coords.append((glat1[i], glon1[j]))
                break  # Stop once the leftmost point is found for this latitude
    elif correct_side == "right":
        # Check from right to left (find the rightmost point)
        for j in range(len(glon1) - 1, -1, -1):  # Start from the rightmost and move left
            if C_crust[i, j, 0] == 3:  # Check if C_crust is 3 at the specific point
                coords.append((glat1[i], glon1[j]))
                break  # Stop once the rightmost point is found for this latitude
    else:
        continue  # If the correct side is neither "left" nor "right", skip this latitude

# Convert the list of coordinates to a numpy array for further use
coords = np.array(coords)


# Save compositional fields as separate arrays
ds = nc.Dataset('/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry_continents.nc', 'w', format="NETCDF4")
glon_di = ds.createDimension('glon_di', len(glon1))
glat_di = ds.createDimension('glat_di', len(glat1))
gdepth_di = ds.createDimension('gdepth_di', len(depth))

glon_var = ds.createVariable('glon_var', np.float32, ('glon_di',))
glon_var[:] = glon1

glat_var = ds.createVariable('glat_var', np.float32, ('glat_di',))
glat_var[:] = glat1

gdepth_var = ds.createVariable('gdepth_var', np.float32, ('gdepth_di',))
gdepth_var[:] = depth

C_bound_var = ds.createVariable('C_bound', 'i1', ('glat_di', 'glon_di', 'gdepth_di'))
C_bound_var[:] = C_bound

C_slab_var = ds.createVariable('C_slab', 'i1', ('glat_di', 'glon_di', 'gdepth_di'))
C_slab_var[:] = C_slab

C_crust_var = ds.createVariable('C_crust', 'i1', ('glat_di', 'glon_di', 'gdepth_di'))
C_crust_var[:] = C_crust

continents_var = ds.createVariable('continents', 'i1', ('glat_di', 'glon_di', 'gdepth_di'))
continents_var[:] = continents

ds.close()

print("saved nc file")









