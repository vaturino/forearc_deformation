#!/usr/bin/env python3

import netCDF4 as nc
import numpy as np
from scipy.interpolate import interp1d, interpn,griddata
from shapely.geometry import Point, Polygon
import pandas as pd
import time
import scipy.io
from scipy.ndimage import generate_binary_structure, binary_dilation
from scipy.ndimage import label
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
depth_limit = 150  # Max depth for Bound in km

# Read in Bird-2003 plate boundaries --- collected by Sam
data = scipy.io.loadmat('/home/vturino/PhD/projects/forearc_deformation/plates/plate_boundaries.mat')
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
glon1 = np.linspace(min_lon, max_lon, int((max_lon - min_lon) / dx) + 1, endpoint=True, dtype=np.float32)  # longitude range from min_lon to max_lon
glat1 = np.linspace(min_lat, max_lat, int((max_lat - min_lat) / dx) + 1, endpoint=True, dtype=np.float32)  # latitude range from min_lat to max_lat
Bound = np.zeros((len(glat1), len(glon1)), dtype=np.uint8)
# Slab = np.zeros((len(glat1), len(glon1)), dtype=np.uint8)
Crust = np.zeros((len(glat1), len(glon1)), dtype=np.uint8)

glat, glon = np.meshgrid(glat1, glon1, indexing='ij')

dlat = 6371 * 1000 * (np.pi) / 180    
dlon = 6371 * 1000 * (np.pi) / 180 * np.cos(np.radians(lat_new))  

print("Created grid")

# Iterate through lon, lat and update Bound array
for i in range(len(lon)):
    d = np.sqrt(((glon - lon[i]) * dlon[i])**2 + ((glat - lat[i]) * dlat)**2) / 1000
    Bound |= (d <= W)

# Apply binary dilation to ensure continuity of plate boundaries (closing small gaps)
# struct = generate_binary_structure(2, 2)  # Structuring element for dilation
struct = np.ones((5, 5), dtype=bool)  # square 3x3 neighborhood
Bound = binary_dilation(Bound, structure=struct, iterations=3).astype(np.uint8)  # Dilation to remove gaps
from scipy.ndimage import binary_erosion
Bound = binary_erosion(Bound, structure=np.ones((2, 2)), iterations=4).astype(np.uint8)

print("Added plate boundaries with dilation")


# Add in slab plate boundaries
slab2_file = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry.nc", 'r')
depth = slab2_file.variables['gdepth1'][:]
slab_C = slab2_file.variables['Composi1'][:]  # Crust (Composi1) from sam_geometry
slab = slab2_file.variables['slabs1'][:]  # Slab (slabs1) from sam_geometry
slab2_file.close()


print("read in slab geometry")

# Ensure correct broadcasting instead of tiling
# Slab = np.broadcast_to(slab, (len(glat1), len(glon1), slab.shape[-1]))  # Keep slab's last dimension
Crust = np.broadcast_to(slab_C, (len(glat1), len(glon1), slab_C.shape[-1]))  # Match depth dimension

depth_mask = depth <= depth_limit  # Boolean mask for depth constraint
# Create writable Bound array and enforce depth constraint
Bound = np.broadcast_to(Bound[:, :, np.newaxis], (len(glat1), len(glon1), len(depth))).copy()
Bound[:, :, ~depth_mask] = 0  # Remove Bound below 100 km

# Initialize compositional arrays
C_bound = np.zeros_like(Bound)  # Bound compositional field
# C_slab = np.zeros_like(Slab)    # Slab compositional field
C_crust = np.zeros_like(Crust)  # Crust compositional field

# Assign Bound, Slab, Crust as per the conditions
C_bound[Bound == 1] = 1  # Assign Bound regions
# C_slab[Slab == 1] = 2   # Assign Slab regions (from slabs1)
C_crust[Crust != 0] = 3  # Assign Crust regions (from Composi1)

# C_bound[Slab == 1] = 0  # Clear Bound regions where Slab is present
C_bound[Crust != 0] = 0  # Clear Bound regions where Crust is present

# print("assigned initial compositional fields")

# # ================================================== Load and preprocess lithosphere/continent thickness data ===============================================

# # # Add lithosphere
# # lith = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/mean_no_slabs.l.grd",'r')
# # lon = lith.variables['lon'][:]
# # lat = lith.variables['lat'][:]
# # lthick = lith.variables['z'][:]

# # newlith=interpn((lat,lon), lthick, (glat, glon), method='linear')

# print("Loading lithosphere thickness data from Afonso et al 2019 (LithoRef18)...")
# lith_thick = pd.read_csv(
#     "/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/LithoRef_model/LithoRef18.xyz",
#     skiprows=8, sep='\s+'
# )

# lat_lith = (lith_thick['LAT'].values + 360) % 360  # Convert to 0–360
# lon_lith = lith_thick['LONG']
# thick_lith = -lith_thick['LAB'] / 1.e3  # Convert to km if necessary

# #this is a point-like dataset, so we need to interpolate it onto the glat glong grid
# print("Interpolating lithosphere thickness onto model grid...")


# # Combine longitudes and latitudes into (N, 2) shape
# points_lith = np.column_stack(((lon_lith + 360) % 360, lith_thick['LAT'].values))  # Ensure 0–360 range for lon
# values_lith = thick_lith

# # Create 2D meshgrid for interpolation (for the model grid)
# glatm2d, glonm2d = np.meshgrid(glat1, glon1, indexing='ij')
# grid_points = np.column_stack((glonm2d.ravel(), glatm2d.ravel()))

# # Interpolate lithosphere thickness data onto model grid
# newlith = griddata(
#     points_lith, values_lith, grid_points, method='cubic'
# ).reshape(len(glat1), len(glon1))


# # Optional: mask invalid (land or missing) areas
# newlith = np.ma.masked_invalid(newlith)



# # plt.imshow(newlith, extent=(glon1.min(), glon1.max(), glat1.min(), glat1.max()), origin='lower', interpolation='nearest')
# # plt.colorbar(label='Lithospheric Thickness (km)')
# # plt.show()
# # exit()



# # newlith = interpn((lat_lith, lon_lith), values_lith, (glat, glon), method='linear')


# # # Initialize the continents array
# # dz = 10
# # continents = np.zeros_like(C_slab)

# # # Add continents based on lithosphere thickness
# # for i in range(len(glat1)):
# #     for j in range(len(glon1)):
# #         if newlith[i, j] >= 170:  # Threshold for continents (e.g., 170 km lithosphere thickness)
# #             # Find the depth index range where lithosphere thickness matches
# #             iz = [idx for idx, val in enumerate(depth) if newlith[i, j] >= np.round((val - (dz / 2))) and newlith[i, j] < np.round((val + (dz / 2)))]
# #             iz = iz[0] + 1  # First valid index, plus 1 for depth step
# #             continents[i, j, :iz] = 5  # Assign 1 for continents up to depth index iz

# # print("added continents")

# # Step 1: Identify where C_crust is 3 (surface crust)
# crust_surface = C_crust[:, :, 0] == 3

# # Step 2: Find the latitudes of the surface crust
# latitudes_with_crust = glat1[crust_surface.any(axis=1)]

# # Step 3: Calculate the center latitude (mean of these latitudes)
# center_latitude = np.mean(latitudes_with_crust)

# #Step 4: find the closest latitude to the center latitude
# closest_lat_index = (np.abs(glat1 - center_latitude)).argmin()
# # Step 1: Identify where C_crust is 3 (surface crust) in the row at closest latitude
# surface_crust_indices = np.where(C_crust[closest_lat_index, :, 0] == 3)[0]

# # Step 2: Find the bounds i, j of the surface crust
# i, j = surface_crust_indices.min(), surface_crust_indices.max()

# # Step 3: Extract the row values between i and j (inclusive)
# row_values = C_crust[closest_lat_index, i:j+1, 0]  # Values of the row at closest latitude within bounds



# #Step 5: find the middle index i of C_crust[closest_lat_index, :, 0]
# middle_index = (i + j) // 2  # Middle index in the original longitude array
# centroid_lon = glon1[middle_index]  # Longitude at the middle index

# # Define the target longitudes (±3 degrees from centroid longitude)
# deg_offset = 3
# target_lon_left = centroid_lon - deg_offset
# target_lon_right = centroid_lon + deg_offset 

# # Find the closest indices within the longitude array
# index_left = np.abs(glon1 - target_lon_left).argmin()
# index_right = np.abs(glon1 - target_lon_right).argmin()

# # Find and print the depth at which C_crust == 3 for the offset points
# offset_points = [index_left, index_right]
# offset_lons = [glon1[index_left], glon1[index_right]]

# correct_side = None  # Variable to store the side where crust should be extended

# for idx, lon_idx in enumerate(offset_points):
#     depths_with_crust = np.where(C_crust[closest_lat_index, lon_idx, :] == 3)[0]
#     side = "left" if idx == 0 else "right"
    
#     if depths_with_crust.size > 0:
#         depth_at_crust = depth[depths_with_crust]
#     else:
#         correct_side = side  # Store the side where crust is missing

# print(f"The correct side to extend the crust is: {correct_side}")


# #Step6: Set up an array of the last points of C_crust on the correct side
# coords = []

# # Iterate over all latitudes
# for i in range(len(glat1)):
#     # Depending on the side, iterate over longitudes in the correct direction
#     if correct_side == "left":
#         # Check from left to right (as before)
#         for j in range(len(glon1)):
#             if C_crust[i, j, 0] == 3:  # Check if C_crust is 3 at the specific point
#                 coords.append((glat1[i], glon1[j]))
#                 break  # Stop once the leftmost point is found for this latitude
#     elif correct_side == "right":
#         # Check from right to left (find the rightmost point)
#         for j in range(len(glon1) - 1, -1, -1):  # Start from the rightmost and move left
#             if C_crust[i, j, 0] == 3:  # Check if C_crust is 3 at the specific point
#                 coords.append((glat1[i], glon1[j]))
#                 break  # Stop once the rightmost point is found for this latitude
#     else:
#         continue  # If the correct side is neither "left" nor "right", skip this latitude

# # Convert the list of coordinates to a numpy array for further use
# coords = np.array(coords)


# # Step 7: fill the correct side with crust and wrong side with OP
# nazca = pd.read_csv('/home/vturino/PhD/projects/forearc_deformation/plates/coords_NZ.csv')
# southamerica = pd.read_csv('/home/vturino/PhD/projects/forearc_deformation/plates/coords_SA_adjusted.csv')

# crust_depth_limit = 15  # km

# # Limit to our region
# nazca = nazca[(nazca['lon'] >= min_lon) & (nazca['lon'] <= max_lon) & (nazca['lat'] >= min_lat) & (nazca['lat'] <= max_lat)]
# southamerica = southamerica[(southamerica['lon'] >= min_lon) & (southamerica['lon'] <= max_lon) & (southamerica['lat'] >= min_lat) & (southamerica['lat'] <= max_lat)]

# #reindex 
# nazca = nazca.reset_index(drop=True)
# southamerica = southamerica.reset_index(drop=True)

# max_top = southamerica[southamerica["lat"] > 10]['lon'].idxmax()
# max_bottom = southamerica[southamerica["lat"] < 0]['lon'].idxmax()


# southamerica["lon"].iloc[max_top] = max_lon+1
# southamerica["lon"].iloc[max_top-8:max_top] = max_lon+1
# southamerica["lon"].iloc[max_bottom] = max_lon+1


# # plt.plot(southamerica["lon"], southamerica["lat"], 'r', label='South America')
# # plt.scatter(southamerica["lon"].iloc[max_top], southamerica["lat"].iloc[max_top], color='blue', label='Top Point')
# # plt.scatter(southamerica["lon"].iloc[max_bottom], southamerica["lat"].iloc[max_bottom], color='green', label='Bottom Point')
# # plt.show()
# # exit()

# # Extend the plate on the right side if needed
# if correct_side == 'left':
#     plate_to_extend = southamerica
# else:
#     plate_to_extend = nazca

# # Ensure plate extends properly on the right if needed
# if plate_to_extend['lon'].iloc[-1] < glon1[-1]:
#     top_lat = plate_to_extend['lat'].iloc[-1]
#     extension = pd.DataFrame({
#         'lon': [glon1[-1]] * len(plate_to_extend),
#         'lat': np.linspace(top_lat, plate_to_extend['lat'].iloc[0], len(plate_to_extend))
#     })
#     if correct_side == 'left':
#         southamerica = pd.concat([southamerica, extension], ignore_index=True)
#     else:
#         nazca = pd.concat([nazca, extension], ignore_index=True)



# # Ensure polygons are closed (important for mask generation)
# if not nazca.iloc[0].equals(nazca.iloc[-1]):
#     nazca = pd.concat([nazca, nazca.iloc[[0]]], ignore_index=True)
# if not southamerica.iloc[0].equals(southamerica.iloc[-1]):
#     southamerica.iloc[-1] = southamerica.iloc[0] 

# southamerica = southamerica[~southamerica.duplicated(keep='last')] 
# southamerica = southamerica[~((southamerica["lon"] > 310) & (southamerica["lat"] > -10) & (southamerica["lat"] < 0) )]
# southamerica.reset_index(drop=True, inplace=True)


# # Create polygons
# nazca_polygon = Polygon(zip(nazca['lon'], nazca['lat']))
# southamerica_polygon = Polygon(zip(southamerica['lon'], southamerica['lat']))

# # plt.plot(nazca_polygon.exterior.xy[0], nazca_polygon.exterior.xy[1], color='red', label='Nazca Polygon')
# # plt.plot(southamerica_polygon.exterior.xy[0], southamerica_polygon.exterior.xy[1], color='blue', label='South America Polygon')
# # plt.scatter(nazca['lon'].iloc[0], nazca['lat'].iloc[0], color='red', label='Nazca Start')
# # plt.scatter(southamerica['lon'].iloc[0], southamerica['lat'].iloc[0], color='blue', label='South America Start')
# # plt.scatter(southamerica['lon'].iloc[-1], southamerica['lat'].iloc[-1], color='green', label='Top Point')
# # plt.show()
# # exit()



# print("created polygons for SP and OP")

# # Precompute the polygon masks
# nazca_mask = np.array([[nazca_polygon.covers(Point(glon1[j], glat1[i])) for j in range(len(glon1))] for i in range(len(glat1))])
# southamerica_mask = np.array([[southamerica_polygon.covers(Point(glon1[j], glat1[i])) for j in range(len(glon1))] for i in range(len(glat1))])

# plt.figure(figsize=(10, 6))
# plt.imshow(southamerica_mask, cmap='Blues', alpha=0.5, origin='lower',
#            extent=[glon1.min(), glon1.max(), glat1.min(), glat1.max()])
# plt.title("South America Mask")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.colorbar(label='Mask Value')
# plt.show()

# # Ensure C_OP extends to the right boundary
# C_OP = np.zeros_like(C_crust, dtype=np.uint8)

# for i in range(len(glat1)):
#     for j in range(len(glon1)):
#         local_op_depth_limit = newlith[i, j]  # lithospheric thickness at this point

#         # Ensure that we are processing the correct plate (Nazca or South America)
#         if correct_side == 'left':
#             if nazca_mask[i, j]:
#                 # Assign crust/slab for Nazca
#                 C_crust[i, j, depth <= crust_depth_limit] = 3
#                 C_slab[i, j, (depth > crust_depth_limit) & (depth <= local_op_depth_limit)] = 2
#                 C_OP[i, j, depth <= local_op_depth_limit] = 0  # Reset OP

#             if southamerica_mask[i, j] and not nazca_mask[i, j]:  # Avoid overlap
#                 # Assign OP for South America only if it's not also Nazca
#                 C_OP[i, j, depth <= local_op_depth_limit] = 4
            

#         else:  # correct_side == 'right'
#             if southamerica_mask[i, j]:  # Check if the point belongs to South America
#                 # Assign Crust and Slab values for South America
#                 C_crust[i, j, depth <= crust_depth_limit] = 3
#                 C_slab[i, j, (depth > crust_depth_limit) & (depth <= local_op_depth_limit)] = 2
#                 # Ensure that OP is not assigned for South America
#                 C_OP[i, j, depth <= local_op_depth_limit] = 0  # Explicitly reset OP for South America
#             elif nazca_mask[i, j]:  # Check if the point belongs to the Nazca plate
#                 # Assign OP for Nazca
#                 C_OP[i, j, depth <= local_op_depth_limit] = 4

# # Ensure that C_OP extends fully to the right boundary if necessary
# right_boundary_col = len(glon1) - 1
# for i in range(len(glat1)):
#     # Check the last column (right boundary) and set C_OP if it's within the region for OP
#     if nazca_mask[i, right_boundary_col]:
#         C_OP[i, right_boundary_col, depth <= newlith[i, right_boundary_col]] = 4
#     elif southamerica_mask[i, right_boundary_col]:
#         C_OP[i, right_boundary_col, depth <= newlith[i, right_boundary_col]] = 4




# # Approximate bounding box with some padding
# lon_min, lon_max = 279.7, 283
# lat_min, lat_max = 4.2, 7.8


# # Find index masks
# lon_mask = (glon1 >= lon_min) & (glon1 <= lon_max)
# lat_mask = (glat1 >= lat_min) & (glat1 <= lat_max)
# i_lat = np.where(lat_mask)[0]
# j_lon = np.where(lon_mask)[0]

# # Loop through the grid section
# for i in i_lat:
#     for j in j_lon:
#         if np.all(C_crust[i, j, :] == 0) or np.all(C_slab[i, j, :] == 0):
#             # Use same fill logic
#             C_crust[i, j, depth <= crust_depth_limit] = 3
#             C_slab[i, j, (depth > crust_depth_limit) & (depth <= newlith[i, j])] = 2



# print("extended crust and OP")

# # Define crust depth indices up to 30 km
# crust_depth_indices = np.where(depth <= crust_depth_limit)[0]

# for lat_val, lon_val in coords:
#     lat_idx = np.argmin(np.abs(glat1 - lat_val))
#     lon_idx = np.argmin(np.abs(glon1 - lon_val))

#     for k in crust_depth_indices:
#         if correct_side == 'left':
#             # Move right (increasing longitude index)
#             for j in range(lon_idx + 1, len(glon1)):
#                 if C_crust[lat_idx, j, k] == 3:  # Crust already exists at this depth
#                     # Fill from lon_idx to j (inclusive) at depth k
#                     C_crust[lat_idx, lon_idx:j + 1, k] = 3
#                     break
#         elif correct_side == 'right':
#             # Move left (decreasing longitude index)
#             for j in range(lon_idx - 1, -1, -1):
#                 if C_crust[lat_idx, j, k] == 3:
#                     C_crust[lat_idx, j:lon_idx + 1, k] = 3
#                     break

# for lat_val, lon_val in coords:
#     lat_idx = np.argmin(np.abs(glat1 - lat_val))
#     lon_idx = np.argmin(np.abs(glon1 - lon_val))

#     # Find the deepest point of the lithosphere (newlith) at the junction
#     local_op_depth_limit = newlith[lat_idx, lon_idx]

#     # Find the first deepest point of the slab within the local lithospheric thickness
#     slab_depth_limit = np.max(depth)  # Start with the deepest depth as a potential slab depth
#     for k, d in enumerate(depth):
#         if C_slab[lat_idx, lon_idx, k] == 2:  # Slab material exists at this depth
#             slab_depth_limit = depth[k]  # Update slab depth if a slab exists at this depth
#             break

#     # Fill from the deepest point of lithosphere to the first slab point, above
#     if local_op_depth_limit < slab_depth_limit:  # Ensure lithosphere depth is below slab depth
#         for k, d in enumerate(depth):
#             if d <= local_op_depth_limit:
#                 C_slab[lat_idx, lon_idx, k] = 2  # Fill slab up to lithosphere depth
#             elif d > local_op_depth_limit and d <= slab_depth_limit:
#                 C_crust[lat_idx, lon_idx, k] = 3  # Fill the boundary above slab
#                 C_slab[lat_idx, lon_idx, k] = 0  # Clear slab above lithosphere depth


# print("smoothed crust composition")


# # Iterate through the grid to adjust C_bound based on C_crust
# for i in range(C_OP.shape[0]):  # Latitude
#     for j in range(C_OP.shape[1]):  # Longitude
#         # Find the first crust depth index (i.e., top of crust)
#         crust_indices = np.where(C_crust[i, j, :] == 3)[0]
#         if crust_indices.size > 0:
#             crust_top = crust_indices[0]

#             # Remove OP starting from that crust depth and downward
#             C_OP[i, j, crust_top:] = 0


# C_crust[C_bound == 1] = 0  # Clear crust regions where Bound is present
# C_slab[(C_bound == 1) | (C_crust == 3)] = 0  # Clear slab regions where Bound is present
# C_OP[(C_bound == 1) | (C_crust == 3) | (C_slab == 2)] = 0  # Clear OP regions where Bound is present

# C = np.zeros_like(C_bound)  # Initialize C array
# C = C_crust+C_OP

# plt.imshow(C[:, :, 0], extent=(glon1.min(), glon1.max(), glat1.min(), glat1.max()), origin='lower', interpolation='nearest')
# plt.colorbar(label='Composition')
# plt.show()
# exit()



gdepth = depth

# Save compositional fields as separate arrays
ds = nc.Dataset('/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry_crust_plbd.nc', 'w', format="NETCDF4")
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


C_crust_var = ds.createVariable('C_crust', 'i1', ('glat_di', 'glon_di', 'gdepth_di'))
C_crust_var[:] = C_crust

ds.close()

print("saved nc file")

fname = '/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry_crust_plbd.txt'
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
    fid.write('# Columns: r phi theta plbd crust\n')
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
                fid.write("%.0f %.4f %.4f %i %i \n" % (
                    r[i,j,k], phi[i,j,k], theta[i,j,k],
                    C_bound[i,j,k], C_crust[i,j,k]
                ))
         
T2=time.time()
print(T2-T1)