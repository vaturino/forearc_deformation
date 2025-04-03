#!/usr/bin/env python3

import netCDF4 as nc
import numpy as np
from scipy.interpolate import interpn
import time
import scipy.io
from scipy.ndimage import generate_binary_structure, binary_dilation

T1 = time.time()
print(T1)

# Make gridded vertical plate boundaries in 2-d from Bird2003 dataset
# specify parameters
W = 15  # half-width of plate boundaries zone (km)
dx = 0.2  # grid spacing (degree)   # keep same spacing with slab geometry
depth_limit = 50  # Max depth for Bound in km

# Read in Bird-2003 plate boundaries --- collected by Sam
data = scipy.io.loadmat('/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/plbd_0_360.mat')
lon = data['lon'][:,-1]  # 6870x1
lat = data['lat'][:,-1]  # 6870x1  

lon = np.interp(np.linspace(0, 1, len(lon)*5), np.linspace(0, 1, len(lon)), lon)  # 30240  
lat = np.interp(np.linspace(0, 1, len(lat)*5), np.linspace(0, 1, len(lat)), lat)  # 30240
lat = lat[~np.isnan(lon)]
lon = lon[~np.isnan(lon)]

# Initialize arrays
glon1 = np.arange(0, 360 + dx, dx, dtype=np.float32)   #1801
glat1 = np.arange(-90, 90 + dx, dx, dtype=np.float32)   #901
Bound = np.zeros((len(glat1), len(glon1)), dtype=np.uint8)
Slab = np.zeros((len(glat1), len(glon1)), dtype=np.uint8)
Crust = np.zeros((len(glat1), len(glon1)), dtype=np.uint8)

glat, glon = np.meshgrid(glat1, glon1, indexing='ij')

dlat = 6371 * 1000 * (np.pi) / 180    
dlon = 6371 * 1000 * (np.pi) / 180 * np.cos(np.radians(lat))  

print("created grid")

for i in range(len(lon)):
    d = np.sqrt(((lon[i] - glon) * dlon[i])**2 + ((lat[i] - glat) * dlat)**2) / 1000 
    Bound[d <= W] = 1

print("added plate boundaries")

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


print("Shapes after broadcasting:")
print("Slab:", Slab.shape)  # Should match (901, 1801, len(depth))
print("Crust:", Crust.shape)  # Should match (901, 1801, len(depth))
print("Bound:", Bound.shape)  # Should match (901, 1801, len(depth))

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
lon = lith.variables['lon'][:]
lat = lith.variables['lat'][:]
lthick = lith.variables['z'][:]

newlith = interpn((lat, lon), lthick, (glat, glon), method='linear')

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

# # Write to the text file
# fname = '/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry_continents.txt'
# glon = np.arange(0, 360 + dx, dx, dtype=np.float32)  #1801
# glat = np.arange(-90, 90 + dx, dx, dtype=np.float32)  #901
# glatm, glonm, gdepthm = np.meshgrid(glat, glon, depth, indexing='ij')

# # Convert into spherical coordinates
# r = (6371 - gdepthm) * 1000  # Radius (in meters)
# phi = glonm * np.pi / 180  # Longitude in radians
# theta = (90 - glatm) * np.pi / 180  # Latitude in radians

# # Slice out the last dimension for a specific depth
# r = r[:,:-1,:]
# phi = phi[:,:-1,:]
# theta = theta[:,:-1,:]

# # Write to text file
# with open(fname, 'w') as fid:
#     fid.write('# POINTS: ' + str(r.shape[2]) + ' ' + str(r.shape[1]) + ' ' + str(r.shape[0]) + '\n')
#     fid.write('# Columns: r phi theta C_bound C_slab C_crust continents\n')
#     fid.write('# Bird plate boundaries, width of boundary zone 30 km (Composition 1)\n')
#     fid.write('# Dipping boundaries above slabs, following SLAB2 slab top\n')
#     fid.write('# Plate boundaries go as deep as lithosphere, from Steinberger and Becker 2018\n')
#     fid.write('# Rest of mantle is background, or I can add continents information\n')

#     for i in range(r.shape[0]):  # Loop over latitudes (theta)
#         for j in range(r.shape[1]):  # Loop over longitudes (phi)
#             for k in range(r.shape[2]):  # Loop over depth (r)
#                 fid.write(f"%.0f %.4f %.4f %i %i %i %i \n" % (
#                     r[i, j, k], phi[i, j, k], theta[i, j, k], 
#                     C_bound[i, j, k], C_slab[i, j, k], C_crust[i, j, k], 
#                     continents[i, j, k]))

# T2 = time.time()

# print("saved txt file")
# print(T2 - T1)
