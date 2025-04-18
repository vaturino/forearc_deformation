#!/usr/bin/env python3

import netCDF4 as nc
import numpy as np
from scipy.interpolate import interpn
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator
import pandas as pd
from scipy.special import erfc, erf
import time
import warnings
import scipy.ndimage as ndimage
import scipy.io
from scipy.ndimage import gaussian_filter


warnings.simplefilter("ignore", category=DeprecationWarning)

# Function to apply Gaussian smoothing to the final temperature field
def smooth_final_temp(temp_field, sigma=1.0):
    """ Apply Gaussian smoothing to the final temperature field (lat-lon-depth). """
    smoothed_field = np.empty_like(temp_field)
    
    # Apply smoothing for each depth level (along the third axis) across the lat-lon plane
    for k in range(temp_field.shape[2]):
        smoothed_field[:, :, k] = ndimage.gaussian_filter(temp_field[:, :, k], sigma=sigma)
    
    return smoothed_field





T1 = time.time()
# Define parameters
vs_to_density_scaling = 0.2
vp_to_density_scaling = 0.4
rho_ref = 3300
temp_ref = 1573
temp_surf = 273
expansivity = 3e-5

min_lat = -60
max_lat = 20
min_lon = 240
max_lon = 320

# Define structured grid, non-uniform resolution, refined where slabs
dx = 0.1  # horizontal resolution away from slabs
dz = 5  # vertical resolution for upper mantle
hr_depth = 300  # depth of high-resolution grid
slab2_file = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry.nc", 'r')  ### Your slab geometry nc file
glon = slab2_file.variables['glon1'][:]   # 0.2 degree
glat = slab2_file.variables['glat1'][:]   # 0.2 degree
slab = slab2_file.variables['slabs1'][:]
slab2_file.close()

slabs = np.sum(slab, axis=2).astype(bool)
slabs_lon = np.sum(slabs, axis=0).astype(bool)  # Longitude where have slabs
slabs_lat = np.sum(slabs, axis=1).astype(bool)  # Latitudes where have slabs

# Combine global with high-resolution (0.2 degrees) where slabs
glon1 = np.concatenate((glon[slabs_lon], np.arange(min_lon, max_lon + dx, dx)))
glon1 = np.sort(np.unique(np.round(glon1, decimals=1)))  # 1382

glat1 = np.concatenate((glat[slabs_lat], np.arange(min_lat, max_lat + dx, dx)))
glat1 = np.sort(np.unique(np.round(glat1, decimals=1)))  # 665
slabs_lon = slabs_lat = None

# Read tomography data file for mantle temperature calculation
tomogrpahy_file = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/t_savani-dlnvs-add.nc", 'r')
depth = tomogrpahy_file.variables['dep'][:]
lon = tomogrpahy_file.variables['lon'][:]
lat = tomogrpahy_file.variables['lat'][:]
tomogrpahy_file.close()

lon = np.round(lon, decimals=2)
lat = np.round(lat, decimals=2)

# Read Savani data
gdepth_hr = np.arange(0, hr_depth + dz, dz).astype(np.float32)
gdepth_mid = np.arange(hr_depth, 670, 4 * dz).astype(np.float32)
gdepth_low = np.arange(670, 1200, 6 * dz).astype(np.float32)
# gdepth1 = np.concatenate((np.arange(0, 670, dz), depth[(depth > 660) & (depth <= 1200)])).astype(np.float32)
gdepth1 = np.concatenate((gdepth_hr, gdepth_mid)).astype(np.float32)
gdepth1 = np.unique(gdepth1)  # Remove duplicates
gdepth1[gdepth1 > 1100] = 1100


# Create a meshgrid with proper lat, lon
glatm, glonm, gdepthm = np.meshgrid(glat1, glon1, gdepth1, indexing='ij')
print(f"Depth range: {gdepth1.min()} km to {gdepth1.max()} km")

# Smooth transition parameters for depth
Dc = 300  # threshold
sw = 100  # smoothing width

# Interpolate tomography data onto the grid
tomogrpahy_file = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/t_savani-dlnvs-add.nc", 'r')
dlnvs = tomogrpahy_file.variables['dlnvs'][:]
dlnvs = np.transpose(dlnvs, (1, 2, 0))
tomogrpahy_file.close()

print("start dlnvs")
newdlnvs = interpn((lat, lon, depth), dlnvs, (glatm, glonm, gdepthm), method='linear', bounds_error=False)

print("calculate temp")
temp = np.float32(temp_ref + (-1. / expansivity) * vs_to_density_scaling * newdlnvs / 100)

# Smooth blend with tomography
W = (gdepth1 - Dc + sw / 2) / sw
W[W < 0] = 0
W[W > 1] = 1
W1 = W[np.newaxis, np.newaxis, :]
W1 = np.tile(W1, (len(glat1), len(glon1), 1))
temp = (W1 * temp + (1 - W1) * temp_ref)

# ================================================== Load and preprocess seafloor age data ===============================================
print("Loading seafloor age data from Seton et al., 2020...")
seafloor_age = nc.Dataset("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/age.2020.1.GTS2012.6m.grd", 'r')
lon_age = seafloor_age.variables['lon'][:]
lat_age = seafloor_age.variables['lat'][:]
age_raw = seafloor_age.variables['z'][:]
# age_raw[age_raw <=2] = 0
seafloor_age.close()

# Convert longitudes to 0–360
lon_age = (lon_age + 360) % 360

# Remove duplicate longitudes and sort
lon_sorted, lon_unique_idx = np.unique(lon_age, return_index=True)
age_unique_lon = age_raw[:, lon_unique_idx]  # keep only unique longitude columns

# Sort latitudes and reorder age accordingly
lat_sort_idx = np.argsort(lat_age)
lat_sorted = lat_age[lat_sort_idx]
age_sorted = age_unique_lon[lat_sort_idx, :]

# Confirm strictly increasing dimensions
assert np.all(np.diff(lat_sorted) > 0), "latitudes not strictly increasing"
assert np.all(np.diff(lon_sorted) > 0), "longitudes not strictly increasing"


# === Interpolate seafloor age onto your model grid ===
print("Interpolating seafloor age onto model grid...")
age_interp_func = RegularGridInterpolator(
    (lat_sorted, lon_sorted),
    age_sorted,
    bounds_error=False,
    fill_value=np.nan  # Leave areas outside the grid as NaN (i.e., land)
)

glatm2d, glonm2d = np.meshgrid(glat1, glon1, indexing='ij')
points = np.column_stack((glatm2d.ravel(), glonm2d.ravel()))
age_on_grid = age_interp_func(points).reshape(len(glat1), len(glon1))


# === Create land mask from missing seafloor age ===
print("Generating land mask (regions with no seafloor age)...")
land_mask = np.isnan(age_on_grid)


# === Compute HSC temperature field ===
print("Computing HSC temperature field...")
gdepth_hsc = gdepth1 * 1e3  # km -> m
kappa = 1e-6  # thermal diffusivity [m^2/s]
seafloor_temp = np.full((len(glat1), len(glon1), len(gdepth1)), np.nan, dtype=np.float32)

# Convert seafloor age to seconds
age_sec = age_on_grid * 1e6 * 365.25 * 24 * 3600  # Myr -> s

for k, z in enumerate(gdepth_hsc):
    with np.errstate(invalid='ignore'):
        T_hsc = temp_surf + (temp_ref - temp_surf) * erf(z / (2 * np.sqrt(kappa * age_sec)))
        seafloor_temp[:, :, k] = T_hsc

# Cap temperature to avoid exceeding mantle reference
seafloor_temp[seafloor_temp >= temp_ref] = np.nan


# ================================================== Load and preprocess lithosphere/continent thickness data ===============================================
print("Loading lithosphere thickness data from Afonso et al 2019 (LithoRef18)...")
lith_thick = pd.read_csv(
    "/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/LithoRef_model/LithoRef18.xyz",
    skiprows=8, sep='\s+'
)

lat_lith = (lith_thick['LAT'].values + 360) % 360  # Convert to 0–360
lon_lith = lith_thick['LONG']
thick_lith = -lith_thick['LAB'] / 1.e3  # Convert to km if necessary




#this is a point-like dataset, so we need to interpolate it onto the glat glong grid
print("Interpolating lithosphere thickness onto model grid...")


# Combine longitudes and latitudes into (N, 2) shape
points_lith = np.column_stack(((lon_lith + 360) % 360, lith_thick['LAT'].values))  # Ensure 0–360 range for lon
values_lith = thick_lith

# Create 2D meshgrid for interpolation (for the model grid)
glatm2d, glonm2d = np.meshgrid(glat1, glon1, indexing='ij')
grid_points = np.column_stack((glonm2d.ravel(), glatm2d.ravel()))

# Interpolate lithosphere thickness data onto model grid
lith_thick_on_grid = griddata(
    points_lith, values_lith, grid_points, method='cubic'
).reshape(len(glat1), len(glon1))

# lith_thick_on_grid = gaussian_filter(lith_thick_on_grid, sigma=10)



# Optional: mask invalid (land or missing) areas
lith_thick_on_grid = np.ma.masked_invalid(lith_thick_on_grid)

# Mask out ocean regions (i.e., where land_mask is False)
lith_thick_land_only = np.where(land_mask, lith_thick_on_grid, np.nan)

# Masked lithosphere thickness only where land (used for geotherm depth limit)
geo_temp_land = np.full((len(glat1), len(glon1), len(gdepth1)), np.nan, dtype=np.float32)


# Compute temperature at each depth using the inferred age and lithospheric thickness
geo_temp_land = np.full((len(glat1), len(glon1), len(gdepth1)), np.nan, dtype=np.float32)



# Modify the temperature profile for the continental lithosphere with a linear gradient
for k, z in enumerate(gdepth1):  # gdepth1 is in km
    # Linear gradient from surface to the LAB temperature
    z_ratio = np.where(lith_thick_land_only != 0, z / (lith_thick_land_only), 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        temp_k = temp_surf + z_ratio * (temp_ref - temp_surf)  # Linear temperature gradient
        
        # Adjust temperatures below the lithosphere depth (below LAB), extending to 50 km deeper
        temp_k[z > (lith_thick_land_only)] = np.nan  # Cap temperature below lithosphere depth
        
    geo_temp_land[:, :, k] = temp_k


# Ensure seafloor_temp has the same resolution as lith_thick and geo_temp_land
seafloor_temp_masked = np.isnan(seafloor_temp)

# Mask seafloor temperature where it's missing and fill with geo_temp_land data
seafloor_temp[seafloor_temp_masked] = geo_temp_land[seafloor_temp_masked]

# Smooth blend between geo_temp_land (continental) and seafloor_temp (oceanic)
W = (gdepth1 - Dc + sw / 2) / sw  # Depth blending
W = np.clip(W, 0, 1)
W3D = np.tile(W[np.newaxis, np.newaxis, :], (len(glat1), len(glon1), 1))

# Mask for regions where the temperature from the oceanic model (seafloor_temp) is missing
seafloor_temp_masked = np.isnan(seafloor_temp)

# Blend the oceanic and continental temperatures based on the mask
seafloor_temp[seafloor_temp_masked] = W3D[seafloor_temp_masked] * geo_temp_land[seafloor_temp_masked] + \
    (1 - W3D[seafloor_temp_masked]) * seafloor_temp[seafloor_temp_masked]





# === Smooth blend with tomography using transition zone ===
print("Blending HSC and tomography temperatures...")
W = (gdepth1 - Dc + sw / 2) / sw
W = np.clip(W, 0, 1)
W3D = np.tile(W[np.newaxis, np.newaxis, :], (len(glat1), len(glon1), 1))

blend_mask = ~np.isnan(seafloor_temp)
temp[blend_mask] = W3D[blend_mask] * temp[blend_mask] + (1 - W3D[blend_mask]) * seafloor_temp[blend_mask]


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

# Assume gdepth_target is what was used to generate temp/newslab_temp
gdepth_target = gdepthm[0, 0, :] 
valid_mask = ~np.isnan(newslab_temp)  
first_valid_idx = valid_mask.argmax(axis=2)  
all_nan_mask = ~valid_mask.any(axis=2)
slab_top_depth = gdepth_target[first_valid_idx] 
slab_top_depth[all_nan_mask] = np.nan

# Broadcast slab_top_depth to 3D
slab_top_3d = np.repeat(slab_top_depth[:, :, np.newaxis], len(gdepth_target), axis=2)  # (401, 401, 83)

# Check for valid slab temperature values (not NaN)
valid_mask = np.isfinite(newslab_temp)  # Ensure valid slab data for blending

# New blending approach to avoid NaN contamination
blended_temp = np.where(
    valid_mask,  # Only blend where valid slab temperatures exist
    np.where(
        gdepthm < slab_top_3d,
        temp,            # Use overriding plate temp above slab
        newslab_temp     # Use slab temp below, only if valid
    ),
    temp  # If slab temp is invalid (NaN), retain the original plate temp
)

# Final blended temperature (no NaN contamination)
final_temp = blended_temp.copy()
final_temp[final_temp < 273] = 273  # Cap negative temperatures to zero C

print(min(final_temp.flatten()), max(final_temp.flatten()))


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
vartemp[:]=final_temp
ds.close()

print("save to netcdf done")


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
                f.write(f"%.0f %.3f %.3f %.1f \n" % (r[i,j,k], phi[i,j,k], theta[i,j,k],final_temp[i,j,k]))
                        
T2=time.time()
T3=(T2-T1)
print(T3)
