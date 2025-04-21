#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
import sys, os
import json
import argparse
from pathlib import Path
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from libraries.functions import *
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker


# === Load Morell data ===
morell_data = pd.read_excel('/home/vturino/PhD/projects/forearc_deformation/postprocessing/morell_data/velocity_obliquity.ods', engine='odf')  
morell_data['lon'] = (morell_data['lon'] + 360) % 360
morell_data = morell_data[morell_data["bin"].str.contains("SAM", na=False)]
morell_data['velocity'] = morell_data['velocity'] / 10  # cm/yr to m/yr

lat = morell_data['lat'].values
lon = morell_data['lon'].values
velocity = morell_data['velocity'].values
obliquity = morell_data['obliquity'].values

# Compute velocity components
vx = velocity * np.cos(np.radians(obliquity))  # trench-parallel
vy = velocity * np.sin(np.radians(obliquity))  # trench-perpendicular

################################### READ IN INITIAL FILES ###################################
parser = argparse.ArgumentParser(description= 'Script that gets n models and the time at the end of computation and gives the temperature and viscosity plots for all time steps')
parser.add_argument('json_file', help='json file with models and end time to plot T and eta field')
args = parser.parse_args()


gz_folder = '/home/vturino/Vale_nas/forearc_deformation/gz_outputs/'  
models_loc = '/home/vturino/Vale_nas/forearc_deformation/raw_outputs/'  
json_loc = '/home/vturino/PhD/projects/forearc_deformation/postprocessing/json_files/'
plt_folder = '/home/vturino/PhD/projects/forearc_deformation/postprocessing/plots/single_models/'
if not os.path.exists(plt_folder):
    os.makedirs(plt_folder)

with open(f"{json_loc}{args.json_file}") as json_file:
        configs = json.load(json_file)

m = configs['model_name']
slines = configs['head_lines']

################################### GLOBAL VARIABLES ###################################
outer_radius = 6371.0e3
inner_radius = 5271.0e3

# Color palette for compositions
palette_by_composition = {
    'background': 'gainsboro',     
    'op': 'yellowgreen',      
    'crust': 'olivedrab',         
    'plbd': 'darkgreen'        
}

# Fixed composition field names
compo_field_mapping = {
    0: 'Background',
    1: 'Plate Boundaries',
    2: 'Oceanic Crust',
    3: 'Overriding Plate'
}
    

################################### GET TIME ARRAY ###################################
time_array = np.zeros((len(os.listdir(f"{gz_folder}{m}/fields")),2))
stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=slines,sep='\s+',header=None)
time_array = grab_dimTime_fields(f"{gz_folder}{m}/fields", stat, time_array, slines-1)
plot_loc_mod = f"{plt_folder}/{m}"

if not os.path.exists(plot_loc_mod):
    os.mkdir(plot_loc_mod)
plot_loc = f"{plot_loc_mod}/Velocity_comparison/"

if not os.path.exists(plot_loc):
    os.mkdir(plot_loc)
else:
    # Count the files in the fields_loc directory
    file_count = len(os.listdir(plot_loc))

################################### READ DATA AND PLOT COMPOSITION AT SURFACE ###################################
for ts in range(len(time_array)):
    all_data = pd.read_parquet(f"{gz_folder}{m}/fields/full.{ts}.gzip")

    # Calculate the radial distance for the entire dataset
    r_all = np.sqrt(all_data['Points:0']**2 + all_data['Points:1']**2 + all_data['Points:2']**2)


    # Find the surface points (within the tolerance)
    tolerance = 0.0  # 0 tolerance for surface points
    surf_idx = np.where(np.abs(r_all - outer_radius) <= tolerance)[0]

    # Filter surface data
    surf_data = all_data.iloc[surf_idx]
    # Filter surface data
    surf_data = all_data.iloc[surf_idx]
    # Normalize compositions to 0 or 1
    crust = (surf_data['crust'].values > 0.5).astype(int)
    op = (surf_data['op'].values > 0.5).astype(int)
    plbd = (surf_data['plbd'].values > 0.5).astype(int)

    # Convert Cartesian coordinates from meters to centimeters
    x = surf_data['Points:0'].values * 100  # cm
    y = surf_data['Points:1'].values * 100
    z = surf_data['Points:2'].values * 100
    # Surface velocities (convert to cm/yr)
    vx_model = surf_data['velocity:0'].values * 100  # cm/yr
    vy_model = surf_data['velocity:1'].values * 100
    vz_model = surf_data['velocity:2'].values * 100

    

    # === Example transformation of Cartesian coordinates to lat/lon ===
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    lat_model = 90 - np.degrees(theta)
    lon_model = (np.degrees(phi) + 360) % 360

    # === Transform Cartesian velocity to east/north ===
    r_hat = np.vstack([x, y, z]) / r
    e_phi = np.vstack([-y, x, np.zeros_like(x)]) / np.sqrt(x**2 + y**2)
    e_theta = np.vstack([
        x*z, y*z, -(x**2 + y**2)
    ]) / (r * np.sqrt(x**2 + y**2))
    v_cart = np.vstack([vx_model, vy_model, vz_model])

    u_east = np.sum(v_cart * e_phi, axis=0)
    u_north = np.sum(v_cart * e_theta, axis=0)

    # === Match Morell points to nearest model point ===
    model_coords = np.column_stack((lat_model, lon_model))
    morell_coords = np.column_stack((lat, lon))
    tree = cKDTree(model_coords)
    _, indices = tree.query(morell_coords)

    matched_u_east = u_east[indices] 
    matched_u_north = u_north[indices] 

    # === Plot ===
    pltname = f"{plot_loc}morell_velocity_comparison_{ts}.png"
    fig = plt.figure(figsize=(8, 10), dpi=300)
    projection = ccrs.Orthographic(central_longitude=-60, central_latitude=-15)
    ax = plt.axes(projection=projection)

    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    norm = Normalize(vmin=velocity.min(), vmax=velocity.max())
    cmap = cm.get_cmap('viridis')

    # Plot Morell data
    scatter = ax.scatter(lon, lat, c=velocity, cmap=cmap, norm=norm,
                        transform=ccrs.PlateCarree(), zorder=3, s=10)
    ax.quiver(lon, lat, vx, vy, transform=ccrs.PlateCarree(), 
            scale=40, width=0.005, color='red', zorder=4, label='Morell', alpha =0.5)

    # Plot model velocity vectors matched to Morell locations
    ax.quiver(lon, lat, matched_u_east, matched_u_north, transform=ccrs.PlateCarree(), 
            scale=40, width=0.005, color='black', alpha=1, zorder=4, label='Modelled')
    
    # Add 2 cm/yr reference vector (0.02 m/yr) in bottom left
    # Reference arrow coordinates and vector (2 cm/yr = 0.02 m/yr)
    ref_lon, ref_lat = 270, -58
    u_ref, v_ref = 2, 0.0  # Eastward arrow

    # Plot the reference vector
    ax.quiver(np.array([ref_lon]), np.array([ref_lat]),
            np.array([u_ref]), np.array([v_ref]),
            transform=ccrs.PlateCarree(),
            scale=40, width=0.005, color='black', zorder=5)

    # Add label for the reference vector
    ax.text(ref_lon + 1, ref_lat, '2 cm/yr', color ="black" , transform=ccrs.PlateCarree(),
            fontsize=12, ha='left', va='center')
    
    # Add gridlines and ticks
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cticker.LongitudeFormatter()  # Format for longitude
    gl.yformatter = cticker.LatitudeFormatter()   # Format for latitude
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xlocator = mticker.FixedLocator(np.arange(270, 311, 10))  # Longitude range
    gl.ylocator = mticker.FixedLocator(np.arange(-60, 21, 10))    # Latitude range
    gl.ylabels_left = True  # Show left latitude labels
    gl.xlabels_bottom = True  # Show bottom longitude labels
    gl.draw_labels = True

    ax.set_extent([270, 310, -60, 20], crs=ccrs.PlateCarree())
    ax.set_title('Surface Velocity Comparison', fontsize=18)
    ax.legend(loc='upper right', fontsize=14)
    plt.text(0.95, 0.01, f"Time: {round(time_array[ts,1]/1.e6, 2)} Myr",
         fontsize=16, transform=ax.transAxes, ha='right', va='bottom')
        
    plt.tight_layout()
    plt.savefig(pltname, dpi=500)
    plt.close(fig)

