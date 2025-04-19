#! /usr/bin/python3
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import math
import matplotlib.pyplot as plt
import sys, os, subprocess
from scipy.signal import savgol_filter


def load_parquet_file(file_path):
    df = pd.read_parquet(file_path)
    return df



def grab_dimTime_timeStep (t:float, stat:str, model_output_dt  = 50, num_header_lines = 15):
    statistics = pd.read_csv(stat,skiprows=num_header_lines,sep='\s+',header=None)
    statistics = statistics.loc[::model_output_dt, [0,1]]
    statistics.columns = ['ts','dimtime']
    statistics['ts'] = range(len(statistics))
    closest_index = statistics['dimtime'].sub(t).abs().idxmin()
    dt, ts = statistics.loc[closest_index, ['dimtime', 'ts']]
    return (dt/1e+06, ts)

def grab_dimTime_fields (dir: str, stat:str, time, skiprows = 15):
    ts = len(os.listdir(dir))
    for t in range(ts):
        time[t,0] = t
    filt = stat.loc[:,skiprows].notnull()
    time[:,1] = stat[filt].loc[:,1]
    return time

def grab_dimensional_time (stat:str, time):
    filt = stat.loc[:,20].notnull()
    time[:] = stat[filt].loc[:,1]
    return time

def grab_dimTime_cont (dir: str, stat:str, time, i: int):
    ts = len(os.listdir(dir))
    # print(ts)
    for t in range(ts):
        time[t,0] = t
    filt = stat.loc[:,i].notnull()
    time[:,1] = stat[filt].loc[:,1]
    return time


def create_grid_velocities_crust (xmin_plot:float, xmax_plot:float, ymin_plot:float, ymax_plot:float, grid_res:float, grid_low_res:float, grid_high_res: float)  :
    # create grid to interpolate stuff onto (for plotting)
    x_low = np.linspace(xmin_plot,xmax_plot,int((xmax_plot-xmin_plot)/grid_res))
    y_low =  np.linspace(ymin_plot,ymax_plot,int((ymax_plot-ymin_plot)/grid_res))
    X_low, Y_low = np.meshgrid(x_low,y_low)
    # lower res grid for velocities
    x_vels = np.linspace(xmin_plot,xmax_plot,int((xmax_plot-xmin_plot)/grid_low_res))
    y_vels =  np.linspace(ymin_plot,ymax_plot,int((ymax_plot-ymin_plot)/grid_low_res))
    X_vels, Y_vels = np.meshgrid(x_vels,y_vels)
    # higher res grid for crust
    x_crust = np.linspace(xmin_plot,xmax_plot,int((xmax_plot-xmin_plot)/grid_high_res))
    y_crust =  np.linspace(ymin_plot,ymax_plot,int((ymax_plot-ymin_plot)/grid_high_res))
    X_crust, Y_crust = np.meshgrid(x_crust,y_crust)
    return (X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust)

def create_grid_velocities_crust_hor (xmin_plot:float, xmax_plot:float, zmin_plot:float, zmax_plot:float, grid_res:float, grid_low_res:float, grid_high_res: float)  :
    # create grid to interpolate stuff onto (for plotting)
    x_low = np.linspace(xmin_plot,xmax_plot,int((xmax_plot-xmin_plot)/grid_res))
    z_low =  np.linspace(zmin_plot,zmax_plot,int((zmax_plot-zmin_plot)/grid_res))
    X_low, Z_low = np.meshgrid(x_low,z_low)
    # lower res grid for velocities
    x_vels = np.linspace(xmin_plot,xmax_plot,int((xmax_plot-xmin_plot)/grid_low_res))
    z_vels =  np.linspace(zmin_plot,zmax_plot,int((zmax_plot-zmin_plot)/grid_low_res))
    X_vels, Z_vels = np.meshgrid(x_vels,z_vels)
    # higher res grid for crust
    x_crust = np.linspace(xmin_plot,xmax_plot,int((xmax_plot-xmin_plot)/grid_high_res))
    z_crust =  np.linspace(zmin_plot,zmax_plot,int((zmax_plot-zmin_plot)/grid_high_res))
    X_crust, Z_crust = np.meshgrid(x_crust,z_crust)
    return (X_low, Z_low, X_vels, Z_vels, X_crust, Z_crust)

def interp_T_visc_vx_vz_compCrust (x, y, T, visc, vx, vz, C, X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust):
    Temp = griddata((x, y), T-273,    (X_low, Y_low), method='cubic')
    Visc = griddata((x, y), visc, (X_low, Y_low), method='linear') 
    Vx = griddata((x, y), vx,   (X_vels, Y_vels), method='cubic')
    Vz = griddata((x, y), vz,   (X_vels, Y_vels), method='cubic')
    Comp = griddata((x, y), C,   (X_crust, Y_crust), method='cubic')
    return (Temp, Visc, Vx, Vz, Comp)


def interp_T_visc_vel_comp (x, y, T, visc, v, C, X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust):
    Temp = griddata((x, y), T-273,    (X_low, Y_low), method='cubic')
    Visc = griddata((x, y), visc, (X_low, Y_low), method='linear') 
    V = griddata((x, y), v,   (X_crust, Y_crust), method='cubic')
    Comp = griddata((x, y), C,   (X_crust, Y_crust), method='cubic')
    return (Temp, Visc,V, Comp)


def interp_press (x, y, p, X_low, Y_low):
    press = griddata((x, y), p,   (X_low, Y_low), method='cubic')
    return (press)

def interp_vmag_comp (x, y, v, C, X_crust, Y_crust):
    V = griddata((x, y), v,   (X_crust, Y_crust), method='cubic')
    Comp = griddata((x, y), C,   (X_crust, Y_crust), method='cubic')
    return (V, Comp)


def get_crust(contour):    
    conts = len(contour.collections[0].get_paths())
    j = 0
    # for i in range(conts):
    #     if len(contour.collections[0].get_paths()[j]) < len(contour.collections[0].get_paths()[i]):
    #         j = i
    pts = contour.collections[0].get_paths()[j].vertices
    return pts


def interp_T_visc_vx_vz_comp_continent (x, y, T, visc, vx, vz, oc, uc, lc, lith, X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust):
    Temp = griddata((x, y), T-273,    (X_low, Y_low), method='cubic')
    Visc = griddata((x, y), visc, (X_low, Y_low), method='linear') 
    Vx = griddata((x, y), vx,   (X_vels, Y_vels), method='cubic')
    Vz = griddata((x, y), vz,   (X_vels, Y_vels), method='cubic')
    Oc = griddata((x, y), oc,   (X_crust, Y_crust), method='cubic')
    Uc = griddata((x, y), uc,   (X_crust, Y_crust), method='cubic')
    Lc = griddata((x, y), lc,   (X_crust, Y_crust), method='cubic')
    Lith = griddata((x, y), lith,   (X_crust, Y_crust), method='cubic')
    return (Temp, Visc, Vx, Vz, Oc, Uc, Lc, Lith)




def interp_T_visc_vel_comp_tau (x, y, T, visc, v, C, t, X_low, Y_low, X_crust, Y_crust):
    Temp = griddata((x, y), T-273,    (X_low, Y_low), method='cubic')
    Visc = griddata((x, y), visc, (X_low, Y_low), method='linear') 
    V = griddata((x, y), v,   (X_crust, Y_crust), method='cubic')
    Comp = griddata((x, y), C,   (X_crust, Y_crust), method='cubic')
    tau = griddata((x, y), t,   (X_crust, Y_crust), method='cubic')
    return (Temp, Visc,V, Comp, tau)




def get_points_with_y_in(data, depth, delta, ymax = 900.e3): 
    plate_prof_loc = ymax - depth
    within_delta_from_depth = (data['Points:1'] < plate_prof_loc + delta) &(data['Points:1'] > plate_prof_loc - delta)
    return data[within_delta_from_depth].sort_values('Points:0').copy(True)

def get_points3d_with_z_in(data, depth, delta, zmax = 900.e3): 
    plate_prof_loc = zmax - depth
    within_delta_from_depth = (data['Points:2'] < plate_prof_loc + delta) &(data['Points:2'] > plate_prof_loc - delta)
    return data[within_delta_from_depth].sort_values('Points:1').copy(True)


def get_points3d_with_y_in(data, depth, delta, ymax = 1800.e3): 
    plate_prof_loc = ymax - depth
    within_delta_from_depth = (data['Points:1'] < plate_prof_loc + delta) &(data['Points:1'] > plate_prof_loc - delta)
    return data[within_delta_from_depth].sort_values('Points:0').copy(True)

def get_points3d_with_x_in(data, depth, delta): 
    plate_prof_loc = depth
    within_delta_from_depth = (data['Points:0'] < plate_prof_loc + delta) &(data['Points:0'] > plate_prof_loc - delta)
    return data[within_delta_from_depth].sort_values('Points:1').copy(True)




def get_trench_position(p, threshold = 0.3e7):
    if {"oc","sed"}.issubset(p.columns):
        sumcomp = p["oc"] + p["sed"]
        tr =  p.loc[(p['Points:0']> threshold) & (sumcomp > 0.3) & (p["Points:1"] >= p["Points:1"].max() - 5.e3),'Points:0'].max()
    elif {"oc","serp"}.issubset(p.columns):
        sumcomp = p["oc"] + p["serp"]
        tr =  p.loc[(p['Points:0']> threshold) & (sumcomp > 0.3) & (p["Points:1"] >= p["Points:1"].max() - 5.e3),'Points:0'].max()
    else:
        tr =  p.loc[(p['Points:0']> threshold) & (p['oc'] > 0.3) & (p["Points:1"] >= p["Points:1"].max() - 5.e3),'Points:0'].max()
    return tr

# def get_trench_position(p, threshold = 2.7e7):
#     if {"opc"}.issubset(p.columns):
#         tr =  p.loc[(p['Points:0']< threshold) & (p["opc"] > 0.3) & (p["Points:1"] >=  p["Points:1"].max() - 2.e3),'Points:0'].min()
#     else:
#         tr =  p.loc[(p['Points:0']< threshold) & (p['op'] > 0.3) & (p["Points:1"] >=  p["Points:1"].max() - 2.e3),'Points:0'].min()
#     return tr



    

def get_trench_position_cont(p, threshold = 0.3e7):
    tr =  p.loc[(p['Points:0']> threshold) & (p['sum_comp'] > 0.3),'Points:0'].max()
    return tr

def get_trench_x_v_sigma(p, threshold = 0.3e7):
    tr =  p.loc[(p['Points:0']> threshold) & (p['C_1'] > 0.5),'Points:0'].idxmax()
    return p.loc[tr][["Points:0", "velocity:0", "shear_stress_xx"]]

def get_trench_x_vx_vy_sigma(p,  threshold = 0.3e7):
    tr =  p.loc[(p['Points:0']> threshold) & (p['C_1'] > 0.3),'Points:0'].idxmax()
    return p.loc[tr][["Points:0", "velocity:0", "velocity:1", "shear_stress_xx"]]

def get_trench_x_y_vx_vy_sigma(p, C,  threshold = 0.3e7):
    if not len(p.loc[(p['Points:0']> threshold) & (p['C_1'] > C),'Points:0']):
        pass
    else:
        tr =  p.loc[(p['Points:0']> threshold) & (p['C_1'] > C),'Points:0'].idxmax()
        return p.loc[tr][["Points:0", "Points:1", "velocity:0", "velocity:1", "shear_stress_xx"]]
    
def convergence_rate(p,trench_point,distance_from_trench = 1.e6):
    left = p.loc[p['Points:0'] < trench_point - distance_from_trench,"velocity:0"].iloc[-1] 
    right = p.loc[p['Points:0'] > trench_point + distance_from_trench,"velocity:0"].iloc[0]
    return left - right
    
def horizontalStress(p,trench_point,distance_from_trench = 1.e6):
    right = p.loc[p['Points:0'] > trench_point + distance_from_trench,"shear_stress:0"].iloc[0]
    return right

def get_V_around_trench(p,trench_point,distance_from_trench = 1.e6):
    left = p.loc[p['Points:0'] < trench_point - distance_from_trench,"velocity:0"].iloc[-1]
    right = p.loc[p['Points:0'] > trench_point + distance_from_trench,"velocity:0"].iloc[0]
    return (left, right)

def get_Vy_around_trench(p,trench_point,distance_from_trench = 1.e6):
    left = p.loc[p['Points:0'] < trench_point - distance_from_trench,"velocity:1"].iloc[-1]
    right = p.loc[p['Points:0'] > trench_point + 1.e6,"velocity:1"].iloc[0]
    return (left, right)

def get_stress_around_trench(p,trench_point,distance_from_trench = 1.e6):
    left = p.loc[p['Points:0'] < trench_point - distance_from_trench,"shear_stress:0"].iloc[-1]
    right = p.loc[p['Points:0'] > trench_point + 1.e6,"shear_stress:0"].iloc[0]
    return (left, right)

def get_pos_around_trench(p,trench_point,distance_from_trench = 1.e6):
    left = p.loc[p['Points:0'] < trench_point - distance_from_trench,"Points:0"].iloc[-1]
    right = p.loc[p['Points:0'] > trench_point + 1.e6,"Points:0"].iloc[0]
    return (left, right)



def getMaxSlabDepth(contour):
    pts = contour.collections[0].get_paths()[0].vertices
    return pts[:,1].max()



def slab_surf_moho(contour, thresh: float):    
    conts = len(contour.collections[0].get_paths())
    j = 0
    for i in range(conts):
        if len(contour.collections[0].get_paths()[j]) < len(contour.collections[0].get_paths()[i]):
            j = i
    pts = contour.collections[0].get_paths()[j].vertices
    threshold_x = (pts[pts[:,1] > thresh]).min(0)[0]
    slab = pts[pts[:,0]> threshold_x]
    tip = slab[:,1].argmax()
    moho = slab[:tip, :]
    slab_surf = slab[tip:, :][::-1, :]
    return slab_surf, moho



def slab_surf_moho_continent(contour, thresh: float):    
    conts = len(contour.collections[0].get_paths())
    j = 0
    for i in range(conts):
        if len(contour.collections[0].get_paths()[j]) < len(contour.collections[0].get_paths()[i]):
            j = i
    pts = contour.collections[0].get_paths()[j].vertices
    threshold_x = (pts[pts[:,1] > thresh]).min(0)[0]
    slab = pts[pts[:,0]> threshold_x]
    tip = slab[:,1].argmax()
    moho = slab[:tip, :]
    slab_surf = slab[tip:, :][::-1, :]
    return slab_surf, moho




def get_slab_top(slab):
    threshold_x = (slab[slab[:,1] > 1000.]).min(0)[0]
    slab_tot = slab[slab[:,0]> threshold_x]
    tip = slab_tot[:,0].argmax()
    slab_top = slab_tot[tip:, :][::-1, :]
    return slab_top


def getDip(contour, d1 = 100., d2 = 200.):
    crust_points = contour.collections[0].get_paths()[0].vertices
    crust_points = savgol_filter(crust_points, 19,1, axis = 0)
    plt.close()

    cr_pt = pd.DataFrame(crust_points)
    last = len(cr_pt) - cr_pt[1].idxmax() -1
    cr_pt.drop(cr_pt.tail(last).index, inplace = True)

    closest_index1 = cr_pt[1].sub(d1).abs().idxmin()
    closest_index2 = cr_pt[1].sub(d2).abs().idxmin()

    if (cr_pt[1].sub(d1).abs().min() <= 2.) & (cr_pt[1].sub(d2).abs().min() <= 2.):
        x1, y1 = cr_pt.loc[closest_index1]
        x2, y2 = cr_pt.loc[closest_index2]
        Dx = abs(x1-x2)
        Dy = abs(y1-y2)
        dip = math.atan(Dy/Dx)
        return math.degrees(dip)

def getSlabDip(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    Dx = abs(x1-x2)
    Dy = abs(y1-y2)
    dip = math.atan(Dy/Dx)
    return math.degrees(dip)

def alongStrikeDip(p1,p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    Dx = abs(x1-x2)
    Dy = abs(y1-y2)
    dip = math.atan(Dx/Dy)
    return math.degrees(dip)


def project_slab_surface(x1:float, y1:float, d: float, theta:float):
    x = x1 - d*math.sin(theta)
    y = y1 + d*(math.cos(theta))
    return x, y


def project_slab_surface_continent(x1:float, y1:float, d: float):
    x = x1 - d
    y = y1 
    return x, y


def getWedgeTemp(p,slab_point,distance_from_slab):
    idx = p.loc[p['Points:0'] > slab_point + distance_from_slab, "Points:0"].idxmin()
    right = p.loc[idx, "T"]
    return right

def getWedgePt(p,trench_point,distance_from_trench):
    right = p.loc[p['Points:0'] > trench_point + distance_from_trench,"Points:0"].min()
    return right
