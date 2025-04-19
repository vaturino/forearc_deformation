#! /usr/bin/python3
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import math
import matplotlib.pyplot as plt
import sys, os, subprocess

def grab_dimTime_particles(dir: str, stat:str, time, skiprows = 15):
    ts = len(os.listdir(dir))
    for t in range(ts):
        time[t,0] = t
    filt = stat.loc[:,skiprows].notnull()
    time[:,1] = stat[filt].loc[:,1]
    return time

def initial_particle_position (data: str, x: float, y: float):
    rows = []
    for i in range(len(data["initial position:0"])):
        rows.append([math.sqrt(pow((data["initial position:0"][i] - x), 2) + pow((data["initial position:1"][i] - y),2))])
    df = pd.DataFrame(rows, columns=["distance"])
    idx = df["distance"].idxmin()
    return(data["initial position:0"][idx], data["initial position:1"][idx])

def initial_particle_index (data: str, x: float, y: float):
    rows = []
    for i in range(len(data["initial position:0"])):
        rows.append([math.sqrt(pow((data["initial position:0"][i] - x), 2) + pow((data["initial position:1"][i] - y),2))])
    df = pd.DataFrame(rows, columns=["distance"])
    idx = df["distance"].idxmin()
    return(idx)




def load_dataset(loc_data:str, data: str):
    if {"initial oc","initial sed"}.issubset(data.columns):
        df = pd.read_parquet(loc_data, columns=["initial position:0", "initial position:1", "Points:0", "Points:1", "initial oc","initial sed", "id"])
    elif {"initial oc","initial serp"}.issubset(data.columns):
        df = pd.read_parquet(loc_data, columns=["initial position:0", "initial position:1", "Points:0", "Points:1", "initial oc","initial serp", "id"])
    elif "initial oc" in data.columns:
        df = pd.read_parquet(loc_data, columns=["initial position:0", "initial position:1", "Points:0", "Points:1", "initial oc", "id"])
    return (df)



def get_incoming_particles(data: str, compo: float, from_trench: float, samples: float):
    if {"initial oc","initial sed"}.issubset(data.columns):
        if data["initial oc"].max() == 1 or data["initial sed"].max() == 1:
            sumcomp = data["initial oc"] + data["initial sed"]
            part = data[(sumcomp >= compo) & (data["Points:0"] <= from_trench + 0.5e3) & (data["Points:0"] >= from_trench - 0.5e3)]
            part = part.sample(n = samples).reset_index()
        # sumcomp = data["initial oc"] + data["initial sed"]
        # part = data[(sumcomp >= compo) & (data["Points:0"] <= from_trench + 2.e3) & (data["Points:0"] >= from_trench - 2.e3)]
        # part = part.sample(n = samples).reset_index()
    elif {"initial oc","initial serp"}.issubset(data.columns):
        if data["initial oc"].max() == 1 or data["initial serp"].max() == 1:
            sumcomp = data["initial oc"] + data["initial serp"]
            part = data[(sumcomp >= compo) & (data["Points:0"] <= from_trench + 2.e3) & (data["Points:0"] >= from_trench - 2.e3)]
            part = part.sample(n = samples).reset_index()
        # sumcomp = data["initial oc"] + data["initial serp"]
        # part = data[(sumcomp >= compo) & (data["Points:0"] <= from_trench + 2.e3) & (data["Points:0"] >= from_trench - 2.e3)]
        # part = part.sample(n = samples).reset_index()
    elif "initial oc" in data.columns:
        if data["initial oc"].max() == 1:
            part = data[(data["initial oc"] >= compo) & (data["Points:0"] <= from_trench + 2.e3) & (data["Points:0"] >= from_trench - 2.e3)]
            part = part.sample(n = samples).reset_index()
        # part = data[(data["initial oc"] >= compo) & (data["Points:0"] <= from_trench + 2.e3) & (data["Points:0"] >= from_trench - 2.e3)]
        # part = part.sample(n = samples).reset_index()
    return (part)

def get_incoming_pos(data: str, compo: float, from_trench: float, samples: float):
    if "initial C_1" in data.columns:
        part = data[(data["initial C_1"] >= compo) & (data["Points:0"] <= from_trench + 2.e3) & (data["Points:0"] >= from_trench - 2.e3)]
        part = part.sample(n = samples).reset_index()
    elif {"initial ocean_crust","initial upper_cont","initial lower_cont"}.issubset(data.columns):
        sumcomp = data["initial ocean_crust"] + data["initial upper_cont"] + data["initial lower_cont"]
        part = data[(sumcomp >= compo) & (data["Points:0"] <= from_trench + 2.e3) & (data["Points:0"] >= from_trench - 2.e3)]
        part = part.sample(n = samples).reset_index()
    else:
        sumcomp = data["initial ocean_crust"]
        part = data[(sumcomp >= compo) & (data["Points:0"] <= from_trench + 2.e3) & (data["Points:0"] >= from_trench - 2.e3)]
        part = part.sample(n = samples).reset_index()
    return (part["initial position:0"], part["initial position:1"])

def get_crust_particles(data: str, compo: float):
    if "initial C_1" in data.columns:
        part = data[(data["initial C_1"] >= compo)]
    elif {"initial ocean_crust","initial upper_cont","initial lower_cont"}.issubset(data.columns):
        sumcomp = data["initial ocean_crust"] + data["initial upper_cont"] + data["initial lower_cont"]
        part = data[(sumcomp >= compo)]
    else:
        sumcomp = data["initial ocean_crust"]
        part = data[(sumcomp >= compo)]
    return part



def new_initial_particle (data: str, x: float, y: float):
    rows = []
    for i in range(len(data["Points:0"])):
        rows.append([math.sqrt(pow((data["Points:0"][i] - x), 2) + pow((data["Points:1"][i] - y),2))])
    df = pd.DataFrame(rows, columns=["distance"])
    idx = df["distance"].idxmin()
    return(data["initial position:0"][idx], data["initial position:1"][idx])


def get_particle (data:float , x: float, y: float):
    # idx = data.index.values[(data['initial position:0']==x) & (data['initial position:1']==y)][0]
    # return(data['Points:0'][idx], data['Points:1'][idx])
    p = (data["initial position:0"] == x) 
    part = data[p]
    fil = (part["initial position:1"] == y)
    idx = part[fil].index.values[0]
    return(data['Points:0'][idx], data['Points:1'][idx])


def get_particle_vy (data:float , x: float, y: float):
    # idx = data.index.values[(data['initial position:0']==x) & (data['initial position:1']==y)][0]
    # return(data['Points:0'][idx], data['Points:1'][idx])
    p = (data["initial position:0"] == x) 
    part = data[p]
    fil = (part["initial position:1"] == part["initial position:1"].max() - y)
    idx = part[fil].index.values[0]
    return(data['velocity:1'][idx])

def get_particle_vy_old (data:float , x: float, y: float):
    # idx = data.index.values[(data['initial position:0']==x) & (data['initial position:1']==y)][0]
    # return(data['Points:0'][idx], data['Points:1'][idx])
    p = (data["initial position:0"] == x) 
    part = data[p]
    fil = (part["initial position:1"] == y)
    idx = part[fil].index.values[0]
    return(data['velocity:1'][idx])



# def get_exhumed (data:float , x: float, len: float, y: float, ylim: float):
#     fil = (data["initial position:0"] >= x) & (data["initial position:0"] <= x + len) & (data["initial position:1"] == data["initial position:1"].max() - y)
#     top = data[fil]
#     pos = (np.sign(top["velocity:1"])) & (top['position:1'] >= data["initial position:1"].max() - ylim)
#     df = pd.DataFrame(top[pos])
#     return(df)


def get_particle_PT(data: str, x: float, y: float):
    idx = data.index.values[(data['initial position:0']==x) & (data['initial position:1']==y)][0]
    return(data['p'][idx], data['T'][idx])


# def get_particle_PT(data: str, x: float, y: float):
#     p = (data["initial position:0"] == x) 
#     part = data[p]
#     fil = (part["initial position:1"] == part["initial position:1"].max() - y)
#     idx = part[fil].index.values[0]
#     return(data['p'][idx], data['T'][idx])



def check_particle(data: str, x: float, y: float):
    idx = data.index.values[(data['initial position:0']==x) & (data['initial position:1']==y)][0]
    return(data["initial position:0"][idx], data["initial position:1"][idx], data["Points:0"][idx], data["Points:1"][idx])




# 3D particles

def initial_particle3d_position (data: str, x: float, y: float, z: float):
    rows = []
    for i in range(len(data["initial position:0"])):
        rows.append([math.sqrt(pow((data["initial position:0"][i] - x), 2) + pow((data["initial position:1"][i] - y),2) + pow((data["initial position:2"][i] - z),2))])
    df = pd.DataFrame(rows, columns=["distance"])
    idx = df["distance"].idxmin()
    return(data["initial position:0"][idx], data["initial position:1"][idx], data["initial position:2"][idx])


def new_initial_particle3d (data: str, x: float, y: float, z: float):
    rows = []
    for i in range(len(data["Points:0"])):
        rows.append([math.sqrt(pow((data["Points:0"][i] - x), 2) + pow((data["Points:1"][i] - y),2) + pow((data["Points:2"][i] - z),2))])
    df = pd.DataFrame(rows, columns=["distance"])
    idx = df["distance"].idxmin()
    return(data["initial position:0"][idx], data["initial position:1"][idx], data["initial position:2"][idx])


def get_particle3d (data: str, x: float, y: float, z: float):
    idx = data.index.values[(data['initial position:0']==x) & (data['initial position:1']==y) & (data['initial position:2']==z)][0]
    return(data['Points:0'][idx], data['Points:1'][idx], data['Points:2'][idx])


def get_particle3d_PT(data: str, x: float, y: float, z: float):
    idx = data.index.values[(data['initial position:0']==x) & (data['initial position:1']==y) & (data['initial position:2']==z)][0]
    return(data['p'][idx], data['T'][idx])




