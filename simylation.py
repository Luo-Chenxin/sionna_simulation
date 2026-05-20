import os
import pandas as pd
import sionna.rt
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMapSolver
from pyproj import Transformer, CRS

LAT_MAX, LAT_MIN = 48.7409, 48.7171
LON_MIN, LON_MAX = 2.2451, 2.3013
LAT_ORIGIN = (LAT_MAX + LAT_MIN)/2
LON_ORIGIN = (LON_MIN + LON_MAX)/2
ALT_ORIGIN = 0
_ZONE = int((LON_ORIGIN + 180) / 6) + 1
_IS_SOUTH = LAT_ORIGIN < 0
UTM_CRS = CRS.from_dict({
        'proj': 'utm', 
        'zone': _ZONE,
        'south': _IS_SOUTH
    })
SCENE_FILE_PATH = 'blender/massy.xml'
TRANSMITTER_FILE_PATH = 'data/transimitters/3490.0_3570.0_mhz.csv'
TRANSMITTER_FILE_ENCODE = 'utf-8-sig'
RECEIVE_FILE_PATH = 'data/sensor_location.csv'
RECEIVE_FILE_ENCODE = 'utf-8'
FREQUENCE = 3530.0
FREQUENCE_UNIT = 'mhz'
DISPLAY_RADIUS = 10

def latlon_to_xyz(lat, lon, alt=50):
    """
    Convert [latitude, longitude, altitude] points into a local rectangular coordinate system (x, y, z) 
    relative to a specified origin[LAT_ORIGIN, LON_ORIGIN, ALT_ORIGIN], with units in meters
    """
    transformer = Transformer.from_crs("epsg:4326", UTM_CRS, always_xy=True)
    y_orign, x_orign = transformer.transform(LON_ORIGIN, LAT_ORIGIN)
    y_obj, x_obj = transformer.transform(lon, lat)
    
    x = x_orign - x_obj
    y = y_obj - y_orign
    z = alt - ALT_ORIGIN
    
    return (x, y, z)

def get_tx_name(id):
    return f'tx_{id}'

def get_rx_name(id):
    return f'rx_{id}'

def add_tx(scene, name, position, orientation=(0,0,0)):
    # TODO: power_dbm?
    tx = Transmitter(name=name, position=position, orientation=orientation, display_radius=DISPLAY_RADIUS)
    scene.add(tx)

def add_txs(scene, df_tx:pd.DataFrame):
    for _, row in df_tx.iterrows():
        name = get_tx_name(row['ID'])
        lat, lon = row['Latitude'], row['Longitude']
        # TODO: alt?  orientation?
        add_tx(scene, name, latlon_to_xyz(lat, lon))

def add_rx(scene, name, position):
    rx = Receiver(name=name, position=position, display_radius=DISPLAY_RADIUS)
    scene.add(rx)

def add_rxs(scene, df_rx:pd.DataFrame):
    for _, row in df_rx.iterrows():
        name = get_rx_name(row['sensor_id'])
        lat, lon = row['Latitude'], row['Longitude']
        # TODO: alt?
        add_rx(scene, name, latlon_to_xyz(lat, lon))

def set_frequence(scene, frequence=FREQUENCE, unit:str=FREQUENCE_UNIT):
    if unit.lower() == 'mhz':
        scene.frequency = frequence * 1e6
    elif unit.lower() == 'ghz':
        scene.frequency = frequence * 1e9
    else:
        raise ValueError(f"Unsupported frequency unit: {unit}")
    
def set_config(filename=SCENE_FILE_PATH) -> sionna.rt.Scene:
    scene = sionna.rt.load_scene(filename=filename)

    # TODO:Configure antenna array for all transmitters and receivers
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                pattern="iso",
                                polarization="V")
    scene.rx_array = scene.tx_array

    df_tx = pd.read_csv(TRANSMITTER_FILE_PATH, encoding=TRANSMITTER_FILE_ENCODE)
    df_rx = pd.read_csv(RECEIVE_FILE_PATH, encoding=RECEIVE_FILE_ENCODE)

    # TODO: Some itu_materials are not defined in some frequences
    # TODO: set_frequence()
    add_txs(scene, df_tx)
    add_rxs(scene, df_rx)
    return scene

def get_radio_map_local(scene):
    # TODO: parameters add
    rm_solver = RadioMapSolver()

    rm = rm_solver(scene,
        max_depth=5,           # Maximum number of ray scene interactions
        samples_per_tx=10**7 , # If you increase: less noise, but more memory required
        cell_size=(5, 5),      # Resolution of the radio map
        center=[0, 0, 0],      # Center of the radio map
        size=[5000, 5000],       # Total size of the radio map
        orientation=[0, 0, 0]) # Orientation of the radio map, e.g., could be also vertical
    
    # Visualize path gain
    rm.show(metric="path_gain")

    # Visualize received signal strength (RSS)
    rm.show(metric="rss")

    # Visulaize SINR
    rm.show(metric="sinr")

def get_radio_map_general(scene):
    # TODO: parameters add
    rm_solver = RadioMapSolver()

    mesurement_surface = scene.objects["terrain"].clone(as_mesh=True)

    rm = rm_solver(scene,
        measurement_surface=mesurement_surface,
        samples_per_tx=10**8,
        max_depth=5)
    
    # scene.preview(radio_map=rm, rm_vmin=-200)

if __name__ == "__main__":
    scene = set_config()
    get_radio_map_general(scene)
    # TODO: save output