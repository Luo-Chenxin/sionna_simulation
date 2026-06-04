from pyproj import Transformer, CRS
import mitsuba as mi
import drjit as dr
import numpy as np
import pandas as pd
import sionna.rt as rt
from config import LON_ORIGIN, LAT_ORIGIN, ALT_ORIGIN, DEFAULT_AZIMUTH_MUTIPLITER, DEFAULT_OFF_SET

# Determine the corresponding UTM (Universal Transverse Mercator) zone based on longitude
_ZONE = int((LON_ORIGIN + 180) / 6) + 1
_IS_SOUTH = LAT_ORIGIN < 0
UTM_CRS = CRS.from_dict({
        'proj': 'utm', 
        'zone': _ZONE,
        'south': _IS_SOUTH
    })

def get_terrain_z_batch(xs, ys, terrain_filename):
    """
    Optional: Get the highest Z coordinate of the terrain for given X and Y arrays.
    
    Inputs:
        xs (np.ndarray): 1D array of X coordinates.
        ys (np.ndarray): 1D array of Y coordinates.
        terrain_filename: Terrain xml filename
        
    Returns:
        np.ndarray: 1D array of Z coordinates.
    """
    # 1. Convert NumPy arrays to Mitsuba/Dr.Jit float arrays
    x_jit = mi.Float(xs)
    y_jit = mi.Float(ys)
    
    # Get the total number of input points
    num_points = dr.shape(x_jit)[0]
    
    # 2. Set ray start position (Origin)
    # Put Z at 10000.0 so the ray starts high in the sky
    z_start = 10000.0
    o_x = x_jit
    o_y = y_jit
    o_z = dr.opaque(mi.Float, z_start, shape=num_points)
    ray_o = mi.Vector3f(o_x, o_y, o_z)
    
    # 3. Set ray direction: shoot straight down (0, 0, -1)
    d_x = dr.zeros(mi.Float, shape=num_points)
    d_y = dr.zeros(mi.Float, shape=num_points)
    d_z = dr.opaque(mi.Float, -1.0, shape=num_points)
    ray_d = mi.Vector3f(d_x, d_y, d_z)
    
    # 4. Create the batch of rays
    rays = mi.Ray3f(ray_o, ray_d)
    
    # 5. Call the terrain scene to find intersections
    terrain_scene = rt.load_scene(filename=terrain_filename)
    si = terrain_scene.mi_scene.ray_intersect(rays)
    
    # 6. Get the Z coordinate of the intersection point
    z_jit = si.p.z
    
    # 7. If a ray misses the terrain, replace its Z value with NaN
    nan_value = dr.opaque(mi.Float, float('nan'), shape=num_points)
    z_jit = dr.select(si.is_valid(), z_jit, nan_value)
    
    # 8. Run the calculation and convert back to a NumPy array
    dr.eval(z_jit)
    return np.array(z_jit)

def latlonh_to_xyz_batch(
    lat: pd.Series | float, lon: pd.Series | float, h: pd.Series | float, terrain_filename
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """
    Batch convert geographic coordinates (Latitude, Longitude, height) to
    the local Cartesian coordinates (x, y, z) of the scene.
    The positive x-axis direction represents the direction of increasing longitude, 
    the positive y-axis direction represents the direction of increasing latitude, 
    and the positive z-axis direction represents the direction of increasing altitude.

    Returns np.ndarray if inputs are Pandas Series, otherwise returns float.
    """
    transformer = Transformer.from_crs("epsg:4326", UTM_CRS, always_xy=True)
    x_origin, y_origin = transformer.transform(LON_ORIGIN, LAT_ORIGIN)

    # Convert potential Series inputs to NumPy arrays to ensure consistent output type
    lat_arr = np.asarray(lat)
    lon_arr = np.asarray(lon)
    h_arr = np.asarray(h)

    # pyproj transform natively accepts NumPy arrays
    x_objs, y_objs = transformer.transform(lon_arr, lat_arr)

    # Calculate relative coordinates based on the scene's origin center
    xs = x_objs - x_origin
    ys = y_objs - y_origin
    if terrain_filename is None:
        zs = h_arr - ALT_ORIGIN
    else:
        zs = h_arr + get_terrain_z_batch(xs, ys, terrain_filename)

    # If the original input was a float, extract the scalar from the 0-d/1-d array
    if isinstance(lat, float) or isinstance(lat, int):
        return float(xs), float(ys), float(zs)

    # Otherwise, ensure it returns a standard NumPy array
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def deflected_azimuth(
    azimuth: pd.Series | float,
    multiplier: float = DEFAULT_AZIMUTH_MUTIPLITER,
    offset: float = DEFAULT_OFF_SET,
) -> np.ndarray | float:
    """
    Apply multiplication and addition to the input azimuth.

    Returns np.ndarray if azimuth is a Pandas Series, otherwise returns float.
    """
    # If the input is a single float, perform scalar math and return a float
    if isinstance(azimuth, (float, int)):
        return azimuth * multiplier + offset

    # If the input is a Pandas Series (or any array-like), convert it to
    # a NumPy array first to ensure the return type is np.ndarray.
    az_arr = np.asarray(azimuth)

    return az_arr * multiplier + offset