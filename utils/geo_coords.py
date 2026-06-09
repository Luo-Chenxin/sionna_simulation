from pyproj import Transformer, CRS
import mitsuba as mi
import drjit as dr
import numpy as np
import pandas as pd
import sionna.rt as rt
from config import DEFAULT_AZIMUTH_MUTIPLITER, DEFAULT_OFF_SET

def _get_terrain_z_batch(
        xs: np.ndarray,
        ys: np.ndarray,
        terrain_scene: rt.Scene,
    ) -> np.ndarray:
        """
        Get terrain Z for each XY point.
        """
        x_jit = mi.Float(xs)
        y_jit = mi.Float(ys)

        num_points = dr.shape(x_jit)[0]

        z_start = 10000.0
        o_x = x_jit
        o_y = y_jit
        o_z = dr.opaque(mi.Float, z_start, shape=num_points)
        ray_o = mi.Vector3f(o_x, o_y, o_z)

        d_x = dr.zeros(mi.Float, shape=num_points)
        d_y = dr.zeros(mi.Float, shape=num_points)
        d_z = dr.opaque(mi.Float, -1.0, shape=num_points)
        ray_d = mi.Vector3f(d_x, d_y, d_z)

        rays = mi.Ray3f(ray_o, ray_d)

        si = terrain_scene.mi_scene.ray_intersect(rays)

        z_jit = si.p.z

        nan_value = dr.opaque(mi.Float, float('nan'), shape=num_points)
        z_jit = dr.select(si.is_valid(), z_jit, nan_value)

        dr.eval(z_jit)
        return np.array(z_jit)

class SceneCoordinateConverter:
    """
    Convert Paris latitude, longitude and height to local XYZ.
    """
    def __init__(
            self, 
            lat_origin: float, 
            lon_origin: float, 
            alt_origin: float,
            original_crs: CRS,
            target_crs: CRS):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.alt_origin = alt_origin
        
        # Create transformer once for speed.
        self._transformer = Transformer.from_crs(original_crs, target_crs, always_xy=True)
        
        # Save the scene origin in projected (Lambert-93 coordinates) meters
        self._x_origin, self._y_origin = self._transformer.transform(
            self.lon_origin, self.lat_origin
        )

    def latlonh_to_xyz_batch(
        self,
        lat: pd.Series | float,
        lon: pd.Series | float,
        h: pd.Series | float,
        terrain_scene: rt.Scene | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[float, float, float]:
        """
        Convert (Lat, Lon, Height) to local Cartesian coordinates (X, Y, Z).
        
        +X: East (increasing longitude)
        +Y: North (increasing latitude)
        +Z: Up (increasing altitude)
        """
        # Check if the input is a single number (scalar)
        is_scalar = np.isscalar(lat) and np.isscalar(lon) and np.isscalar(h)

        lat_arr = np.asarray(lat, dtype=float)
        lon_arr = np.asarray(lon, dtype=float)
        h_arr = np.asarray(h, dtype=float)

        # Reuse the internal transformer to save CPU time
        x_objs, y_objs = self._transformer.transform(lon_arr, lat_arr)

        xs = x_objs - self._x_origin
        ys = y_objs - self._y_origin

        if terrain_scene is None:
            zs = h_arr - self.alt_origin
        else:
            terrain_z = _get_terrain_z_batch(xs, ys, terrain_scene)
            zs = h_arr + terrain_z

        # If input is a single number, return floats
        if is_scalar:
            return float(xs), float(ys), float(zs)

        # Otherwise, return NumPy arrays
        return xs, ys, zs

def deflected_azimuth(
    azimuth: pd.Series | float,
    multiplier: float = DEFAULT_AZIMUTH_MUTIPLITER,
    offset: float = DEFAULT_OFF_SET,
) -> np.ndarray | float:
    """
    Multiply and add to azimuth input.

    Return np.ndarray for array input or float for scalar input.
    """
    # If the input is a single float, perform scalar math and return a float
    if isinstance(azimuth, float):
        return azimuth * multiplier + offset

    # If the input is a Pandas Series (or any array-like), convert it to
    # a NumPy array first to ensure the return type is np.ndarray.
    az_arr = np.asarray(azimuth)

    return az_arr * multiplier + offset