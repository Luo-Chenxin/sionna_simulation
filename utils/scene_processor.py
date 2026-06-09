import numpy as np
import osmnx as ox
from rasterio import transform, features
from config import LocalCRS

class SceneProcessor:
    def __init__(self, block_meta, resolution_m):
        """
        Initialize processor with block metadata object.
        """
        self.meta = block_meta
        self.resolution = resolution_m
        
        # Total size including overlap in meters
        total_width_m = self.meta.x_end - self.meta.x_start
        total_height_m = self.meta.y_end - self.meta.y_start
        
        # Total matrix size in pixels
        self.total_w = int(np.ceil(total_width_m / self.resolution))
        self.total_h = int(np.ceil(total_height_m / self.resolution))
        
        # Calculate crop pixel indices for the core area (no overlap)
        # Convert overlap from meters to exact pixels
        self.overlap_pixels = int(self.meta.over_lap_m / self.resolution)
        
        # Core width and height in pixels
        self.core_w = int(self.meta.block_size_m / self.resolution)
        self.core_h = int(self.meta.block_size_m / self.resolution)

    def export_3d_scene_xml(self, output_dir):
        """Export 3D scene (XML and PLY).
        
        This uses the wide area with overlap to catch reflection buildings.
        """
        # Pass these lat/lon bounds directly to your OSM-to-Sionna exporter
        # lat_min = self.meta.lat_min
        # lat_max = self.meta.lat_max
        # lon_min = self.meta.lon_min
        # lon_max = self.meta.lon_max
        pass

    def generate_overlap_building_mask(self):
        """Download OSM buildings and convert to 2D numpy mask (with overlap)"""
        # Download buildings using the wide bounds from block_meta
        try:
            gdf = ox.features_from_bbox(
                bbox = (
                    self.meta.lon_min,  # left
                    self.meta.lat_min,  # bottom
                    self.meta.lon_max,  # right
                    self.meta.lat_max   # top
                ),
                tags={"building": True}
            )
        except Exception:
            # Return empty matrix if no buildings found
            return np.zeros((self.total_h, self.total_w), dtype=np.uint8)

        # Project to Lambert-93
        gdf_projected = gdf.to_crs(LocalCRS.FRANCE_LAMBERT93.crs)
        
        # Create transform matrix for rasterization
        # Note: Image starts from Top-Left corner (x_start, y_end)
        transform_matrix = transform.from_origin(
            west=self.meta.x_start, 
            north=self.meta.y_end, 
            xsize=self.resolution, 
            ysize=self.resolution
        )
        
        # Burn polygons into the total matrix
        full_mask = features.rasterize(
            shapes=gdf_projected.geometry,
            out_shape=(self.total_h, self.total_w),
            transform=transform_matrix,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
        
        return full_mask

    def crop_to_core(self, total_matrix):
        """Cut off the overlap border

        This changes full matrix to core matrix.
        Works for building_mask, transmitters_mask, and radiomap.
        """
        p = self.overlap_pixels
        
        # Get the core pixels by slicing rows and columns
        core_matrix = total_matrix[p : p + self.core_h, p : p + self.core_w]
        
        return core_matrix