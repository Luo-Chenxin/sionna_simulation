import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from config import LocalCRS


class BuildingRasterizer:
    """
    Convert OSM building GeoDataFrame into a binary 2D raster matrix.
    Each cell represents [resolution_m]m x [resolution_m]m on the ground.
    """
    
    def __init__(self, gdf: gpd.GeoDataFrame, resolution_m):
        """
        Initialize the rasterizer.
        
        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame containing OSM building polygons (already filtered)
        resolution : float
            Cell size in meters
        """
        self.gdf = gdf.copy()
        self.resolution_m = resolution_m
        self.matrix = None
        
        # Check if CRS is geographic (uses degrees as units)
        # If so, we MUST reproject to a projected CRS that uses meters
        # Otherwise 1 cell = 1 degree, which makes no physical sense
        if self.gdf.crs is None:
            raise ValueError("GeoDataFrame has no CRS defined. Please set a CRS.")
        
        if self.gdf.crs.is_geographic:
            print(f"Detected geographic CRS ({self.gdf.crs.name}). Reprojecting to {LocalCRS.FRANCE_LAMBERT93.crs.name}...")
            self.gdf = self.gdf.to_crs(LocalCRS.FRANCE_LAMBERT93.crs)
        else:
            print(f"Using existing projected CRS: {self.gdf.crs.name}")
    
    def _prepare_raster_params(self):
        """
        Calculate all parameters needed for rasterization.
        
        Returns
        -------
        tuple
            (out_shape, transform) for rasterio.rasterize()
        """
        # Get the total bounding box of all buildings
        bounds = self.gdf.total_bounds  # (minx, miny, maxx, maxy)
        
        # Calculate grid dimensions in pixels
        width = int(np.ceil((bounds[2] - bounds[0]) / self.resolution_m))
        height = int(np.ceil((bounds[3] - bounds[1]) / self.resolution_m))
        
        # Create affine transform: maps pixel coordinates to real-world coordinates
        # from_origin(west, north, pixel_width, pixel_height)
        # origin is at top-left corner of the grid
        transform = from_origin(
            bounds[0],       # west edge (minimum x)
            bounds[3],       # north edge (maximum y)
            self.resolution_m, # cell width in meters
            self.resolution_m  # cell height in meters
        )
        
        out_shape = (height, width)
        return out_shape, transform, bounds
    
    def rasterize(self):
        """
        Create a binary 2D matrix: 1 for building, 0 for empty.
        
        Returns
        -------
        numpy.ndarray
            2D matrix of shape (height, width) with binary values (0 or 1)
        """
        # Prepare rasterization parameters
        out_shape, transform, _ = self._prepare_raster_params()
        
        # Create (geometry, value) pairs: each building footprint gets value 1
        shapes = [(geom, 1) for geom in self.gdf.geometry]
        
        # Burn vector shapes into the raster grid
        # - shapes: list of (geometry, burn_value) tuples
        # - out_shape: (height, width) of the output array
        # - transform: affine mapping from pixel coords to world coords
        # - fill: background value (0 = no building)
        # - all_touched: if False, only pixels whose CENTER is inside the polygon are filled
        self.matrix:np.ndarray = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False
        )
        
        print(f"Rasterization complete. Matrix shape: {self.matrix.shape}")
        print(f"Coverage: {np.sum(self.matrix > 0)} cells with buildings "
              f"({np.sum(self.matrix > 0) / self.matrix.size * 100:.2f}%)")
        
        return self.matrix