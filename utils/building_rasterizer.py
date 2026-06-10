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
    
    def __init__(self, gdf: gpd.GeoDataFrame, resolution_m: float) -> None:
        """
        Initialize the rasterizer.
        
        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame containing OSM building polygons (already filtered)
        resolution_m : float
            Cell size in meters
        """
        self.gdf = gdf.copy()
        self.resolution_m = resolution_m
        self.matrix = None
        
        # Check if CRS is geographic (uses degrees as units)
        if self.gdf.crs is None:
            raise ValueError("GeoDataFrame has no CRS defined. Please set a CRS.")
        
        if self.gdf.crs.is_geographic:
            self.gdf = self.gdf.to_crs(LocalCRS.FRANCE_LAMBERT93.crs)
    
    def rasterize(self, bounds: tuple[int, int, int, int]):
        """
        Create a binary 2D matrix: 1 for building, 0 for empty.
        
        Parameters
        ----------
        bounds : tuple or None
            (x_min, y_min, x_max, y_max) in the projected CRS (meters).
            If None, use the bounding box of all buildings.
        
        Returns
        -------
        numpy.ndarray
            2D matrix of shape (height, width) with binary values (0 or 1)
        """
        out_shape, transform, used_bounds = self._prepare_raster_params(bounds=bounds)
        
        # Create (geometry, value) pairs: each building footprint gets value 1
        shapes = [(geom, 1) for geom in self.gdf.geometry]
        
        # Burn vector shapes into the raster grid
        self.matrix = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False
        )
        
        # Store bounds for later overlap removal
        self.raster_bounds = used_bounds
        
        return self.matrix
    
    def remove_overlap(self, overlap_m: int):
        """
        Remove overlap from the rasterized matrix.
        Crops the matrix from (block_size + 2*overlap) to block_size.
        
        Parameters
        ----------
        overlap_m : int
            Overlap in meters that was added to each side.
            Matrix will be cropped by overlap_m/resolution_m pixels on each side.
        
        Returns
        -------
        numpy.ndarray
            Cropped matrix without overlap area
        """
        if self.matrix is None:
            raise ValueError("No matrix to crop. Call rasterize() first.")
        
        # Calculate how many pixels to remove from each side
        pixels_to_crop = int(overlap_m / self.resolution_m)
        
        # Validate that we have enough pixels to crop
        if pixels_to_crop * 2 >= self.matrix.shape[0]:
            raise ValueError(
                f"Overlap too large for matrix height. "
                f"Matrix height: {self.matrix.shape[0]}, "
                f"pixels to crop (each side): {pixels_to_crop}"
            )
        if pixels_to_crop * 2 >= self.matrix.shape[1]:
            raise ValueError(
                f"Overlap too large for matrix width. "
                f"Matrix width: {self.matrix.shape[1]}, "
                f"pixels to crop (each side): {pixels_to_crop}"
            )
        
        # Crop matrix: remove overlap pixels from all four sides
        # Note: y-axis is inverted in raster coordinates
        # top-left is (0,0), so we crop from top, bottom, left, right
        self.matrix = self.matrix[
            pixels_to_crop:-pixels_to_crop,    # rows: top and bottom
            pixels_to_crop:-pixels_to_crop     # cols: left and right
        ]
        
        return self.matrix
    
    def _prepare_raster_params(self, bounds: tuple[int, int, int, int]):
        """
        Calculate all parameters needed for rasterization.
        
        Parameters
        ----------
        bounds : tuple or None
            (x_min, y_min, x_max, y_max) in the projected CRS (meters).
            If None, use the bounding box of all buildings.
        
        Returns
        -------
        tuple
            (out_shape, transform, bounds)
        """
        x_min, y_min, x_max, y_max = bounds
        
        # Calculate grid dimensions in pixels
        width = int(np.ceil((x_max - x_min) / self.resolution_m))
        height = int(np.ceil((y_max - y_min) / self.resolution_m))
        
        # Create affine transform: origin at top-left corner
        transform = from_origin(
            x_min,              # west edge
            y_max,              # north edge (note: y_max, not y_min)
            self.resolution_m,  # cell width in meters
            self.resolution_m   # cell height in meters
        )
        
        out_shape = (height, width)
        return out_shape, transform, (x_min, y_min, x_max, y_max)