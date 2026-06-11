import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from typing import Dict
from config import LocalCRS
from utils.map_splitter import BlockMeta

class BuildingRasterizer:
    """
    Convert OSM building GeoDataFrame into a binary 2D raster matrix.
    Each cell represents [resolution_m]m x [resolution_m]m on the ground.
    """
    
    def __init__(
            self, 
            gdf: gpd.GeoDataFrame, 
            block_meta:BlockMeta, 
            resolution_m: float) -> None:
        """
        Initialize the rasterizer.
        
        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame containing OSM building polygons (already filtered)
        block_meta : BlockMeta
        resolution_m : float
            Cell size in meters
        """
        self.gdf = gdf.copy()
        self.resolution_m = resolution_m
        self.block_meta = block_meta
        self._setup_matrix_bounds()

        # Matrix storage - using dict for future extension
        # Key: property name (e.g., 'presence', 'height')
        # Value: 2D numpy array
        self.matrixs: Dict[str, np.ndarray] = {}
        
        # Check if CRS is geographic (uses degrees as units)
        if self.gdf.crs is None:
            raise ValueError("GeoDataFrame has no CRS defined. Please set a CRS.")
        
        if self.gdf.crs.is_geographic:
            self.gdf = self.gdf.to_crs(LocalCRS.FRANCE_LAMBERT93.crs)
    
    def rasterize_with_presence(self)->np.ndarray:
        """
        Create a binary 2D matrix: 1 for building, 0 for empty.
        
        Returns
        -------
        numpy.ndarray
            2D matrix of shape (height, width) with binary values (0 or 1)
        """

        # Create (geometry, value) pairs: each building footprint gets value 1
        shapes = [(geom, 1) for geom in self.gdf.geometry]

        transform = from_origin(
            self.x_min,              # west edge
            self.y_max,              # north edge (note: y_max, not y_min)
            self.resolution_m,  # cell width in meters
            self.resolution_m   # cell height in meters
        )
        
        # Burn vector shapes into the raster grid
        matrix: np.ndarray = rasterize(
            shapes=shapes,
            out_shape=(self.n_rows, self.n_cols),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False
        )

        # Remove overlop
        matrix = matrix[self.crop_slice]

        self.matrixs['presence'] = matrix

        return matrix
    
    def _setup_matrix_bounds(self):
        """
        Calculate matrix boundaries from BlockMeta in projected coordinates.
        BlockMeta already has x_start, x_end, y_start, y_end in meters.
        """
        # Use BlockMeta's meter coordinates directly
        self.x_min = self.block_meta.x_start
        self.x_max = self.block_meta.x_end
        self.y_min = self.block_meta.y_start
        self.y_max = self.block_meta.y_end

        self.overlap_m =self.block_meta.overlap_m
        
        # Calculate matrix dimensions
        total_width = self.x_max - self.x_min
        total_height = self.y_max - self.y_min
        
        self.n_cols = int(np.ceil(total_width / self.resolution_m))
        self.n_rows = int(np.ceil(total_height / self.resolution_m))
        
        # Pre-calculate crop indices for overlap removal
        if self.overlap_m > 0:
            cells_to_crop = int(np.ceil(self.overlap_m / self.resolution_m))
            # Check if cropping is possible
            if cells_to_crop == 0:
                self.crop_slice = np.s_[:self.n_rows, :self.n_cols]
            elif cells_to_crop * 2 < self.n_rows and cells_to_crop * 2 < self.n_cols:
                self.crop_slice = np.s_[cells_to_crop:-cells_to_crop, cells_to_crop:-cells_to_crop]
            else:
                # Overlap is too large relative to matrix size
                self.crop_slice = np.s_[:self.n_rows, :self.n_cols]
        else:
            self.crop_slice = np.s_[:self.n_rows, :self.n_cols]

    def get_matrix(self, matrix_type: str = 'presence') -> np.ndarray:
        """
        Get a specific matrix by type.
        
        Parameters
        ----------
        matrix_type: str 
            'presence', ['count', or 'height' for future]
            
        Returns
        -------
        ndarray
            Requested matrix array
        """
        if matrix_type not in self.matrixs:
            available = list(self.matrixs.keys())
            raise ValueError(f"matrix type '{matrix_type}' not created. Available: {available}")
        
        matrix = self.matrixs[matrix_type]
        
        return matrix