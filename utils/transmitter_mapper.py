import pandas as pd
import numpy as np
from pyproj import Transformer
from typing import Tuple, Dict
from config import LocalCRS
from utils.map_splitter import BlockMeta


class TransmitterMapper:
    """Map transmitter locations to a matrix using BlockMeta bounds.
    
    Notice:
    ---
    Geographic coordinates use LocalCRS.OSM_STORAGE.
    Planar coordinates use LocalCRS.FRANCE_LAMBERT93.
    """
    
    def __init__(
        self, 
        block_meta: BlockMeta,  # BlockMeta from TileSplitter
        resolution_m: float
    ):
        """
        Initialize mapper with block metadata.
        
        Parameters
        ----------
        block_meta : BlockMeta
        resolution_m : float
            Cell size in meters
        """
        self.block_meta = block_meta
        self.resolution_m = resolution_m
        
        # Coordinate transformers
        self.to_meters = Transformer.from_crs(LocalCRS.OSM_STORAGE.crs, LocalCRS.FRANCE_LAMBERT93.crs, always_xy=True)
        self.to_latlon = Transformer.from_crs(LocalCRS.FRANCE_LAMBERT93.crs, LocalCRS.OSM_STORAGE.crs, always_xy=True)
        
        # Matrix storage - using dict for future extension
        # Key: property name (e.g., 'presence', 'count', 'height')
        # Value: 2D numpy array
        self.matrixs: Dict[str, np.ndarray] = {}
        
        # Calculate matrix boundaries from BlockMeta
        self._setup_matrix_bounds()
    
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
    
    def filter_transmitters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter transmitters within BlockMeta's lat/lon bounds.
        
        Parameters
        ----------
        df : Dataframe 
            Contain transmitters with 'Latitude' and 'Longitude' columns
            
        Returns
        -------
        DataFrame
            Filtered dataframe
        """
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            raise ValueError("Dataframe must contain 'Latitude' and 'Longitude' columns")
        
        # Filter using BlockMeta's lat/lon bounds
        mask = (
            (df['Latitude'] >= self.block_meta.lat_min) & 
            (df['Latitude'] <= self.block_meta.lat_max) &
            (df['Longitude'] >= self.block_meta.lon_min) & 
            (df['Longitude'] <= self.block_meta.lon_max)
        )
        
        self.filtered_data = df[mask].copy()
        return self.filtered_data
    
    def _project_transmitters(self) -> np.ndarray:
        """
        Convert transmitter lat/lon to projected coordinates.
        
        Returns
        -------
        ndarray
            Nx2 array of (x, y) coordinates in meters
        """
        if not hasattr(self, 'filtered_data'):
            raise ValueError("No filtered data. Run filter_transmitters() first.")
        
        coords = []
        for _, row in self.filtered_data.iterrows():
            x, y = self.to_meters.transform(row['Longitude'], row['Latitude'])
            coords.append([x, y])
        
        return np.array(coords)
    
    def _coordinates_to_matrix_indices(self, x_coords: np.ndarray, y_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert meter coordinates to matrix indices.
        
        Parameters
        ----------
        x_coords : ndarray
            Array of x coordinates in meters
        y_coords : ndarray
            Array of y coordinates in meters
            
        Returns
        -------
        tuple(ndarray, ndarray)
            Tuple of (row_indices, col_indices)
        """
        col_indices = ((x_coords - self.x_min) / self.resolution_m).astype(int)
        row_indices = ((y_coords - self.y_min) / self.resolution_m).astype(int)
        return row_indices, col_indices
    
    def _create_matrix(self, dtype=np.uint8) -> np.ndarray:
        """
        Create an empty matrix with proper dimensions.
        
        Parameters
        ----------
        dtype : type
            Data type for matrix values
            
        Returns
        -------
        ndarray
            Zero-filled matrix array
        """
        return np.zeros((self.n_rows, self.n_cols), dtype=dtype)
    
    def create_presence_matrix(self) -> np.ndarray:
        """
        Create binary presence matrix (1 where transmitter exists, 0 otherwise).
        
        Returns
        -------
        ndarray
            2D numpy array indicating transmitter presence
        """
        if not hasattr(self, 'filtered_data'):
            raise ValueError("No filtered data. Run filter_transmitters() first.")
        
        # Project transmitters to meter coordinates
        coords = self._project_transmitters()
        if len(coords) == 0:
            self.matrixs['presence'] = self._create_matrix()
            return self.matrixs['presence']
        
        # Convert to matrix indices
        row_indices, col_indices = self._coordinates_to_matrix_indices(coords[:, 0], coords[:, 1])
        
        # Create presence matrix
        matrix = self._create_matrix()
        
        # Filter valid indices
        valid_mask = (0 <= row_indices) & (row_indices < self.n_rows) & \
                     (0 <= col_indices) & (col_indices < self.n_cols)
        
        row_indices = row_indices[valid_mask]
        col_indices = col_indices[valid_mask]
        
        # Set presence to 1
        matrix[row_indices, col_indices] = 1

        # Remove overlop
        matrix = matrix[self.crop_slice]
        
        self.matrixs['presence'] = matrix

        return matrix
    
    def get_matrix(self, matrix_type: str = 'presence') -> np.ndarray:
        """
        Get a specific matrix by type.
        
        Args:
            matrix_type: 'presence', ['count', or 'height' for future]
            remove_overlap: If True, return matrix without overlap
            
        Returns:
            Requested matrix array
        """
        if matrix_type not in self.matrixs:
            available = list(self.matrixs.keys())
            raise ValueError(f"matrix type '{matrix_type}' not created. Available: {available}")
        
        matrix = self.matrixs[matrix_type]
        
        return matrix