import numpy as np
from pyproj import Transformer
from config import LocalCRS
from dataclasses import dataclass

@dataclass
class BlockMeta:
    row: int
    col: int
    name: str
    block_size_m: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    # Latitude and longitude with overlap
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    # Latitude and longitude coordinates without overlap (core area)
    lat_min_core: float
    lat_max_core: float
    lon_min_core: float
    lon_max_core: float
    overlap_m: int

class TileSplitter:
    def __init__(
            self, 
            lat_min: float, 
            lat_max: float, 
            lon_min: float, 
            lon_max: float, 
            block_size_m: int, 
            overlap_m: int):
        """
        Initialize the splitter with pure meter-level grid alignment.
        """
        self.block_size_m = block_size_m
        self.overlap_m = overlap_m
        
        self.to_utm = Transformer.from_crs(LocalCRS.OSM_STORAGE.crs, LocalCRS.FRANCE_LAMBERT93.crs, always_xy=True)
        self.to_latlon = Transformer.from_crs(LocalCRS.FRANCE_LAMBERT93.crs, LocalCRS.OSM_STORAGE.crs, always_xy=True)
        
        # Change map corners from Lat/Lon to UTM meters.
        x_left_bottom, y_left_bottom = self.to_utm.transform(lon_min, lat_min)
        x_right_top, y_right_top = self.to_utm.transform(lon_max, lat_max)
        
        # Get min and max values for X and Y
        # Force boundaries to be absolute integers to avoid float bugs
        self.x_min = int(np.floor(min(x_left_bottom, x_right_top)))
        self.x_max = int(np.ceil(max(x_left_bottom, x_right_top)))
        self.y_min = int(np.floor(min(y_left_bottom, y_right_top)))
        self.y_max = int(np.ceil(max(y_left_bottom, y_right_top)))
        
        # Calculate total width and height in meters.
        total_width_m = self.x_max - self.x_min
        total_height_m = self.y_max - self.y_min
        
        # Calculate how many rows and columns we need.
        # Use ceil to cover the whole area.
        self.cols = int(np.ceil(total_width_m / self.block_size_m))
        self.rows = int(np.ceil(total_height_m / self.block_size_m))
        
        print(f"--- Grid Split Done ---")
        print(f"Total Width: {total_width_m:.2f}m, Total Height: {total_height_m:.2f}m")
        print(f"Grid Size: {self.rows} rows x {self.cols} cols. Total blocks: {self.rows * self.cols}")

    def get_block_latlon_bounds(self, row, col) -> BlockMeta:
        """
        Get Lat/Lon bounds for one block.
        row: index from 0 to rows-1
        col: index from 0 to cols-1
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError("Index out of range!")
            
        # Calculate core block corners in meters (no overlap)
        x_start_core = self.x_min + col * self.block_size_m
        x_end_core = x_start_core + self.block_size_m
        y_start_core = self.y_min + row * self.block_size_m
        y_end_core = y_start_core + self.block_size_m
        
        # Add overlap buffer for extended block
        x_start_ext = x_start_core - self.overlap_m
        x_end_ext = x_end_core + self.overlap_m
        y_start_ext = y_start_core - self.overlap_m
        y_end_ext = y_end_core + self.overlap_m
            
        # Transform core meters to Lat/Lon (without overlap)
        lon_core_1, lat_core_1 = self.to_latlon.transform(x_start_core, y_start_core)
        lon_core_2, lat_core_2 = self.to_latlon.transform(x_end_core, y_end_core)
        
        # Transform extended meters to Lat/Lon (with overlap)
        lon_ext_1, lat_ext_1 = self.to_latlon.transform(x_start_ext, y_start_ext)
        lon_ext_2, lat_ext_2 = self.to_latlon.transform(x_end_ext, y_end_ext)
            
        return BlockMeta(
            row=row,
            col=col,
            name=f"block_{row}_{col}",
            block_size_m=self.block_size_m,
            x_start=x_start_ext, x_end=x_end_ext,
            y_start=y_start_ext, y_end=y_end_ext,
            # Latitude and longitude boundaries with overlap
            lat_min=min(lat_ext_1, lat_ext_2),
            lat_max=max(lat_ext_1, lat_ext_2),
            lon_min=min(lon_ext_1, lon_ext_2),
            lon_max=max(lon_ext_1, lon_ext_2),
            # Latitude and longitude boundaries without overlap (core area)
            lat_min_core=min(lat_core_1, lat_core_2),
            lat_max_core=max(lat_core_1, lat_core_2),
            lon_min_core=min(lon_core_1, lon_core_2),
            lon_max_core=max(lon_core_1, lon_core_2),
            overlap_m=self.overlap_m
        )

    def get_all_blocks(self):
        """
        Get a list of all block indices and names.
        """
        blocks_list = []
        for r in range(self.rows):
            for c in range(self.cols):
                blocks_list.append({
                    "row": r,
                    "col": c,
                    "name": f"block_{r}_{c}"
                })
        return blocks_list