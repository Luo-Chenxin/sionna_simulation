import numpy as np
from pyproj import Transformer
from config import LocalCRS

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
        Initialize the splitter.
        """
        self.block_size_m = block_size_m
        self.overlap_m = overlap_m
        
        self.to_utm = Transformer.from_crs(LocalCRS.OSM_STORAGE.crs, LocalCRS.FRANCE_LAMBERT93.crs, always_xy=True)
        self.to_latlon = Transformer.from_crs(LocalCRS.FRANCE_LAMBERT93.crs, LocalCRS.OSM_STORAGE.crs, always_xy=True)
        
        # Change map corners from Lat/Lon to UTM meters.
        x_left_bottom, y_left_bottom = self.to_utm.transform(lon_min, lat_min)
        x_right_top, y_right_top = self.to_utm.transform(lon_max, lat_max)
        
        # Get min and max values for X and Y.
        self.x_min = min(x_left_bottom, x_right_top)
        self.x_max = max(x_left_bottom, x_right_top)
        self.y_min = min(y_left_bottom, y_right_top)
        self.y_max = max(y_left_bottom, y_right_top)
        
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

    def get_block_latlon_bounds(self, row, col, with_overlap=True):
        """
        Get Lat/Lon bounds for one block.
        row: index from 0 to rows-1
        col: index from 0 to cols-1
        with_overlap: True for Sionna simulation, False for final data crop.
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError("Index out of range!")
            
        # Calculate block corners in meters (no overlap).
        x_start = self.x_min + col * self.block_size_m
        x_end = x_start + self.block_size_m
        y_start = self.y_min + row * self.block_size_m
        y_end = y_start + self.block_size_m
        
        # Add overlap buffer if needed.
        if with_overlap:
            x_start -= self.overlap_m
            x_end += self.overlap_m
            y_start -= self.overlap_m
            y_end += self.overlap_m
            
        # Change meters back to Lat/Lon.
        lon_min_block, lat_min_block = self.to_latlon.transform(x_start, y_start)
        lon_max_block, lat_max_block = self.to_latlon.transform(x_end, y_end)
        
        return {
            "lat_min": min(lat_min_block, lat_max_block),
            "lat_max": max(lat_min_block, lat_max_block),
            "lon_min": min(lon_min_block, lon_max_block),
            "lon_max": max(lon_min_block, lon_max_block)
        }

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

# --- Example Usage ---
if __name__ == "__main__":
    # Test values for Paris
    LAT_MAX, LAT_MIN = 48.9059, 48.8138
    LON_MIN, LON_MAX = 2.2429, 2.4574

    splitter = TileSplitter(
        lat_min=LAT_MIN, lat_max=LAT_MAX, 
        lon_min=LON_MIN, lon_max=LON_MAX, 
        block_size_m=2000, overlap_m=150
    )

    # Get bounds for block (0,0) with overlap
    osm_bounds = splitter.get_block_latlon_bounds(row=0, col=0, with_overlap=True)
    print("\nBounds with overlap for OSM download:")
    print(osm_bounds)