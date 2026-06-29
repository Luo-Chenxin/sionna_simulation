#!/usr/bin/env python3
"""
Paris Radio Map Generation Pipeline

This script processes the Paris area radio map generation. It can run:
- Step 1: Region Grid Splitting
- Step 2: Per-Block PLY & XML Generation
- Step 3: Building Raster Generation  
- Step 4: Transmitter Raster Generation
- Step 5: Transmitter Relocation (to nearest building)
- Step 6: Per-Block Radio Map Generation
- Step 7: Merge Block HDF5 Files

Usage:
    python paris_simulation.py [--skip-merge] [--start-step STEP] [--end-step STEP] [--step STEP]

Options:
    --skip-merge       Skip the merge step (only process individual blocks)
    --start-step N     Start from step N (1-7)
    --end-step N       End at step N (1-7)
    --step N           Run only step N (equivalent to --start-step N --end-step N)
"""

import argparse
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import h5py
import sionna.rt as rt

# Import utility modules
from utils.geo_coords import SceneCoordinateConverter
from utils.generate_radiomap import RadioMapGenerator
from utils.h5_manager import H5Manager
from utils.osm_fetcher import OSMFetcher, OSM_TAGS
from utils.building_rasterizer import BuildingRasterizer
from utils.transmitter_mapper import TransmitterMapper
from utils.map_splitter import TileSplitter
from utils.osm_to_ply import OSMToPLY, generate_flat_terrain_ply
from utils.generate_xml import SionnaXMLGenerator
from utils.preprocess_raw_data import process_csv_with_buffer
from config import LocalCRS

# =============================================================================
# Configuration Constants
# =============================================================================

# Directories
TRANSMITTER_DIRECTORY = Path('data/transmitters')
XML_DIR = Path('data/xml')
DATASET_DIR = Path('data/dataset')
MERGED_DIR = Path('data/merged_dataset')
MERGED_FILENAME = MERGED_DIR / 'block.h5'

# Simulation parameters
TERRAIN_HEIGHT = 0.0
TERRAIN_RESOLUTION = 10.0
RM_RESOLUTION_M = 1.0
FREQUENCY = 2.6e9  # 2.6 GHz
TRANSMITTER_PATH = Path('data/transmitters/2600_mhz.csv')

# Block splitting parameters
BLOCK_SIZE_M = 256
OVERLAP_M = 0
STRIDE_M = 128

# Transmitter relocation parameter
USE_GRID_MODE = True  # True: use corrected grid from HDF5, False: use original CSV
TX_SEARCH_RANGE = 20  # ±20 pixels (±20m) search range for relocation

# Geographic boundaries (calculated from transmitter locations + 500m extension)
LAT_MAX, LAT_MIN, LON_MIN, LON_MAX = process_csv_with_buffer(
    tx_csv_path=TRANSMITTER_PATH, buffer_meters=OVERLAP_M
)

# Calculate the center origin point of the scene
LAT_ORIGIN = (LAT_MAX + LAT_MIN) / 2
LON_ORIGIN = (LON_MIN + LON_MAX) / 2

# Merge parameters
MERGED_BLOCK_SIZE_M = 256

# Layer definitions for Step 2
LAYERS = [
    {
        "tag_name": "buildings",
        "ply_filename": "buildings.ply",
        "default_height": 3,
        "handle_missing_height": "use_default",
    },
]

FULL_BOX = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)

# Global OSMFetcher instance — created once and reused across steps
# Caching is automatic: the fetcher saves chunks to data/osm_cache/
# and reuses them on subsequent runs.
_fetcher = None


def get_fetcher():
    """
    Get or create the shared OSMFetcher instance.
    
    Uses lazy initialization to avoid downloading data until needed.
    The fetcher caches data automatically in data/osm_cache/chunks/.
    """
    global _fetcher
    if _fetcher is None:
        print(f"\nInitializing OSM Fetcher for area: {FULL_BOX}")
        # cache_filepath is set to None — large area will use automatic
        # chunk caching in data/osm_cache/chunks/
        _fetcher = OSMFetcher(
            bbox=FULL_BOX,
            tags=OSM_TAGS['buildings'],
            cache_filepath=None,  # Auto-managed chunk cache
        )
        print(f"Fetcher ready. Cache directory: data/osm_cache/")
    return _fetcher


def setup_antenna_array():
    """Setup the antenna array for ray-tracing"""
    return rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )


# =============================================================================
# Transmitter Relocator Class
# =============================================================================

class TransmitterRelocator:
    """
    Relocates transmitters to the nearest unoccupied building
    within a square search range.

    For each transmitter not already on a building, the class searches
    for the nearest unoccupied building within ±d pixels (clipped to
    image boundaries). If multiple buildings are at the same distance,
    the one with the smallest x (row), then smallest y (column) is chosen.

    Buildings already hosting a transmitter are marked as occupied
    and cannot be selected by other transmitters.

    Parameters
    ----------
    search_range : int
        Half-size of the square search window. A transmitter at (i, j)
        will search within rows [i-d, i+d] and cols [j-d, j+d].
    """

    def __init__(self, search_range: int):
        if search_range < 0:
            raise ValueError("search_range must be non-negative")
        self.search_range = search_range

    def process_file(self, file_path: Union[str, Path]) -> None:
        """
        Process a single HDF5 file in-place.

        Reads 'transmitters' and 'buildings' datasets, relocates
        transmitters according to the rules, and writes the updated
        'transmitters' dataset back. All other datasets are left untouched.

        Parameters
        ----------
        file_path : str or Path
            Path to the HDF5 file to process.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with h5py.File(file_path, "r+") as f:
            if "transmitters" not in f or "buildings" not in f:
                raise KeyError(
                    f"File {file_path} must contain 'transmitters' and 'buildings' datasets"
                )

            transmitters = f["transmitters"][:]  # shape (M, M), uint8
            buildings = f["buildings"][:]        # shape (M, M), uint8

            # Perform relocation
            new_transmitters = self._relocate(transmitters, buildings)

            # Write back to file
            f["transmitters"][:] = new_transmitters

    def _relocate(
        self,
        transmitters: np.ndarray,
        buildings: np.ndarray
    ) -> np.ndarray:
        """
        Core relocation logic for a single map.

        Parameters
        ----------
        transmitters : np.ndarray
            2D array of shape (M, M), dtype uint8.
            1 indicates a transmitter, 0 indicates none.
        buildings : np.ndarray
            2D array of shape (M, M), dtype uint8.
            1 indicates a building, 0 indicates none.

        Returns
        -------
        np.ndarray
            Updated transmitters array after relocation.
        """
        M = transmitters.shape[0]
        d = self.search_range

        # Copy transmitters to modify
        result = transmitters.copy()

        # Track which buildings are already occupied by a transmitter
        # Initially: buildings that already have a transmitter on them
        occupied = (buildings == 1) & (transmitters == 1)

        # Get coordinates of all transmitter positions (row, col order)
        tx_rows, tx_cols = np.where(transmitters == 1)

        for tx_r, tx_c in zip(tx_rows, tx_cols):
            # If transmitter is already on a building, it stays
            # (occupied was already marked during initialization)
            if buildings[tx_r, tx_c] == 1:
                continue

            # Search for the nearest unoccupied building within range
            best_coords = self._find_nearest_building(
                tx_r, tx_c, buildings, occupied, M, d
            )

            if best_coords is not None:
                best_r, best_c = best_coords
                # Move transmitter: clear old position, set new position
                result[tx_r, tx_c] = 0
                result[best_r, best_c] = 1
                # Mark the new building as occupied
                occupied[best_r, best_c] = True
            # else: no building found within range, transmitter stays in place

        return result

    def _find_nearest_building(
        self,
        tx_r: int,
        tx_c: int,
        buildings: np.ndarray,
        occupied: np.ndarray,
        M: int,
        d: int
    ) -> tuple[int, int] | None:
        """
        Find the nearest unoccupied building within ±d of (tx_r, tx_c).

        Tie-breaking: smallest row (x), then smallest column (y).

        Parameters
        ----------
        tx_r, tx_c : int
            Transmitter coordinates (row, column).
        buildings : np.ndarray
            Building map.
        occupied : np.ndarray
            Mask of buildings already occupied by a transmitter.
        M : int
            Size of the square map (M x M).
        d : int
            Search range half-width.

        Returns
        -------
        (int, int) or None
            Coordinates of the nearest building, or None if not found.
        """
        # Compute search boundaries, clipped to valid range
        r_min = max(0, tx_r - d)
        r_max = min(M - 1, tx_r + d)
        c_min = max(0, tx_c - d)
        c_max = min(M - 1, tx_c + d)

        best_dist_sq = float("inf")
        best_r = None
        best_c = None

        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                # Skip if no building here or building is already occupied
                if buildings[r, c] == 0 or occupied[r, c]:
                    continue

                # Compute squared Euclidean distance
                dr = r - tx_r
                dc = c - tx_c
                dist_sq = dr * dr + dc * dc

                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_r = r
                    best_c = c
                elif dist_sq == best_dist_sq:
                    # Tie-breaking: smaller row first, then smaller column
                    if r < best_r or (r == best_r and c < best_c):
                        best_r = r
                        best_c = c

        if best_r is not None:
            return best_r, best_c
        return None


# =============================================================================
# Pipeline Steps
# =============================================================================

def step1_split_region(splitter):
    """
    Step 1: Region Grid Splitting
    
    Split the large Paris area into smaller blocks with overlap.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Region Grid Splitting")
    print("=" * 60)
    
    all_blocks = splitter.get_all_blocks(tx_csv_path=TRANSMITTER_PATH)
    
    print(f"Geographic boundaries:")
    print(f"  Latitude:  {LAT_MIN:.6f} to {LAT_MAX:.6f}")
    print(f"  Longitude: {LON_MIN:.6f} to {LON_MAX:.6f}")
    print(f"Block size: {BLOCK_SIZE_M} m")
    print(f"Overlap:    {OVERLAP_M} m")
    print(f"\nTotal blocks: {len(all_blocks)}")
    
    print("\nStep 1 complete.")
    return all_blocks


def step2_generate_ply_and_xml(splitter, all_blocks):
    """
    Step 2: Per-Block PLY & XML Generation
    
    For each block, fetch OSM data, generate PLY meshes, and create XML scene file.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Per-Block PLY & XML Generation")
    print("=" * 60)

    # Get shared fetcher (only downloads once, cached on disk)
    full_fetcher = get_fetcher()
    print(f"Using cached OSM data from data/osm_cache/")
    
    for block_info in all_blocks:
        row = block_info["row"]
        col = block_info["col"]
        block_name = block_info["name"]

        block_dir = XML_DIR / block_name
        if block_dir.exists():
            print(f"Skipping {block_name} (directory already exists)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {block_name} (row={row}, col={col})")
        print(f"{'='*60}")
        
        # Get block metadata (with overlap for consistent coverage)
        meta = splitter.get_block_latlon_bounds(row, col)
        
        # Create output directories
        mesh_dir = block_dir / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate PLY for each layer
        for layer in LAYERS:
            tag_name = layer["tag_name"]
            ply_filename = layer["ply_filename"]
            ply_path = mesh_dir / ply_filename
            
            # Filter the pre-fetched data for this specific layer
            # The fetcher lazily loads only relevant chunks for this sub_bbox
            gdf = full_fetcher.get_filtered_features(
                OSM_TAGS[tag_name],
                sub_bbox=(meta.lon_min, meta.lat_min, meta.lon_max, meta.lat_max),
            )
            
            if gdf.empty:
                print(f"  [{tag_name}] No features found — writing empty PLY")
            
            # Convert OSM geometries to 3D PLY mesh
            converter = OSMToPLY(
                gdf=gdf,
                ply_path=ply_path,
                default_height=layer["default_height"],
                block_meta=meta,
                handle_missing_height=layer["handle_missing_height"],
            )
            converter.save_to_ply()
            
            print(f"  [{tag_name}] Saved to {ply_path}")
        
        # Generate terrain PLY
        terrain_path = mesh_dir / "terrain.ply"
        generate_flat_terrain_ply(
            output_path=terrain_path,
            x_min=meta.x_start,
            x_max=meta.x_end,
            y_min=meta.y_start,
            y_max=meta.y_end,
            resolution=TERRAIN_RESOLUTION,
            height=TERRAIN_HEIGHT
        )
        print(f"  [terrain] Saved to {terrain_path}")
        
        # Generate Sionna XML scene file
        xml_path = block_dir / f"{block_name}.xml"
        xml_generator = SionnaXMLGenerator(mesh_dir=mesh_dir, output_path=xml_path)
        xml_generator.generate(validate_meshes=False)
        print(f"  [xml] Saved to {xml_path}")
        
        print(f"Finished {block_name}")
    
    print(f"\nStep 2 complete. Processed {len(all_blocks)} blocks.")
    print(f"Output directory: {XML_DIR.resolve()}")


def step3_generate_building_rasters(splitter, all_blocks):
    """
    Step 3: Building Raster Generation
    
    For each block, rasterize building footprints into a binary presence matrix.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Building Raster Generation")
    print("=" * 60)
    
    # Reuse shared fetcher (already cached on disk from Step 2)
    full_fetcher = get_fetcher()
    print(f"Using cached OSM data from data/osm_cache/")
    
    for block_info in all_blocks:
        row = block_info["row"]
        col = block_info["col"]
        block_name = block_info["name"]
        
        print(f"\n{'='*60}")
        print(f"Processing {block_name} (row={row}, col={col})")
        print(f"{'='*60}")
        
        # Get block metadata
        meta = splitter.get_block_latlon_bounds(row, col)
        
        # Fetch OSM building footprints using shared fetcher
        # Lazy loading: only chunks that intersect this sub_bbox are loaded
        buildings_gdf = full_fetcher.get_filtered_features(
            OSM_TAGS["buildings"],
            sub_bbox=(meta.lon_min, meta.lat_min, meta.lon_max, meta.lat_max),
        )
        
        if buildings_gdf.empty:
            print(f"  [buildings] No building footprints found in {block_name}")
            building_map = np.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=np.uint8)
        else:
            # Rasterize building footprints
            rasterizer = BuildingRasterizer(
                gdf=buildings_gdf,
                block_meta=meta,
                resolution_m=RM_RESOLUTION_M,
            )
            building_map = rasterizer.rasterize_with_presence()
        
        # Write to HDF5
        h5_path = DATASET_DIR / f"{block_name}.h5"

        if not h5_path.exists():
            H5Manager.init_block_file(h5_path, meta, RM_RESOLUTION_M)
        
        H5Manager.write_dataset(
            h5_path,
            H5Manager.DATASET_BUILDINGS,
            building_map,
            dtype='uint8',
        )
        print(f"  [h5] Written buildings to {h5_path}")
    
    print(f"\nStep 3 complete. Processed {len(all_blocks)} blocks.")


def step4_generate_transmitter_rasters(splitter, all_blocks):
    """
    Step 4: Transmitter Raster Generation
    
    For each block, rasterize transmitter locations into a binary presence matrix.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Transmitter Raster Generation")
    print("=" * 60)
    
    # Load all transmitters
    if not TRANSMITTER_PATH.exists():
        print(f"Error: Transmitter file not found: {TRANSMITTER_PATH}")
        return
    
    df_tx_all = pd.read_csv(TRANSMITTER_PATH)
    
    for block_info in all_blocks:
        row = block_info["row"]
        col = block_info["col"]
        block_name = block_info["name"]
        
        print(f"\n{'='*60}")
        print(f"Processing {block_name} (row={row}, col={col})")
        print(f"{'='*60}")
        
        # Get block metadata
        meta = splitter.get_block_latlon_bounds(row, col)
        
        # Initialize mapper and filter transmitters
        mapper = TransmitterMapper(
            block_meta=meta,
            resolution_m=RM_RESOLUTION_M,
        )
        
        df_tx_block = mapper.filter_transmitters(df_tx_all)
        
        if len(df_tx_block) == 0:
            print(f"  [transmitters] No transmitters in core area of {block_name}")
            tx_map = np.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=np.uint8)
        else:
            tx_map = mapper.create_presence_matrix()
        
        # Write to HDF5
        h5_path = DATASET_DIR / f"{block_name}.h5"

        if not h5_path.exists():
            H5Manager.init_block_file(h5_path, meta, RM_RESOLUTION_M)
        
        H5Manager.write_dataset(
            h5_path,
            H5Manager.DATASET_TRANSMITTERS,
            tx_map,
            dtype='uint8',
        )
        print(f"  [h5] Written transmitters to {h5_path}")
    
    print(f"\nStep 4 complete. Processed {len(all_blocks)} blocks.")


def step5_relocate_transmitters(splitter, all_blocks):
    """
    Step 5: Transmitter Relocation
    
    For each block, relocate transmitters that are not on buildings
    to the nearest unoccupied building within search range.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Transmitter Relocation")
    print("=" * 60)
    
    print(f"Search range: ±{TX_SEARCH_RANGE} pixels (±{TX_SEARCH_RANGE}m)")
    
    relocator = TransmitterRelocator(search_range=TX_SEARCH_RANGE)
    
    total_relocated = 0
    total_unchanged = 0
    total_not_found = 0
    
    for block_info in all_blocks:
        block_name = block_info["name"]
        h5_path = DATASET_DIR / f"{block_name}.h5"
        
        print(f"\n{'='*60}")
        print(f"Processing {block_name}")
        print(f"{'='*60}")
        
        if not h5_path.exists():
            print(f"  [skip] HDF5 file not found: {h5_path}")
            continue
        
        try:
            # Count transmitters before relocation
            with h5py.File(h5_path, "r") as f:
                if "transmitters" not in f or "buildings" not in f:
                    print(f"  [skip] Missing required datasets in {h5_path}")
                    continue
                
                tx_before = f["transmitters"][:]
                buildings = f["buildings"][:]
                
                n_tx_total = int(np.sum(tx_before))
                n_tx_on_building = int(np.sum(tx_before & buildings))
                n_tx_off_building = n_tx_total - n_tx_on_building
            
            # Perform relocation
            relocator.process_file(h5_path)
            
            # Count transmitters after relocation
            with h5py.File(h5_path, "r") as f:
                tx_after = f["transmitters"][:]
                n_tx_on_building_after = int(np.sum(tx_after & buildings))
                n_tx_off_building_after = n_tx_total - n_tx_on_building_after
            
            n_relocated = n_tx_on_building_after - n_tx_on_building
            n_not_found = n_tx_off_building - n_relocated
            
            total_relocated += n_relocated
            total_unchanged += n_tx_on_building
            total_not_found += n_not_found
            
            print(f"  Total TX:                {n_tx_total}")
            print(f"  Already on building:     {n_tx_on_building} (unchanged)")
            print(f"  Relocated to building:   {n_relocated}")
            if n_not_found > 0:
                print(f"  No building in range:    {n_not_found} (stayed in place)")
            
        except Exception as e:
            print(f"  [error] Failed to process {block_name}: {e}")
            continue
    
    print(f"\nStep 5 complete.")
    print(f"Summary across all blocks:")
    print(f"  Total relocated:         {total_relocated}")
    print(f"  Already on building:     {total_unchanged}")
    if total_not_found > 0:
        print(f"  No building in range:    {total_not_found}")


def step6_generate_radiomaps(splitter, all_blocks):
    """
    Step 6: Per-Block Radio Map Generation
    
    For each block, load the XML scene and compute the radio map.
    Supports two transmitter placement modes:
      - Grid mode (USE_GRID_MODE=True):  reads corrected transmitter grid from HDF5
      - CSV mode  (USE_GRID_MODE=False): reads original transmitter locations from CSV
    """
    print("\n" + "=" * 60)
    print("STEP 6: Per-Block Radio Map Generation")
    print("=" * 60)
    print(f"Tx placement mode: {'GRID (from HDF5)' if USE_GRID_MODE else 'CSV (original)'}")
    
    tx_array = setup_antenna_array()
    
    # Ensure dataset directory exists
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    for block_info in all_blocks:
        row = block_info["row"]
        col = block_info["col"]
        block_name = block_info["name"]
        
        print(f"\n{'='*60}")
        print(f"Processing {block_name} (row={row}, col={col})")
        print(f"{'='*60}")
        
        # Get block metadata
        meta = splitter.get_block_latlon_bounds(row, col)
        
        # Paths for this block
        block_dir = XML_DIR / block_name
        xml_path = block_dir / f"{block_name}.xml"
        
        # Check if XML exists
        if not xml_path.exists():
            print(f"  [skip] XML file not found: {xml_path}")
            continue
        
        # Initialize coordinate converter (needed for both modes)
        lat_origin = (meta.lat_min + meta.lat_max) / 2.0
        lon_origin = (meta.lon_min + meta.lon_max) / 2.0
        converter = SceneCoordinateConverter(
            lat_origin,
            lon_origin,
            TERRAIN_HEIGHT,
            LocalCRS.OSM_STORAGE.crs,
            LocalCRS.FRANCE_LAMBERT93.crs,
        )
        
        # Initialize generator
        generator = RadioMapGenerator(converter, meta, RM_RESOLUTION_M)

        # Define h5_path for this block
        h5_path = DATASET_DIR / f"{block_name}.h5"
        
        # Branch on transmitter placement mode
        if USE_GRID_MODE:
            # ---- Grid mode: read transmitter grid from HDF5 ----
            
            if not h5_path.exists():
                print(f"  [skip] HDF5 file not found: {h5_path}")
                continue
            
            with h5py.File(h5_path, "r") as f:
                if "transmitters" not in f:
                    print(f"  [skip] 'transmitters' dataset not found in {h5_path}")
                    continue
                tx_grid = f["transmitters"][:]
            
            if not np.any(tx_grid):
                print(f"  [skip] No transmitters in grid for {block_name}")
                # Ensure HDF5 file exists and clean any stale radiomap
                if h5_path.exists():
                    H5Manager.init_block_file(h5_path, meta, RM_RESOLUTION_M)
                    H5Manager.clean_dataset(h5_path, H5Manager.DATASET_RADIOMAP)
                    print(f"  [h5] Cleared radiomap dataset in {h5_path}")
                continue
            
            # Generate radio map from grid
            rss_map = generator.generate_from_grid(
                xml_path=xml_path,
                tx_grid=tx_grid,
                tx_array=tx_array,
                frequency=FREQUENCY,
            )
        else:
            # ---- CSV mode: filter transmitters from CSV ----
            rss_map = generator.generate(
                xml_path=xml_path,
                csv_path=TRANSMITTER_PATH,
                tx_array=tx_array,
                frequency=FREQUENCY,
            )

        if not h5_path.exists():
            H5Manager.init_block_file(h5_path, meta, RM_RESOLUTION_M)
        
        if rss_map is None:
            print(f"  [skip] No transmitters in core area of {block_name}")
            if h5_path.exists():
                H5Manager.clean_dataset(h5_path, H5Manager.DATASET_RADIOMAP)
                print(f"  [h5] Cleared radiomap dataset in {h5_path}")
        else:
            print(f"  [done] RSS map shape: {rss_map.shape}, "
                  f"dtype: {rss_map.dtype}, "
                  f"min: {rss_map.min():.6e}, max: {rss_map.max():.6e}")
            H5Manager.write_dataset(
                h5_path,
                H5Manager.DATASET_RADIOMAP,
                rss_map,
                dtype='float32',
            )
            print(f"  [h5] Written radiomap to {h5_path}")
    
    print(f"\nStep 6 complete. Processed {len(all_blocks)} blocks.")


def step7_merge_blocks():
    """
    Step 7: Merge Block HDF5 Files
    
    Merge all individual block HDF5 files into a single unified file.
    """
    print("\n" + "=" * 60)
    print("STEP 7: Merge Block HDF5 Files")
    print("=" * 60)
    
    print(f"Merging blocks from: {DATASET_DIR}")
    print(f"Target block size:   {MERGED_BLOCK_SIZE_M} m")
    print(f"Output:              {MERGED_FILENAME}")
    
    # Ensure output directory exists
    MERGED_FILENAME.parent.mkdir(parents=True, exist_ok=True)
    
    H5Manager.merge_blocks(
        input_dir=DATASET_DIR,
        new_block_size_m=MERGED_BLOCK_SIZE_M,
        output_path=MERGED_FILENAME,
    )
    
    if MERGED_FILENAME.exists():
        with h5py.File(MERGED_FILENAME, "r") as f:
            N = f.attrs["dataset_size"]
            print(f"\nMerged {N} samples into {MERGED_FILENAME}")
            print(f"  buildings:    {f['buildings'].shape}  | dtype={f['buildings'].dtype}")
            print(f"  transmitters: {f['transmitters'].shape}  | dtype={f['transmitters'].dtype}")
            print(f"  radiomap:     {f['radiomap'].shape}  | dtype={f['radiomap'].dtype}")
    else:
        print("\nNo output file created (no valid samples).")
    
    print("\nStep 7 complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Paris Radio Map Generation Pipeline"
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip the merge step (only process individual blocks)"
    )
    parser.add_argument(
        "--start-step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        default=1,
        help="Start from step N (1-7, default: 1)"
    )
    parser.add_argument(
        "--end-step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="End at step N (1-7, default: 7 if not --skip-merge else 6)"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Run only step N (equivalent to --start-step N --end-step N)"
    )
    parser.add_argument(
        "--skip-xml-check",
        action="store_true",
        help="Skip checking if XML directory exists (for Step 6 when XML not yet generated)"
    )
    parser.add_argument(
        "--clear-osm-cache",
        action="store_true",
        help="Clear the OSM chunk cache before running (forces re-download)"
    )
    args = parser.parse_args()
    
    # Handle --step as shorthand for single step
    if args.step is not None:
        args.start_step = args.step
        args.end_step = args.step
    
    # Set default end_step
    if args.end_step is None:
        if args.skip_merge:
            args.end_step = 6
        else:
            args.end_step = 7
    
    # Validate step range
    if args.start_step > args.end_step:
        print(f"Error: start_step ({args.start_step}) cannot be greater than end_step ({args.end_step})")
        sys.exit(1)
    
    print("=" * 60)
    print("Paris Radio Map Generation Pipeline")
    print("=" * 60)
    print(f"Running steps: {args.start_step} -> {args.end_step}")
    print(f"XML Directory:      {XML_DIR}")
    print(f"Dataset Directory:  {DATASET_DIR}")
    print(f"Merged Output:      {MERGED_FILENAME}")
    print(f"Block Size:         {BLOCK_SIZE_M} m")
    print(f"Overlap:            {OVERLAP_M} m")
    print(f"Resolution:         {RM_RESOLUTION_M} m")
    print(f"Frequency:          {FREQUENCY / 1e9:.1f} GHz")
    print(f"Tx Placement Mode:  {'GRID (HDF5)' if USE_GRID_MODE else 'CSV (original)'}") 
    print(f"TX Search Range:    ±{TX_SEARCH_RANGE} m")
    print(f"Skip Merge:         {args.skip_merge}")
    print(f"OSM Cache:          data/osm_cache/ (auto-managed)")
    print("=" * 60)
    
    # Clear OSM cache if requested
    if args.clear_osm_cache:
        import shutil
        from utils.osm_fetcher import CHUNKS_DIR, METADATA_DIR
        if CHUNKS_DIR.exists():
            shutil.rmtree(CHUNKS_DIR)
            print(f"Cleared chunk cache: {CHUNKS_DIR}")
        if METADATA_DIR.exists():
            shutil.rmtree(METADATA_DIR)
            print(f"Cleared metadata cache: {METADATA_DIR}")
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if XML directory exists for steps that need it (Step 6)
    if args.start_step >= 6 and not XML_DIR.exists() and not args.skip_xml_check:
        print(f"\nWarning: XML directory '{XML_DIR}' does not exist.")
        print("Step 6 requires pre-generated PLY and XML files.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting.")
            sys.exit(0)
    
    # Initialize tile splitter (needed for all steps)
    splitter = TileSplitter(
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
        block_size_m=BLOCK_SIZE_M,
        overlap_m=OVERLAP_M,
        stride_m=STRIDE_M
    )
    
    all_blocks = None
    
    # Run steps
    for step in range(args.start_step, args.end_step + 1):
        # Get all_blocks when needed (Step 1 doesn't need it to return blocks)
        if step >= 2 and all_blocks is None:
            all_blocks = splitter.get_all_blocks(tx_csv_path=TRANSMITTER_PATH)
            print(f"\nTotal blocks to process: {len(all_blocks)}")
        
        if step == 1:
            all_blocks = step1_split_region(splitter)
        elif step == 2:
            step2_generate_ply_and_xml(splitter, all_blocks)
        elif step == 3:
            step3_generate_building_rasters(splitter, all_blocks)
        elif step == 4:
            step4_generate_transmitter_rasters(splitter, all_blocks)
        elif step == 5:
            step5_relocate_transmitters(splitter, all_blocks)
        elif step == 6:
            step6_generate_radiomaps(splitter, all_blocks)
        elif step == 7:
            if not args.skip_merge:
                step7_merge_blocks()
            else:
                print("\nSkipping merge step as requested.")
    
    print("\n" + "=" * 60)
    print("Pipeline execution completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()