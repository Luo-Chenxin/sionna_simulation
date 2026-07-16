import osmnx as ox
import geopandas as gpd
import pandas as pd
from typing import Dict, List, Union, Optional
from shapely.geometry import box
from pathlib import Path
import numpy as np
import time 
import json

# =============================================================================
# Path Configuration - Centralized directory management
# =============================================================================

# Base directory for all OSM cache data
CACHE_BASE_DIR = Path("data/osm_cache")

# Directory for chunk GeoPackage files
CHUNKS_DIR = CACHE_BASE_DIR / "chunks"

# Directory for metadata files (JSON)
METADATA_DIR = CACHE_BASE_DIR / "metadata"

# =============================================================================
# OSM Feature Tags for ox.features_from_bbox
# These tags are organized by feature categories and combined views.
# =============================================================================

# --- Individual Category Tags ---

# Buildings: all types of structures
TAGS_BUILDINGS: Dict[str, Union[bool, str, List[str]]] = {
    "building": True,
}

# Railways: tracks, stations, and related infrastructure
TAGS_RAILWAYS: Dict[str, Union[bool, str, List[str]]] = {
    "railway": True,
}

# Water bodies: rivers, lakes, ponds, reservoirs, etc.
TAGS_WATER: Dict[str, Union[bool, str, List[str]]] = {
    "natural": ["water", "bay", "strait"],
    "waterway": True,
    "water": True,
}

# Forests and wooded areas
TAGS_FOREST: Dict[str, Union[bool, str, List[str]]] = {
    "landuse": ["forest", "forestry"],
    "natural": ["wood", "scrub", "tree_row"],
    "leisure": "park",  # parks often contain trees
}

# Roads: highways of all levels (motorway to residential to footpath)
TAGS_ROADS: Dict[str, Union[bool, str, List[str]]] = {
    "highway": True,
}

# --- Combined Tags ---

# Complete urban scene: buildings + roads + railways + water + vegetation
TAGS_COMPLETE: Dict[str, Union[bool, str, List[str]]] = {
    "building": True,
    "highway": True,
    "railway": True,
    "waterway": True,
    "water": True,
    "natural": ["water", "wood", "scrub", "tree_row", "bay", "strait"],
    "landuse": ["forest", "forestry"],
    "leisure": "park",
}

# --- Tag Collection for Easy Reference ---

# Dictionary mapping tag names to their definitions
OSM_TAGS: Dict[str, Dict[str, Union[bool, str, List[str]]]] = {
    "buildings": TAGS_BUILDINGS,
    "railways": TAGS_RAILWAYS,
    "water": TAGS_WATER,
    "forest": TAGS_FOREST,
    "roads": TAGS_ROADS,
    "complete": TAGS_COMPLETE,
}

class OSMFetcher:
    """A class to fetch OpenStreetMap data and filter it dynamically."""

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        tags: Dict[str, Union[bool, str, List[str]]],
        cache_filepath: Optional[Path] = None,
    ):
        """
        Initialize the fetcher, configure cache, and automatically fetch the base data.
        
        Parameters
        ----------
        bbox
            Bounding box as `(left, bottom, right, top)`. Coordinates should be in
            unprojected latitude-longitude degrees (EPSG:4326).
        tags
            OSM tags dictionary to query features.
        cache_filepath
            Optional path to a local GeoPackage file. If provided, data will be
            loaded from this file instead of downloading from OSM. If the file
            does not exist, data will be downloaded and saved to this path.
            For large areas, data may be stored as multiple chunk files in a
            subdirectory.
        """
        # Configure osmnx cache to save time and network data
        ox.settings.use_cache = True
        ox.settings.log_console = False
        
        # Make sure cache directories exist
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize internal storage
        self._raw_gdf = gpd.GeoDataFrame()
        self._chunk_files: List[Path] = []  # For large areas, store chunk file paths
        self._chunk_metadata: Dict[str, tuple] = {}  # Bounding boxes for each chunk
        
        # Store the full extent for reference
        self._full_bbox = bbox
        self._tags = tags
        
        # Automatically fetch or load data during initialization
        if cache_filepath is not None:
            self._load_or_fetch(bbox, tags, cache_filepath)
        else:
            self._fetch_all(bbox, tags)

    def _load_or_fetch(
        self,
        bbox: tuple[float, float, float, float],
        tags: Dict[str, Union[bool, str, List[str]]],
        filepath: Path,
    ) -> None:
        """
        Load data from a local GeoPackage file, or download and save if not exists.
        For large areas, supports loading from chunk files.
        
        Parameters
        ----------
        bbox
            Bounding box as `(left, bottom, right, top)`.
        tags
            OSM tags dictionary to query features.
        filepath
            Path to the local GeoPackage file. If data is split into chunks,
            they will be stored in CHUNKS_DIR with a unique prefix.
        """
        # Check if single file cache exists
        if filepath.exists():
            print(f"Loading OSM data from local file: {filepath}")
            try:
                self._raw_gdf = gpd.read_file(filepath)
                print(f"Successfully loaded {len(self._raw_gdf)} features from local file.")
                return
            except Exception as e:
                print(f"Error loading local file: {e}")
                print("Falling back to online fetch...")
        
        # Check for chunk files using unique cache prefix
        # import hashlib
        # cache_key = hashlib.md5(f"{bbox}{sorted(tags.keys())}".encode()).hexdigest()[:8]
        cache_key = '9ea77292'
        chunk_pattern = f"{cache_key}_chunk_*_*.gpkg"
        chunk_files = sorted(CHUNKS_DIR.glob(chunk_pattern))
        
        if chunk_files:
            print(f"Found {len(chunk_files)} cached chunks in {CHUNKS_DIR}")
            print("Using lazy loading mode - chunks will be loaded on demand.")
            self._chunk_files = chunk_files
            self._load_chunk_metadata(cache_key)
            return
        
        # Download from OSM
        print(f"Local file not found: {filepath}")
        print("Downloading from OSM...")
        self._fetch_all(bbox, tags)

    def _load_chunk_metadata(self, cache_key: str) -> None:
        """
        Load bounding box metadata for chunk files from JSON file.
        Falls back to reading from files if metadata file doesn't exist.
        
        Parameters
        ----------
        cache_key
            Unique key used to identify this cache session.
        """
        if not self._chunk_files:
            return
        
        metadata_file = METADATA_DIR / f"{cache_key}_chunks_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self._chunk_metadata = json.load(f)
                print(f"Loaded metadata for {len(self._chunk_metadata)} chunks")
                return
            except Exception:
                pass
        
        # If metadata file doesn't exist, read from chunk files
        self._save_chunk_metadata(cache_key)

    def _save_chunk_metadata(self, cache_key: str) -> None:
        """
        Save bounding box metadata for chunk files to JSON for faster queries.
        
        Parameters
        ----------
        cache_key
            Unique key used to identify this cache session.
        """
        if not self._chunk_files:
            return
        
        metadata = {}
        for chunk_file in self._chunk_files:
            try:
                # Read just the bounds without loading all data
                with gpd.read_file(chunk_file) as src:
                    metadata[str(chunk_file)] = tuple(src.total_bounds)
            except Exception:
                continue
        
        if metadata:
            metadata_file = METADATA_DIR / f"{cache_key}_chunks_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            self._chunk_metadata = metadata
            print(f"Saved metadata for {len(metadata)} chunks")

    def _fetch_all(
        self,
        bbox: tuple[float, float, float, float],
        tags: Dict[str, Union[bool, str, List[str]]],
    ) -> None:
        """
        Fetch base OSM data using a bounding box and store it internally.
        Automatically splits large areas into smaller chunks to avoid memory issues.
        
        Parameters
        ----------
        bbox
            Bounding box as `(left, bottom, right, top)`. Coordinates should be in
            unprojected latitude-longitude degrees (EPSG:4326).
        tags
            OSM tags dictionary to query features.
        """
        import hashlib
        
        print(f"Fetching OSM data for bbox: {bbox}")

        lat_span = bbox[3] - bbox[1]
        lon_span = bbox[2] - bbox[0]
        area_km2 = lat_span * 111 * lon_span * 111 * np.cos(np.radians((bbox[1] + bbox[3]) / 2))
        
        # Split if area > 20 km² (adjustable threshold)
        if area_km2 > 20:
            n_splits = max(2, int(np.ceil(np.sqrt(area_km2 / 10))))
            lat_edges = np.linspace(bbox[1], bbox[3], n_splits + 1)
            lon_edges = np.linspace(bbox[0], bbox[2], n_splits + 1)
            
            print(f"Large area ({area_km2:.1f} km²), splitting into {n_splits}x{n_splits} chunks...")
            
            # Generate unique cache prefix based on bbox and tags
            cache_key = hashlib.md5(f"{bbox}{sorted(tags.keys())}".encode()).hexdigest()[:8]
            
            chunk_files = []
            for i in range(n_splits):
                for j in range(n_splits):
                    sub_bbox = (lon_edges[j], lat_edges[i], lon_edges[j+1], lat_edges[i+1])
                    chunk_file = CHUNKS_DIR / f"{cache_key}_chunk_{i}_{j}.gpkg"
                    
                    # Check if chunk already downloaded
                    if chunk_file.exists():
                        print(f"  Chunk ({i},{j}): loading from cache {chunk_file.name}")
                        chunk_files.append(chunk_file)
                        continue
                    
                    print(f"  Chunk ({i},{j}): downloading...")
                    try:
                        gdf_chunk = ox.features_from_bbox(bbox=sub_bbox, tags=tags)
                        if not gdf_chunk.empty:
                            # Safely drop columns that may or may not exist
                            columns_to_drop = ['ID', 'Company']
                            existing_cols = [col for col in columns_to_drop if col in gdf_chunk.columns]
                            if existing_cols:
                                gdf_chunk = gdf_chunk.drop(columns=existing_cols)
                            
                            # Save chunk to permanent cache
                            gdf_chunk.to_file(chunk_file, driver="GPKG")
                            chunk_files.append(chunk_file)
                            print(f"    -> {len(gdf_chunk)} features (saved to cache)")
                            del gdf_chunk
                        else:
                            # Create empty file as marker to skip next time
                            chunk_file.touch()
                            print(f"    -> empty")
                        time.sleep(2)
                    except Exception as e:
                        print(f"    -> Failed: {e}")
                        # Remove incomplete file on failure so it retries next time
                        if chunk_file.exists():
                            chunk_file.unlink()
                            print(f"    -> Removed incomplete cache file")
            
            if chunk_files:
                self._chunk_files = chunk_files
                self._save_chunk_metadata(cache_key)
                print(f"Successfully processed {len(chunk_files)} chunks (lazy loading mode).")
            else:
                print("Warning: No data found for the given box and tags.")
                self._raw_gdf = gpd.GeoDataFrame()
        else:
            # Small area, download directly
            try:
                gdf = ox.features_from_bbox(bbox=bbox, tags=tags)

                if gdf.empty:
                    print("Warning: No data found for the given box and tags.")
                    self._raw_gdf = gpd.GeoDataFrame()
                    return

                print(f"Successfully fetched {len(gdf)} features.")
                self._raw_gdf = gdf

            except Exception as e:
                print(f"Error while fetching OSM data: {e}")
                self._raw_gdf = gpd.GeoDataFrame()

    def get_filtered_features(
        self, 
        filter_tags: Optional[Dict[str, Union[bool, str, List[str]]]] = None,
        sub_bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Filter the pre-fetched GeoDataFrame based on tags and/or a sub-bounding box.
        For chunked data (large areas), only loads relevant chunks to save memory.
        
        Lazy loading explanation:
        When multiple .gpkg chunk files exist, they are NOT loaded into memory
        until this method is called. At that point, only the chunks that intersect
        with sub_bbox are loaded. This avoids loading the entire dataset at once.
        
        Parameters
        ----------
        filter_tags
            A dictionary where keys are column names and values are the filter criteria.
            If None, no tag filtering is applied.
        sub_bbox
            Sub-bounding box as `(left, bottom, right, top)` in EPSG:4326.
            If provided, only features intersecting this box are returned.
            If None, no spatial filtering is applied.
            
        Returns
        -------
        gpd.GeoDataFrame
            A filtered copy of the GeoDataFrame.
        """
        # If we have the full dataset loaded in memory, use it directly
        if not self._raw_gdf.empty:
            return self._filter_gdf(self._raw_gdf, filter_tags, sub_bbox)
        
        # If using chunked data, load only relevant chunks
        if self._chunk_files:
            if sub_bbox is not None:
                # Find which chunks overlap with the query bbox
                # Only these chunks will be read from disk
                relevant_chunks = self._find_relevant_chunks(sub_bbox)
                if relevant_chunks:
                    print(f"Loading {len(relevant_chunks)} relevant chunks out of {len(self._chunk_files)}")
                    combined_gdf = gpd.GeoDataFrame()
                    for chunk_file in relevant_chunks:
                        chunk = gpd.read_file(chunk_file)
                        if not chunk.empty:
                            combined_gdf = pd.concat([combined_gdf, chunk], ignore_index=True)
                    return self._filter_gdf(combined_gdf, filter_tags, sub_bbox)
                else:
                    print("No relevant chunks found for the given sub_bbox.")
                    return gpd.GeoDataFrame()
            else:
                # No spatial filter, but need to apply tag filters
                # Must load all chunks since we don't know which ones have matching tags
                print(f"Loading all {len(self._chunk_files)} chunks for tag filtering")
                combined_gdf = gpd.GeoDataFrame()
                for chunk_file in self._chunk_files:
                    chunk = gpd.read_file(chunk_file)
                    if not chunk.empty:
                        combined_gdf = pd.concat([combined_gdf, chunk], ignore_index=True)
                return self._filter_gdf(combined_gdf, filter_tags, None)
        
        # No data available
        return gpd.GeoDataFrame()

    def _find_relevant_chunks(
        self, 
        sub_bbox: tuple[float, float, float, float]
    ) -> List[Path]:
        """
        Find chunk files that intersect with the given bounding box.
        Uses cached metadata (JSON) if available, otherwise reads from files.
        Only reads first 10 rows of each chunk file when metadata is missing.
        
        Parameters
        ----------
        sub_bbox
            Bounding box as `(left, bottom, right, top)`.
            
        Returns
        -------
        List[Path]
            List of chunk file paths that intersect with sub_bbox.
        """
        from shapely.geometry import box as shapely_box
        
        query_box = shapely_box(*sub_bbox)
        relevant_chunks = []
        
        for chunk_file in self._chunk_files:
            # Try to use cached metadata first (fast, no file read needed)
            chunk_key = str(chunk_file)
            if chunk_key in self._chunk_metadata:
                chunk_bounds = self._chunk_metadata[chunk_key]
                chunk_box = shapely_box(*chunk_bounds)
                if query_box.intersects(chunk_box):
                    relevant_chunks.append(chunk_file)
            else:
                # Fall back to reading bounds from file (slower)
                # Only reads first 10 rows to get the extent
                try:
                    chunk_gdf = gpd.read_file(chunk_file, rows=10)
                    if not chunk_gdf.empty:
                        chunk_bounds = chunk_gdf.total_bounds
                        chunk_box = shapely_box(*chunk_bounds)
                        if query_box.intersects(chunk_box):
                            relevant_chunks.append(chunk_file)
                except Exception:
                    continue
        
        return relevant_chunks

    def _filter_gdf(
        self, 
        gdf: gpd.GeoDataFrame, 
        filter_tags: Optional[Dict[str, Union[bool, str, List[str]]]] = None,
        sub_bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Apply tag and spatial filters to a GeoDataFrame.
        
        Parameters
        ----------
        gdf
            GeoDataFrame to filter.
        filter_tags
            Tag filter dictionary. If None, no tag filtering is applied.
        sub_bbox
            Sub-bounding box. If None, no spatial filtering is applied.
            
        Returns
        -------
        gpd.GeoDataFrame
            Filtered GeoDataFrame.
        """
        if gdf.empty:
            return gdf
        
        # Start with all features
        filtered_gdf = gdf
        
        # Apply spatial filter if a sub-bbox is provided
        if sub_bbox is not None:
            left, bottom, right, top = sub_bbox
            clip_box = box(left, bottom, right, top)
            spatial_mask = filtered_gdf.intersects(clip_box)
            filtered_gdf = filtered_gdf[spatial_mask]
        
        # Apply tag filter if provided
        if filter_tags is not None:
            combined_condition = pd.Series(False, index=filtered_gdf.index)
            
            for key, val in filter_tags.items():
                # Skip columns that don't exist
                if key not in filtered_gdf.columns:
                    continue
                
                col = filtered_gdf[key]
                
                if val == True:
                    # Match rows where the column has any meaningful value
                    current_condition = col.notna() & (col != False) & (col != '')
                elif isinstance(val, str):
                    # Match rows where the column value exactly equals the string
                    # Also handle list-like string columns from OSM (semicolon-separated)
                    current_condition = (col == val) | col.str.contains(
                        f'(?:^|;){val}(?:;|$)', na=False, regex=True
                    )
                elif isinstance(val, list):
                    # Match rows where the column value is in the list
                    # Also handle list-like string columns
                    base_condition = col.isin(val)
                    list_condition = pd.Series(False, index=col.index)
                    for v in val:
                        if isinstance(v, str):
                            list_condition |= col.str.contains(
                                f'(?:^|;){v}(?:;|$)', na=False, regex=True
                            )
                    current_condition = base_condition | list_condition
                else:
                    continue
            
                # Use OR to combine: match if ANY tag condition is met
                combined_condition |= current_condition
                
            filtered_gdf = filtered_gdf[combined_condition]
        
        return filtered_gdf.copy()