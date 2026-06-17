import osmnx as ox
import geopandas as gpd
gpd.options.io_engine = "fiona"
import pandas as pd
from typing import Dict, List, Union, Optional
from shapely.geometry import box
from pathlib import Path
import numpy as np
import time 

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
        """
        # Configure osmnx cache to save time and network data
        ox.settings.use_cache = True
        ox.settings.log_console = False
        # ox.settings.requests_timeout = 180
        # ox.settings.overpass_url = 'https://overpass.private.coffee/api/interpreter'
        
        # Initialize internal storage
        self._raw_gdf = gpd.GeoDataFrame()
        
        # Store the full extent for reference
        self._full_bbox = bbox
        
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
        
        Parameters
        ----------
        bbox
            Bounding box as `(left, bottom, right, top)`.
        tags
            OSM tags dictionary to query features.
        filepath
            Path to the local GeoPackage file.
        """
        import os
        
        if os.path.exists(filepath):
            print(f"Loading OSM data from local file: {filepath}")
            try:
                # Load the entire file (spatial filtering can be applied later)
                self._raw_gdf = gpd.read_file(filepath)
                print(f"Successfully loaded {len(self._raw_gdf)} features from local file.")
            except Exception as e:
                print(f"Error loading local file: {e}")
                print("Falling back to online fetch...")
                self._fetch_all(bbox, tags)
                if not self._raw_gdf.empty:
                    self._raw_gdf.to_file(filepath, driver="GPKG")
        else:
            print(f"Local file not found: {filepath}")
            print("Downloading from OSM...")
            self._fetch_all(bbox, tags)
            if not self._raw_gdf.empty:
                print(f"Saving downloaded data to: {filepath}")
                self._raw_gdf.to_file(filepath, driver="GPKG")

    def _fetch_all(
        self,
        bbox: tuple[float, float, float, float],
        tags: Dict[str, Union[bool, str, List[str]]],
    ) -> None:
        """
        Fetch base OSM data using a bounding box and store it internally.
        Automatically splits large areas into smaller chunks to avoid memory issues.
        Supports resume: saves intermediate chunks to disk to avoid re-downloading.
        
        Parameters
        ----------
        bbox
            Bounding box as `(left, bottom, right, top)`. Coordinates should be in
            unprojected latitude-longitude degrees (EPSG:4326).
        tags
            OSM tags dictionary to query features.
        """
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
            
            # Create temp directory for chunk cache
            import hashlib
            cache_dir = Path('data') / "osm_chunks"
            cache_dir.mkdir(exist_ok=True)
            
            # Generate unique cache prefix based on bbox and tags
            cache_key = hashlib.md5(f"{bbox}{sorted(tags.keys())}".encode()).hexdigest()[:8]
            
            tmp_files = []
            for i in range(n_splits):
                for j in range(n_splits):
                    sub_bbox = (lon_edges[j], lat_edges[i], lon_edges[j+1], lat_edges[i+1])
                    chunk_file = cache_dir / f"{cache_key}_chunk_{i}_{j}.gpkg"
                    
                    # Check if chunk already downloaded
                    if chunk_file.exists():
                        try:
                            test_read = gpd.read_file(chunk_file)
                            if len(test_read) >= 0:
                                print(f"  Chunk ({i},{j}): loading from cache {chunk_file.name}")
                                tmp_files.append(chunk_file)
                                del test_read
                                continue
                            else:
                                print(f"  Chunk ({i},{j}): cache file is empty or corrupted, re-downloading...")
                                chunk_file.unlink()
                        except Exception as read_error:
                            print(f"  Chunk ({i},{j}): cache file corrupted ({read_error}), re-downloading...")
                            chunk_file.unlink()
                    
                    print(f"  Chunk ({i},{j}): downloading...")
                    try:
                        gdf_chunk = ox.features_from_bbox(bbox=sub_bbox, tags=tags)
                        if not gdf_chunk.empty:
                            if 'ID' in gdf_chunk.columns:
                                gdf_chunk = gdf_chunk.drop(columns=['ID'])
                            # Save chunk immediately
                            gdf_chunk.to_file(chunk_file, driver="GPKG")
                            tmp_files.append(chunk_file)
                            print(f"    -> {len(gdf_chunk)} features (saved to cache)")
                            del gdf_chunk
                        else:
                            # Create empty file as marker to skip next time
                            chunk_file.touch()
                            print(f"    -> empty")
                        time.sleep(2)
                    except Exception as e:
                        print(f"    -> Failed: {e}")
                        # Don't create cache file on failure, so it retries next time
                        if chunk_file.exists():
                            chunk_file.unlink()
                            print(f"    -> Removed incomplete cache file")
            
            if tmp_files:
                self._raw_gdf = gpd.GeoDataFrame()
                for tmp_file in tmp_files:
                    chunk = gpd.read_file(tmp_file)
                    self._raw_gdf = pd.concat([self._raw_gdf, chunk], ignore_index=True)
                    print(f"  Merged {len(chunk)} features from {tmp_file.name}")
                    del chunk
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
        if self._raw_gdf.empty:
            return self._raw_gdf
        
        # Start with all features
        filtered_gdf = self._raw_gdf
        
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