import osmnx as ox
import geopandas as gpd
import pandas as pd
from typing import Dict, List, Union

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
    ):
        """
        Initialize the fetcher, configure cache, and automatically fetch the base data.
        """
        # Configure osmnx cache to save time and network data
        ox.settings.use_cache = True
        ox.settings.log_console = False
        
        # Initialize internal storage
        self._raw_gdf = gpd.GeoDataFrame()
        
        # Automatically fetch data during initialization
        self._fetch_all(bbox, tags)

    def _fetch_all(
        self,
        bbox: tuple[float, float, float, float],
        tags: Dict[str, Union[bool, str, List[str]]],
    ) -> None:
        """
        Fetch base OSM data using a bounding box and store it internally.
        
        Parameters
        ----------
        bbox
            Bounding box as `(left, bottom, right, top)`. Coordinates should be in
            unprojected latitude-longitude degrees (EPSG:4326).
        """
        print(f"Fetching OSM data for bbox: {bbox}")

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
        filter_tags: Dict[str, Union[bool, str, List[str]]]
    ) -> gpd.GeoDataFrame:
        """
        Filter the pre-fetched GeoDataFrame based on a new tags dictionary.
        
        Parameters
        ----------
        filter_tags
            A dictionary where keys are column names and values are the filter criteria.
        """
        combined_condition = pd.Series(False, index=self._raw_gdf.index)
        
        for key, val in filter_tags.items():
            # Skip columns that don't exist
            if key not in self._raw_gdf.columns:
                continue
            
            col = self._raw_gdf[key]
            
            if val == True:
                # Match rows where the column has any meaningful value
                current_condition = col.notna() & (col != False) & (col != '')
            elif isinstance(val, str):
                # Match rows where the column value exactly equals the string
                # Also handle list-like string columns from OSM (semicolon-separated)
                current_condition = (col == val) | col.str.contains(f'(^|;){val}(;|$)', na=False, regex=True)
            elif isinstance(val, list):
                # Match rows where the column value is in the list
                # Also handle list-like string columns
                base_condition = col.isin(val)
                list_condition = pd.Series(False, index=col.index)
                for v in val:
                    if isinstance(v, str):
                        list_condition |= col.str.contains(f'(^|;){v}(;|$)', na=False, regex=True)
                current_condition = base_condition | list_condition
            else:
                continue
        
            # Use OR to combine: match if ANY tag condition is met
            combined_condition |= current_condition
            
        # Return the filtered GeoDataFrame copy
        return self._raw_gdf[combined_condition].copy()