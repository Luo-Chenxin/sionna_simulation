import osmnx as ox
import geopandas as gpd
import pandas as pd
from typing import Dict, List, Union

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
        # If the internal GeoDataFrame is empty, return an empty GeoDataFrame immediately
        if self._raw_gdf.empty:
            return gpd.GeoDataFrame()

        # Start with all True values (match everything initially)
        combined_condition = pd.Series(True, index=self._raw_gdf.index)
        
        for key, val in filter_tags.items():
            # If the column does not exist in the dataframe, nothing can match this criteria
            if key not in self._raw_gdf.columns:
                return gpd.GeoDataFrame()
                
            if val is True:
                # Match rows where the column is not null and not explicitly False
                current_condition = self._raw_gdf[key].notna() & (self._raw_gdf[key] != False)
            elif isinstance(val, str):
                # Match rows where the column value exactly equals the string
                current_condition = (self._raw_gdf[key] == val)
            elif isinstance(val, list):
                # Match rows where the column value is within the given list
                current_condition = self._raw_gdf[key].isin(val)
            else:
                # Handle unsupported types gracefully
                current_condition = pd.Series(False, index=self._raw_gdf.index)
                
            # Combine the current condition with the main mask using bitwise AND
            combined_condition &= current_condition
            
        # Return the filtered GeoDataFrame copy
        return self._raw_gdf[combined_condition].copy()