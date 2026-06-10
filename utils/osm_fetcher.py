import osmnx as ox
import geopandas as gpd
from typing import Dict, List, Union

# Buildings
OSM_BUILDING_TAGS: Dict[str, Union[bool, str, List[str]]] = {
    "building": True  # Get all types of buildings
}

# Railways
OSM_RAILWAY_TAGS: Dict[str, Union[bool, str, List[str]]] = {
    "railway": [
        "rail",
        "subway",
        "tram",
        "light_rail",
        "station",
    ]  # Main railway features
}

# Water bodies
OSM_WATER_TAGS: Dict[str, Union[bool, str, List[str]]] = {
    "natural": ["water", "wetland"],
    "waterway": ["river", "stream", "canal"],  # Linear water features
}

# Forests and green areas
OSM_FOREST_TAGS: Dict[str, Union[bool, str, List[str]]] = {
    "natural": ["wood", "scrub"],
    "landuse": ["forest", "orchard"],
    "leisure": ["nature_reserve"],  # Protected green spaces
}

# Roads and highways
OSM_ROAD_TAGS: Dict[str, Union[bool, str, List[str]]] = {
    "highway": [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "service",
    ]
}

# Combined OSM tags template including buildings, railways, water, forest, and roads
OSM_ALL_TAGS: Dict[str, Union[bool, str, List[str]]] = {
    "building": True,  # Get all types of buildings
    "railway": [
        "rail",
        "subway",
        "tram",
        "light_rail",
        "station",
    ],  # Main railway features
    "natural": [
        "water",
        "wetland",
        "wood",
        "scrub",
    ],  # Combined water and forest natural features
    "waterway": ["river", "stream", "canal"],  # Linear water features
    "landuse": ["forest", "orchard"],  # Forest land use areas
    "leisure": ["nature_reserve"],  # Protected green spaces
    "highway": [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "service",
    ],  # Main and service roads
}



class OSMFetcher:
    """A class to fetch OpenStreetMap data and convert it to GeoDataFrame."""

    def __init__(self):
        # Configure osmnx cache to save time and network data
        ox.settings.use_cache = True
        ox.settings.log_console = False

    def fetch_by_bbox(
        self,
        bbox: tuple[float, float, float, float],
        tags: Dict[str, Union[bool, str, List[str]]],
    ) -> gpd.GeoDataFrame:
        """Fetch OSM data using a bounding box and return a GeoDataFrame.

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
                return gpd.GeoDataFrame()

            print(f"Successfully fetched {len(gdf)} features.")
            return gdf

        except Exception as e:
            print(f"Error while fetching OSM data: {e}")
            return gpd.GeoDataFrame()