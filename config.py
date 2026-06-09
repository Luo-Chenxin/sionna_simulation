from pathlib import Path
from enum import Enum
from pyproj import CRS

PRE_TRANSMITTER_FILE_ENCODE = 'cp1252'
FILE_ENCODE = 'utf-8-sig'

DEFAULT_AZIMUTH_MUTIPLITER=1
DEFAULT_OFF_SET=0
DEFAULT_PITCH=0
DEFAULT_ROLL=0

DISPLAY_RADIUS=None
DEFAULT_TRANSIMITTER_DBM=40

# Material Database (Template)
# You can add your own materials here
# Note: "freshwater" does not use a,b,c,d parameters. It uses a specific function instead.
MATERIAL_DATABASE = {
    "asphalt_concrete": {
        "type": "itur_abcd",
        "a": 4.83, "b": 0.0, "c": 0.0108, "d": 1.3969,
        "f_min": 1.0, "f_max": 40.0,
        "color": (0.12, 0.12, 0.13), # Dark gray/asphalt
    },
    "freshwater": {
        "type": "itu_p527",
        "f_min": 0.1,  # Set your preferred minimum frequency in GHz
        "f_max": 1000.0, # Set your preferred maximum frequency in GHz
        "color": (0.00, 0.15, 0.75), # Deep blue water
    },
}

DEFAULT_SALINITY = 0.5   # Unit: g/kg or ppt
DEFAULT_TEMPERATURE = 20.0  # Unit: celsius

class LocalCRS(Enum):
    """
    Spatial Reference System (SRS) macro definitions for Île-de-France (Paris) and OSM.
    """
    
    # Raw OSM data storage coordinate system (Geographic: Latitude/Longitude in degrees)
    OSM_STORAGE = "EPSG:4326"
    
    # Paris region planar projection (Standard global UTM Zone 31N, units in meters)
    PARIS_UTM = "EPSG:32631"
    
    # Official French national projection (Lambert-93, units in meters)
    # Strongly recommended for local data integration in France.
    FRANCE_LAMBERT93 = "EPSG:2154"
    
    # Web map tile projection (Used for map rendering/displaying in Folium, Leaflet, etc.)
    WEB_MERCATOR = "EPSG:3857"

    @property
    def crs(self) -> CRS:
        """Dynamically retrieves the pyproj CRS object to avoid redundant initialization."""
        return CRS.from_user_input(self.value)