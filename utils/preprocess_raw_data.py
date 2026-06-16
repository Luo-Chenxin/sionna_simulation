from pathlib import Path
import pandas as pd
from config import PRE_TRANSMITTER_FILE_ENCODE, FILE_ENCODE, LocalCRS
from typing import Tuple
from pyproj import Transformer

def preprocess_massy_antenna_data(
    input_file: Path,
    output_dir: Path,
) -> None:
    """
    Clean antenna data and split it into files by frequency
    
    The returned table preserves all main rows and includes columns:
    [ID, Numéro de Station, Numéro d'antenne, Latitude, Longitude, height, Azimut, frequency]
    """

    # Columns to change to numbers
    num_cols = ["Latitude", "Longitude", "height", "Hauteur en m"]

    # Target columns for the output Sionna files
    output_cols = [
        "Numéro de Station",
        "Numéro d'antenne",
        "Latitude",
        "Longitude",
        "height",
        "Azimut",
        "frequency",
    ]

    # ==========================================
    # 1. INITIALIZATION
    # ==========================================
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Cannot find file {input_file}")
        return

    # Create output folder if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Start processing: {input_file}")

    # ==========================================
    # 2. LOAD DATA & CLEAN HEADERS
    # ==========================================
    # Read CSV and drop 'Unnamed' columns
    df = pd.read_csv(
        input_file, encoding=PRE_TRANSMITTER_FILE_ENCODE, 
        usecols=lambda x: "Unnamed" not in str(x)
    )

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # ==========================================
    # 3. PROCESS COORDINATES & HEIGHTS
    # ==========================================
    # Split 'Lat-Lon' column into 'Latitude' and 'Longitude'
    df[["Latitude", "Longitude"]] = df["Lat-Lon"].str.split(expand=True)

    # Convert columns to numbers
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # If 'height' is missing, use value from 'Hauteur en m'
    df["height"] = df["height"].fillna(df["Hauteur en m"])

    # ==========================================
    # 4. CLEAN FREQUENCY
    # ==========================================
    # Convert to string and lowercase
    df["frequency"] = df["frequency"].astype(str)
    freq_clean = df["frequency"].str.strip().str.lower()

    # Get only the numbers from the end of the string
    df["frequency_num_part"] = (
        freq_clean.str.extract(r"(\d+)\s*$", expand=False)
    ).fillna("unknown")

    # ==========================================
    # 5. GROUP BY FREQUENCY AND EXPORT
    # ==========================================
    # Keep only columns that exist in the dataframe
    output_cols = [col for col in output_cols if col in df.columns]

    # Group data by the frequency number
    for freq_num, group in df.groupby("frequency_num_part"):

        # Make a copy of the group data
        result_df = group[output_cols].copy()

        # Add 'ID' column at the beginning (starts from 1)
        result_df.insert(0, "ID", range(1, len(result_df) + 1))

        # Save to a new CSV file
        file_name = f"{freq_num}_mhz.csv"
        path_out = output_dir / file_name

        result_df.to_csv(path_out, index=False, encoding=FILE_ENCODE)
        print(f"Saved: {path_out} ({len(result_df)} rows)")

    print("\nAll done!")


def preprocess_antenna_data_by_frequency_and_postal(
    main_input_file: Path,
    secondary_input_file: Path,
    frequency: int,
    postal_code: str,
    output_dir: Path,
) -> None:
    """
    Filter antenna records by Système frequency and postal code, then join coordinates.

    Notice: frequency unit is `mhz` both in the parameter and input files

    Main table is deduplicated on `Numéro de support`.
    Secondary table renames `Numéro du support` to `Numéro de support` and left-joins on that key.
    The returned table preserves all main rows and includes columns:
    [ID, Numéro de support, Latitude, Longitude, height, Azimut, frequency, Code postal]

    Deduplication rule: 
    For rows with identical values in ['Numéro de support', 'Latitude', 'Longitude', 
    'frequency', 'Code postal'], only the first occurrence is kept.
    """

    # Load main antenna table
    main_df = pd.read_csv(
        main_input_file,
        encoding=PRE_TRANSMITTER_FILE_ENCODE,
        sep=';',
        usecols=lambda x: 'Unnamed' not in str(x),
    )
    main_df.columns = main_df.columns.str.strip()

    # Load secondary site table
    secondary_df = pd.read_csv(
        secondary_input_file,
        encoding=PRE_TRANSMITTER_FILE_ENCODE,
        sep=';',
        usecols=lambda x: 'Unnamed' not in str(x),
    )
    secondary_df.columns = secondary_df.columns.str.strip()

    # Rename main table fields
    main_df = main_df.rename(columns={'Hauteur / sol': 'height', 'Système': 'frequency'})

    # Rename secondary key and deduplicate coordinates in secondary table
    secondary_df = secondary_df.rename(columns={'Numéro du support': 'Numéro de support'})

    # Left join secondary coordinates and postal code onto main table
    merged_df = pd.merge(
        main_df,
        secondary_df[['Numéro de support', 'Longitude', 'Latitude', 'Code postal']],
        on='Numéro de support',
        how='left',
    )

    # Filter final result by frequency and postal code
    freq_clean = merged_df['frequency'].astype(str).str.strip().str.lower()
    freq_extracted = freq_clean.str.extract(r"(\d+)\s*$", expand=False).fillna("unknown")
    merged_df = merged_df[freq_extracted == str(frequency)].copy()
    merged_df = merged_df[merged_df['Code postal'].astype(str).str.match(postal_code)].copy()

    merged_df = merged_df.drop_duplicates(
        subset=['Numéro de support', 'Latitude', 'Longitude', 'frequency', 'Code postal'],
        keep='first'
    )

    # Keep only the requested output columns and preserve order
    output_cols = [
        'Numéro de support', 
        'Latitude', 
        'Longitude', 
        'height', 
        'Azimut', 
        'frequency',
        'Code postal',
    ]
    result_df = merged_df[[col for col in output_cols if col in merged_df.columns]].copy()
    result_df.insert(0, 'ID', range(1, len(result_df) + 1))

    # Save to a new CSV file
    file_name = f"{frequency}_mhz.csv"
    path_out = output_dir / file_name

    result_df.to_csv(path_out, index=False, encoding=FILE_ENCODE)
    print(f"Saved: {path_out} ({len(result_df)} rows)")

def process_csv_with_buffer(tx_csv_path: Path, buffer_meters: float) -> Tuple[float, float, float, float]:
    """
    Read a CSV file with Latitude and Longitude columns, compute the bounding box,
    convert to France Lambert-93 projection, expand by buffer_meters, and convert back.
    
    Args:
        tx_csv_path (Path): Path to the CSV file containing Latitude and Longitude columns
        buffer_meters (float): Buffer distance in meters to expand the bounding box (default: 500)
    
    Returns:
        Tuple[float, float, float, float]: (LAT_MAX, LAT_MIN, LON_MIN, LON_MAX) 
        in the original WGS84 coordinate system
    
    Raises:
        ValueError: If required columns are missing or data is empty
    """
    # Read the CSV file
    df = pd.read_csv(tx_csv_path)
    
    # Check if required columns exist
    required_cols = ['Latitude', 'Longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check if data is empty
    if df.empty:
        raise ValueError("CSV file contains no data")
    
    # Remove any rows with NaN values in Latitude or Longitude
    df_clean = df[['Latitude', 'Longitude']].dropna()
    
    if df_clean.empty:
        raise ValueError("No valid data after removing NaN values")
    
    # Calculate bounding box in WGS84 (Latitude/Longitude)
    lat_max = df_clean['Latitude'].max()
    lat_min = df_clean['Latitude'].min()
    lon_min = df_clean['Longitude'].min()
    lon_max = df_clean['Longitude'].max()
    
    print(f"Original bounding box (WGS84):")
    print(f"  LAT_MAX: {lat_max:.6f}, LAT_MIN: {lat_min:.6f}")
    print(f"  LON_MIN: {lon_min:.6f}, LON_MAX: {lon_max:.6f}")
    
    # Create transformer for WGS84 <-> Lambert-93
    transformer_to_lambert = Transformer.from_crs(
        LocalCRS.OSM_STORAGE.value, 
        LocalCRS.FRANCE_LAMBERT93.value
    )
    transformer_to_wgs84 = Transformer.from_crs(
        LocalCRS.FRANCE_LAMBERT93.value,
        LocalCRS.OSM_STORAGE.value
    )
    
    # Convert the four corners to Lambert-93
    # (Longitude, Latitude) -> (X, Y) in Lambert-93
    corners_lambert = []
    corners_wgs84 = [
        (lon_min, lat_max),  # NW corner
        (lon_max, lat_max),  # NE corner
        (lon_min, lat_min),  # SW corner
        (lon_max, lat_min)   # SE corner
    ]
    
    for lon, lat in corners_wgs84:
        x, y = transformer_to_lambert.transform(lon, lat)
        corners_lambert.append((x, y))
    
    # Calculate bounding box in Lambert-93
    x_coords = [x for x, _ in corners_lambert]
    y_coords = [y for _, y in corners_lambert]
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    print(f"\nBounding box in Lambert-93 (meters):")
    print(f"  X_MIN: {x_min:.2f}, X_MAX: {x_max:.2f}")
    print(f"  Y_MIN: {y_min:.2f}, Y_MAX: {y_max:.2f}")
    
    # Apply buffer in Lambert-93 (expand by buffer_meters in all directions)
    x_min_buffered = x_min - buffer_meters
    x_max_buffered = x_max + buffer_meters
    y_min_buffered = y_min - buffer_meters
    y_max_buffered = y_max + buffer_meters
    
    print(f"\nBounding box with {buffer_meters}m buffer in Lambert-93:")
    print(f"  X_MIN: {x_min_buffered:.2f}, X_MAX: {x_max_buffered:.2f}")
    print(f"  Y_MIN: {y_min_buffered:.2f}, Y_MAX: {y_max_buffered:.2f}")
    
    # Convert back to WGS84 (Latitude/Longitude)
    # Convert the four corners of the buffered bounding box back to WGS84
    corners_buffered_wgs84 = []
    for x, y in [(x_min_buffered, y_min_buffered),  # SW
                 (x_max_buffered, y_min_buffered),  # SE
                 (x_min_buffered, y_max_buffered),  # NW
                 (x_max_buffered, y_max_buffered)]: # NE
        lon, lat = transformer_to_wgs84.transform(x, y)
        corners_buffered_wgs84.append((lon, lat))
    
    # Calculate the final bounding box in WGS84
    lons_buffered = [lon for lon, _ in corners_buffered_wgs84]
    lats_buffered = [lat for _, lat in corners_buffered_wgs84]
    
    lat_max_buffered = max(lats_buffered)
    lat_min_buffered = min(lats_buffered)
    lon_min_buffered = min(lons_buffered)
    lon_max_buffered = max(lons_buffered)
    
    print(f"\nFinal bounding box with buffer (WGS84):")
    print(f"  LAT_MAX: {lat_max_buffered:.6f}, LAT_MIN: {lat_min_buffered:.6f}")
    print(f"  LON_MIN: {lon_min_buffered:.6f}, LON_MAX: {lon_max_buffered:.6f}")
    
    return lat_max_buffered, lat_min_buffered, lon_min_buffered, lon_max_buffered