from pathlib import Path
import pandas as pd
from config import PRE_TRANSMITTER_FILE_ENCODE, FILE_ENCODE

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

    # Deduplicate main table on Numéro de support
    main_df = main_df.drop_duplicates(subset=['Numéro de support']).copy()

    # Rename main table fields
    main_df = main_df.rename(columns={'Hauteur / sol': 'height', 'Système': 'frequency'})

    # Rename secondary key and deduplicate coordinates in secondary table
    secondary_df = secondary_df.rename(columns={'Numéro du support': 'Numéro de support'})
    secondary_df = secondary_df.drop_duplicates(subset=['Numéro de support']).copy()

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