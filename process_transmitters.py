import os
import pandas as pd

# Define file path
base_dir = 'data'
output_dir = os.path.join(base_dir, 'transimitters')
file_antenna = os.path.join(base_dir, 'bs_antenna_verify.csv')
file_bands = os.path.join(base_dir, 'Antennes_Emetteurs_Bandes_Cartoradio.csv')

# Create the output folder (if it does not exist).
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder created: {output_dir}")

# Read a CSV file (encoded as cp1252).
df_antenna = pd.read_csv(file_antenna, encoding='cp1252', sep=';', usecols=lambda x: 'Unnamed' not in str(x))
df_bands = pd.read_csv(file_bands, encoding='cp1252', sep=';', usecols=lambda x: 'Unnamed' not in str(x))

df_antenna.columns = df_antenna.columns.str.strip()
df_bands.columns = df_bands.columns.str.strip()

# Preprocess bs_antenna_verify.csv: Deduplication
df_antenna = df_antenna.drop_duplicates()

# Split the "Lat-Lon" column into "Latitude" and "Longitude"
df_antenna[['Latitude', 'Longitude']] = df_antenna['Lat-Lon'].str.split(',', expand=True)
df_antenna['Latitude'] = pd.to_numeric(df_antenna['Latitude'])
df_antenna['Longitude'] = pd.to_numeric(df_antenna['Longitude'])

# Convert 'height' to a number
df_antenna['height'] = df_antenna['height'].str.replace(',', '.', regex=False)
df_antenna['height'] = pd.to_numeric(df_antenna['height'])

# Change ‘Unité’ to lowercase
df_bands['Unité'] = df_bands['Unité'].str.lower()

# Data Merging
# Perform an inner join based on "Numéro de Station" and "Numéro d'antenne"
# An inner join will automatically retain rows that match in both tables.
merged_df = pd.merge(
    df_antenna, 
    df_bands, 
    on=['Numéro de Station', 'Numéro d\'antenne'], 
    how='inner',
    suffixes=('_antenna', '_bands'),
)
print(f"Merging complete, {len(merged_df)} matching records")

# 5. Generate final files by group, according to the values ​​in the three columns ['Début', 'Fin', 'Unité'].
grouped = merged_df.groupby(['Début', 'Fin', 'Unité'])

for name, group in grouped:
    debut, fin, unite = name

    # Extract and organize the columns that need to be saved.
    final_columns = [
        'Numéro de Station', 
        'Numéro d\'antenne',
        'Latitude', 
        'Longitude', 
        'height', 
        'Azimut_antenna', 
        'Type d\'antenne',
        'Début',
        'Fin',
        'Unité'
    ]
    
    # Extract a copy of data from a specified column
    result_df = group[final_columns].copy()

    result_df.rename(columns={
        'Azimut_antenna': 'Azimut'
    }, inplace=True)
    
    # Calculate the frequency average
    result_df['Frequence'] = (debut + fin) / 2

    # Set ID
    result_df['ID'] = range(1, len(result_df) + 1)

    # Define the final column order you want
    final_columns = [
        'ID',
        'Numéro de Station', 
        'Numéro d\'antenne',
        'Latitude', 
        'Longitude', 
        'height', 
        'Azimut', 
        'Type d\'antenne',
        'Frequence',
        'Début',
        'Fin',
        'Unité'
    ]
    result_df = result_df[final_columns]

    # Build filename [Début]_[Fin]_[Unité].csv
    filename = f"{debut}_{fin}_{unite}.csv"
    output_path = os.path.join(output_dir, filename)
    
    # Save as a CSV file
    # index=False indicates that row indexes are not saved; 
    # UTF-8 SIG encoding ensures that Excel will not display garbled characters.
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"The file has been saved: {output_path} (containing {len(result_df)} data entries).")

print("\nAll files have been processed.")