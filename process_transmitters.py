from pathlib import Path
import pandas as pd

# file path
base_dir = Path('data')
output_dir = base_dir / 'transimitters'
file_antenna = base_dir / 'bs_antenna_verify.csv'
file_bands = base_dir / 'Antennes_Emetteurs_Bandes_Cartoradio.csv'

# make output directory
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Folder created: {output_dir}")

# read and preprocess
df_antenna = pd.read_csv(file_antenna, encoding='cp1252', sep=';', usecols=lambda x: 'Unnamed' not in str(x))
df_bands = pd.read_csv(file_bands, encoding='cp1252', sep=';', usecols=lambda x: 'Unnamed' not in str(x))
df_antenna.columns = df_antenna.columns.str.strip()
df_bands.columns = df_bands.columns.str.strip()

# filter out 'non disponible' items
if 'Date de mise en service' in df_bands.columns:
    df_bands = df_bands[df_bands['Date de mise en service'].str.strip() != 'non disponible']

# drop duplicates in df_antenna
df_antenna = df_antenna.drop_duplicates(subset=['Numéro de Station', 'Numéro d\'antenne'])

# drop duplicates in df_bands
df_bands = df_bands.drop_duplicates(subset=['Numéro de Station', 'Numéro d\'antenne', 'Début', 'Fin', 'Unité'])

# split coordinations
df_antenna[['Latitude', 'Longitude']] = df_antenna['Lat-Lon'].str.split(',', expand=True)

# format number
df_antenna['Latitude'] = pd.to_numeric(df_antenna['Latitude'])
df_antenna['Longitude'] = pd.to_numeric(df_antenna['Longitude'])
df_antenna['height'] = df_antenna['height'].str.replace(',', '.', regex=False)
df_antenna['height'] = pd.to_numeric(df_antenna['height'])

# fill NaN `height` field using the corresponding 'Hauteur en m' field
df_antenna['height'] = df_antenna['height'].fillna(df_antenna['Hauteur en m'])

# unify Unité to lowercase
df_bands['Unité'] = df_bands['Unité'].str.lower()

# left join
merged_df = pd.merge(
    df_antenna, 
    df_bands, 
    on=['Numéro de Station', 'Numéro d\'antenne'], 
    how='left',
    suffixes=('_antenna', '_bands'),
)
print(f"Merging complete, total records: {len(merged_df)}")

# caculate center frequence
merged_df['Frequence'] = (merged_df['Début'] + merged_df['Fin']) / 2

# classification logic
def classify_band(row):
    freq = row['Frequence']
    unite = row['Unité']
    
    if pd.isna(freq) or unite != 'mhz':
        return 'other'
    
    # 700 MHz：703–748 MHz or 758–803 MHz 之间
    if (703 <= freq <= 748) or (758 <= freq <= 803):
        return '700'
    # 800 MHz：832–862 MHz or 791–821 MHz 之间
    elif (832 <= freq <= 862) or (791 <= freq <= 821):
        return '800'
    # 1800 MHz 1710–1785 MHz or 1805–1880 MHz 之间
    elif (1710 <= freq <= 1785) or (1805 <= freq <= 1880):
        return '1800'
    # 2100 MHz 1920–1980 MHz or 2110–2170 MHz 之间
    elif (1920 <= freq <= 1980) or (2110 <= freq <= 2170):
        return '2100'
    # 3500MHz 3300–3800 MHz
    elif 3300 <= freq <= 3800:
        return '3500'
    else:
        return 'other'

merged_df['Band_Group'] = merged_df.apply(classify_band, axis=1)

grouped = merged_df.groupby('Band_Group')
for band_name, group in grouped:

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
    
    result_df = group[final_columns].copy()
    result_df.rename(columns={'Azimut_antenna': 'Azimut'}, inplace=True)

    # Generate auto-incrementing ID, which will be used in sionna
    result_df['ID'] = range(1, len(result_df) + 1)

    # Rearrange the final column order
    final_columns_order = [
        'ID', 
        'Numéro de Station', 
        'Numéro d\'antenne', 
        'Latitude', 
        'Longitude', 
        'height', 
        'Azimut', 
        'Type d\'antenne', 
        'Début', 
        'Fin', 
        'Unité'
    ]
    result_df = result_df[final_columns_order]

    # get filename based on group name
    filename = "other.csv" if band_name == 'other' else f"{band_name}_mhz.csv"
    output_path = output_dir / filename
    
    # save as CSV
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"The file has been saved: {output_path} (containing {len(result_df)} data entries).")

print("\nAll files have been processed.")