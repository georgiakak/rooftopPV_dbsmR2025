"""
Author: Georgia Kakoulaki (eorgia.kakoulaki@ec.europa.eu or georgia.kakoulaki@gmail.come)
Version: 1.0
Date: April 2025
Description: Final step for the DBSM R2025 processing.
"""


from osgeo import ogr
import geopandas as gpd
import pandas as pd
import glob
import os
import pyogrio
import json
from datetime import date


###### Constants
data_folder = r"/path/to/input/folder"
output_folder = r"/path/to/output/folder"
dbsm_v = "dbsm-v2"#DBSM's version used 
today_data = date.today().strftime("%Y-%m-%d")

# Get country names
country_names = [
    f.split('_')[0] for f in filter(lambda f: f.endswith('.gpkg'), os.listdir(DATA_FOLDER))
]
#print(country_names)

column_data_types = {
    'source': str, 'eub-height': float, 'eub-age': float, 'eub-type': str, 'osm-building': str,
    'osm-levels': str, 'osm-roof-shape': str, 'osm-height': str, 'osm-date': str,
    'msb_height': float, 'area': float, 'unique_id': str, 'flag': str,
    'eub_json_attrib': str, 'osm_json_attrib': str, 'msb_json_attrib': str
}

gpkg_files = glob.glob(os.path.join(data_folder, '*.gpkg'))

#############################################################################################################################
def row_to_pretty_string(row, columns_to_condense, prefix_to_strip='', separator=', ', key_value_format='{key}: {value}'):
    """
    Converts specific attribute fields in a row to a json formatted string.

    Args:
        row (dict or pd.Series): The data row containing key-value pairs.
        columns_to_condense (list): List of columns to include in the output string.
        prefix_to_strip (str): A prefix to remove from column names (default: '').
        separator (str): The separator to use between key-value pairs (default: ', ').
        key_value_format (str): Format string for key-value pairs (default: '{key}: {value}').

    Returns:
        str: A formatted string with key-value pairs.

    parts = []
    for col in columns_to_condense:
        if col in row:
            key = col.replace(prefix_to_strip, '')
            value = row[col] if pd.notna(row[col]) else ''
            parts.append(key_value_format.format(key=key, value=value))
        else:
            raise ValueError(f"Column '{col}' not found in row.")
    return separator.join(parts)




def save_attributes_as_json_col(gdf, prefixes=None):
    """
    Save specific attribute fields in JSON-like string format in new columns.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame with attribute fields.
        prefixes (dict): A dictionary where keys are new column names and values are prefixes to match in column names.
                         Example: {'eub_json_attrib': 'eub-', 'osm_json_attrib': 'osm-', 'msb_json_attrib': 'msb_'}

    Returns:
        GeoDataFrame: Modified GeoDataFrame with new JSON-like attribute columns.

    
    if prefixes is None:
        prefixes = {
            'eub_json_attrib': 'eub-',
            'osm_json_attrib': 'osm-',
            'msb_json_attrib': 'msb_'
        }
    
    for new_col, prefix in prefixes.items():
        matching_columns = [col for col in gdf.columns if col.startswith(prefix)]
        if matching_columns:
            gdf[new_col] = gdf[matching_columns].apply(
                lambda row: row_to_pretty_string(row, matching_columns, prefix), axis=1
            )
        else:
            print(f"No columns found for prefix '{prefix}'. Skipping creation of '{new_col}'.")
    
    return gdf

#############################################################################################################################
for gpkg_file in gpkg_files:
    # Open the GeoPackage file
    ds = ogr.Open(gpkg_file)
    name_gpkg = gpkg_file.split('/')[-1]
    country_name = name_gpkg.split('_')[0]
    # Get the first layer (assuming there's only one)
    layer = ds.GetLayer(0)

    # Get the field names (columns) of the layer
    field_names = [field.GetName() for field in layer.schema]
    gdf = gpd.read_file(gpkg_file, engine='pyogrio')
    gdf_mod = save_attributes_as_json_col(gdf)
    gdf_mod['area'] = gdf_mod['area'].round(2)
    print('first round done')
    # Check if all expected columns exist
    for col in column_data_types:
        if col not in field_names:
            print(f"Warning: Column '{col}' is missing from {gpkg_file}")

    # Delete any extra columns
    extra_columns = [col for col in field_names if col not in column_data_types and col !='geometry']

    if extra_columns:
        print(f"Deleting extra columns from {gpkg_file}: {extra_columns}")
        #if 'msb_height' in gdf.columns:
         #   gdf.rename(columns={'msb_height':'msb-height'},inplace=True
        gdf_mod.drop(columns=extra_columns, inplace=True)
        gdf_mod.to_file(os.path.join(output_path,f'{dbsm_v}-{country_name}-{today_day}.gpkg'), driver='GPKG')

    else:
        print(f"No extra columns found in {gpkg_file}")
        #gdf.to_file(os.path.join(output_path,f'{dbsm_v}-{country_name}-{today_day}.gpkg'), driver='GPKG')
        gdf_mod.to_file(os.path.join(output_path,f'{dbsm_v}-{country_name}-{today_day}.gpkg'), driver='GPKG')

    # Close the dataset
    ds = None
