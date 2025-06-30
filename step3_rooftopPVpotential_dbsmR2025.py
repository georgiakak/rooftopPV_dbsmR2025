"""
Author: Georgia kakoulaki (georgia.kakoulaki@ec.europa.eu or georgia.kakoulaki@gmail.com)
Version: 1.0
Initial Creation: February 2024
Last Modification: April 2025

Copyright (c) 2025 European Union  
Licensed under the EUPL v1.2 or later: https://joinup.ec.europa.eu/collection/eupl/eupl-text-eup

Description:
----------
Python based tool for the claculation of the rooftop PV potential for residential and non rediential buildings using the DBSM (Digital Building Stock Model) R2025
The scriprts support:
-Definition of modules and estimation of rooftop PV potential (Mwh, kWp) assuming 25%, 40% of usuable rooftop area for PV installation
-Output as GeoPackage files per country.

Define coordinate system use : 'EPSG:3035' # all the input datasets were reprojected

Dataset DOI: https://doi.org/?
Code repository: https://code.europa.eu/?

For support, contact: ?@ec.europa.eu

"""


import os
import geopandas as gpd
import pandas as pd
import numpy as np
import pyogrio
import glob
import pycountry
import math
from datetime import date
from joblib import Parallel, delayed




# Base folder containing per-country building data 
data_folder = './data/buildings_per_country/'  
# Folder containing building footprint tiles per country
tiles_country_folder = './data/building_footprint_tiles/'  

# Versioning and data creation metadata
dbsm_version = 'dbsm-v2-'
data_creation_date = '2024-12-12'
# Current date for logging or versioning
today = date.today().strftime("%Y-%m-%d")

# Output folder for generated results
output_folder = './output/pv_results/'
os.makedirs(output_folder, exist_ok=True)

# Extract list of countries based on the filenames in the data folder
# Assumes filenames are formatted as '<country>-<dbsmversion>-R2025-*.fgb'
country_names = [
    filename.split('-')[0]
    for filename in os.listdir(data_folder)
    if filename.endswith('.fgb')
]


#############################  module characteristics ###########################
degrees_mod=20
module_inclin=degrees_mod #degrees inclination
module_base=1 #m
module_height=1.65 #m height
module_effic=0.22 #22%
module_power=module_effic*module_base*module_height*1000 #Wp Nominal module peak power, per module
module_footprint=round(module_height*math.cos(math.radians(module_inclin)),2) #in m2
power_density=round(module_power/module_footprint,2)#Wp/m2, number of Wp installed in every m2, for rooftops we go for high power density
########
floorarea_bins = [0, 250, 1000, 2000, 1e9]
bin_lbl_floor = ['0-250', '250-1000', '1000-2000', '>2000']


def pv_calc(tiles_country_path,country_name,tile,floorarea_bins,bin_lbl_floor):
    try:
        tile_path = os.path.join(tiles_country_path, country_name, tile)
        tile_gpkg = gpd.read_file(tile_path, engine='pyogrio')
        xmin, ymin, xmax, ymax = tile_gpkg.total_bounds
        bbox = (xmin, ymin, xmax, ymax)
        dbsm_tile_read = gpd.read_file(glob.glob(glob_pattern)[0], bbox=bbox, engine='pyogrio')
        print(f'building {tile} READ')

        if not dbsm_tile_read.empty:
            dbsm_tile_read['floor_area_m2'] = dbsm_tile_read['area'] * (dbsm_tile_read['height'] / 3).round()

            # when height is NaN and area <= 100
            mask = dbsm_tile_read['area'] <= 100
            mask &= pd.isna(dbsm_tile_read['height'])
            dbsm_tile_read.loc[mask, 'floor_area_m2'] = dbsm_tile_read.loc[mask, 'area'] * 1.5

            # when height is NaN and area > 100
            fallback_mask = pd.isna(dbsm_tile_read['height']) & ~mask
            dbsm_tile_read.loc[fallback_mask, 'floor_area_m2'] = dbsm_tile_read.loc[fallback_mask, 'area']

            dbsm_tile_read['floor_area_bin'] = pd.cut(dbsm_tile_read['floor_area_m2'], bins=floorarea_bins,
                                                          right=True, include_lowest=True, labels=bin_lbl_floor)
            dbsm_tile_read['floor_area_bin'] = dbsm_tile_read['floor_area_bin'].astype(str)



            dbsm_tile_read['num_floors'] = round(dbsm_tile_read['height'] / 3)
            dbsm_tile_read['volume_m3'] = dbsm_tile_read['height'] * dbsm_tile_read['area']
            dbsm_tile_read['wall_area_m2'] = dbsm_tile_read['height'] * np.sqrt(dbsm_tile_read['area'])

            result = dbsm_tile_read.apply(calculate_orientation, axis=1)
            dbsm_tile_read[['orientation', 'len_x', 'len_y']] = result
            dbsm_tile_read['len_x_height_m2'] = dbsm_tile_read['len_x'] * dbsm_tile_read['height']
            dbsm_tile_read['len_y_height_m2'] = dbsm_tile_read['len_y'] * dbsm_tile_read['height']

            dbsm_tile_read['nuts3_id'] = dbsm_tile_read['unique_id'].apply(lambda x: x.split('_')[0])
            ###################
            mean_e_y = round(dbsm_tile_read.e_y.mean(), 3)
            print(f'{tile[:-5]} mean E_Y: {mean_e_y}')

            if any(isinstance(e, str) or (isinstance(e, (int, float)) and e < 0) for e in dbsm_tile_read.e_y):
                print("Values contain text or -9999")
                non_numrows = dbsm_tile_read[
                    ~dbsm_tile_read['error'].isnull()]  # check the column error and find which one have an error as text
                # print(non_numrows)
                dbsm_tile_read.loc[non_numrows.index, ['e_y']] = mean_e_y  # replace the zeros with the averge e_y values
                print('Some E_Y values, should be replaced with the avg E_Y')
            else:
                print('All buiildings have E_Y values')

            #####################-------ESTIMATIOn PV OUTPUT --------------------------###################################
            dbsm_tile_read['system_yield'] = round(dbsm_tile_read['e_y'] * (power_density / 1000),
                                                   2)  # kWh/m2, how much i can produce with each system

            dbsm_tile_read['pvelect_25perArea_MWh'] = round(
                dbsm_tile_read['area'] * 0.25 * (dbsm_tile_read['system_yield'] / 1000), 2)  # MWh/yr (here 25% of the building footprint area)
            
            dbsm_tile_read['pvelect_40perArea_MWh'] = round(dbsm_tile_read['area'] * 0.4 * (dbsm_tile_read['system_yield'] / 1000),2)  # MWh/yr (here  40% of the building footprint area)
            

            #########-------------------##################-------TECHNICAL INSTALLED CAPACITY-----------------------------###############
            dbsm_tile_read['inst_cap_25perc_kwp'] = round(dbsm_tile_read['area'] * 0.25 * (power_density / 1000),
                                                          2)  # unit: kwp  that i install
            dbsm_tile_read['inst_cap_40perc_kwp'] = round(dbsm_tile_read['area'] * 0.4 * (power_density / 1000),
                                                          2)  # unit: kwp  that i install
            
            #########-------------------#############---------------------------------------########------------#####################-------------#################
            print(f'{tile[:-5]} PV output estimated')
        else:
            print(f'{tile} is EMPTY')
            pass
    except Exception as e:
            print(f'Error {e} - Working on {tile}')
    return dbsm_tile_read
	
	

def joblib_loop(files_tiles,country_name):
    results = Parallel(n_jobs=20, verbose=5)(delayed(pv_calc)(tiles_country_path,country_name,tile,floorarea_bins,bin_lbl_floor) for tile in files_tiles)
    return results

def check_empty_geometries(gdf:gpd.GeoDataFrame)-> gpd.GeoDataFrame:
    before_empty = gdf.shape[0]
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.notna()]
    after_empty = gdf.shape[0]
    removed_geometries = before_empty - after_empty
    if removed_geometries > 0:
        print(f" Found and removed {removed_geometries} EMPTY geometries")
    return gdf

def check_valid_geometries(gdf:gpd.GeoDataFrame)-> gpd.GeoDataFrame:
    valid = gdf[gdf['geometry'].is_valid]
    invalid = gdf[~gdf['geometry'].is_valid]
    if invalid.shape[0] > 0:
        print(f" Trying to fix { invalid.shape[0]} INVALID geometries")
        invalid = invalid.copy()
        invalid['geometry'] = invalid['geometry'].buffer(0)
        stil =  invalid[~invalid['geometry'].is_valid]
        if stil.shape[0] > 0:
            print(" Can not fix geometries. Removed!")
            gdf = valid
        else:
            print(' Fixed')
            gdf = pd.concat([valid, invalid], ignore_index=True)
    gdf = check_empty_geometries(gdf)
    return gdf

def remove_duplicate_geom(gdf:gpd.GeoDataFrame)-> gpd.GeoDataFrame:
    n_geoms_before = gdf.shape[0]
    gdf.drop_duplicates(subset='geometry', inplace=True)
    n_geoms_after = gdf.shape[0]
    if n_geoms_after < n_geoms_before:
        print(f" Found and removed {n_geoms_before - n_geoms_after} DUPLICATE geometries")
    return gdf

def remove_duplicate_geom_in_same_country(gdf:gpd.GeoDataFrame)-> gpd.GeoDataFrame:
    n_geoms_before = gdf.shape[0]
    gdf.drop_duplicates(subset=['geometry', 'NUTS_ID'], inplace=True)
    n_geoms_after = gdf.shape[0]
    if n_geoms_after < n_geoms_before:
        print(f" Found and removed {n_geoms_before - n_geoms_after} DUPLICATE geometries")
    return gdf

def replace_nan_string(gdf:gpd.GeoDataFrame)-> gpd.GeoDataFrame:
    gdf.replace('nan', None, inplace=True)
    return gdf


def build_nuts3_use_statistics(gdf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute building and PV statistics grouped by NUTS3 region, use type, and floor area bin.

    Parameters:
    ----------
    gdf : pd.DataFrame

    Returns:
    -------
    pd.DataFrame
        Aggregated statistics.
    """
    # Optional: extract NUTS ID from unique_id if needed
    if 'unique_id' in gdf.columns and 'nuts_id' not in gdf.columns:
        gdf['nuts_id'] = gdf['unique_id'].str.split('_').str[0]

    # Flat dictionary with custom column names using pd.NamedAgg
    agg_dict = {
        'TotalFootptinArea_m2': pd.NamedAgg(column='area', aggfunc='sum'),
        'max_Footprint_m2': pd.NamedAgg(column='area', aggfunc='max'),
        'min_Footprint_m': pd.NamedAgg(column='area', aggfunc='min'),
        'mean_m': pd.NamedAgg(column='area', aggfunc='mean'),
        'median_m2': pd.NamedAgg(column='area', aggfunc='median'),
        'std_Footprint_m': pd.NamedAgg(column='area', aggfunc='std'),
        'var_Footprint_m': pd.NamedAgg(column='area', aggfunc='var'),
        'cv_Footprint_m': pd.NamedAgg(column='area', aggfunc=lambda x: x.std() / x.mean()),

        'mean_floor_num': pd.NamedAgg(column='num_floors', aggfunc='mean'),
        'min_floor_num': pd.NamedAgg(column='num_floors', aggfunc='min'),
        'max_floor_num': pd.NamedAgg(column='num_floors', aggfunc='max'),
        'std_floor_area_m2': pd.NamedAgg(column='floor_area_m2', aggfunc='std'),
        'IQR_Footprint_m': pd.NamedAgg(column='area', aggfunc=lambda x: x.quantile(0.75) - x.quantile(0.25)),

        'count_use_0': pd.NamedAgg(column='use', aggfunc=lambda x: (x == 0).sum()),
        'count_use_1': pd.NamedAgg(column='use', aggfunc=lambda x: (x == 1).sum()),
        'count_use_2': pd.NamedAgg(column='use', aggfunc=lambda x: (x == 2).sum()),

        'TotalFloorArea_m2': pd.NamedAgg(column='floor_area_m2', aggfunc='sum'),
        'max_floorarea_m2': pd.NamedAgg(column='floor_area_m2', aggfunc='max'),
        'min_floorarea_m2': pd.NamedAgg(column='floor_area_m2', aggfunc='min'),
        'mean_floor_area': pd.NamedAgg(column='floor_area_m2', aggfunc='mean'),
        'median_floor_area': pd.NamedAgg(column='floor_area_m2', aggfunc='median'),

        'Total_volume_m3': pd.NamedAgg(column='volume_m3', aggfunc='sum'),
        'max_volume_m3': pd.NamedAgg(column='volume_m3', aggfunc='max'),
        'min_volume_m3': pd.NamedAgg(column='volume_m3', aggfunc='min'),
        'mean_volume_m3': pd.NamedAgg(column='volume_m3', aggfunc='mean'),
        'median_volume_m3': pd.NamedAgg(column='volume_m3', aggfunc='median'),
        'std_volume_m3': pd.NamedAgg(column='volume_m3', aggfunc='std'),

        'max_e_y_kwh_kwp': pd.NamedAgg(column='e_y', aggfunc='max'),
        'min_e_y_kwh_kwp': pd.NamedAgg(column='e_y', aggfunc='min'),
        'mean_e_y_kwh_kwp': pd.NamedAgg(column='e_y', aggfunc='mean'),
        'median_e_y_kwh_kwp': pd.NamedAgg(column='e_y', aggfunc='median'),
        'std_e_y_kwh_kwp': pd.NamedAgg(column='e_y', aggfunc='std'),

        'Total_building': pd.NamedAgg(column='source', aggfunc='count')
    }

    # Dynamically add PV and capacity stats
    for perc in [25, 40]:
        agg_dict.update({
            f'TotalPVgener_{perc}_MWh': pd.NamedAgg(column=f'pvelect_{perc}perArea_MWh', aggfunc='sum'),
            f'MeanPVgener_{perc}_MWh': pd.NamedAgg(column=f'pvelect_{perc}perArea_MWh', aggfunc='mean'),
            f'MaxPVgener_{perc}_MWh': pd.NamedAgg(column=f'pvelect_{perc}perArea_MWh', aggfunc='max'),
            f'MinPVgener_{perc}_MWh': pd.NamedAgg(column=f'pvelect_{perc}perArea_MWh', aggfunc='min'),
            f'STDPVgener_{perc}_MWh': pd.NamedAgg(column=f'pvelect_{perc}perArea_MWh', aggfunc='std'),

            f'Total_kWp_{perc}per': pd.NamedAgg(column=f'inst_cap_{perc}perc_kwp', aggfunc='sum'),
            f'mean_kWp_{perc}per': pd.NamedAgg(column=f'inst_cap_{perc}perc_kwp', aggfunc='mean'),
            f'max_kWp_{perc}per': pd.NamedAgg(column=f'inst_cap_{perc}perc_kwp', aggfunc='max'),
            f'min_kWp_{perc}per': pd.NamedAgg(column=f'inst_cap_{perc}perc_kwp', aggfunc='min'),
            f'STDPVgener_{perc}_kWp': pd.NamedAgg(column=f'inst_cap_{perc}perc_kwp', aggfunc='std'),

        })

    # Now group and aggregate (with multiple groupby keys)
    stats = gdf.groupby(['nuts3_id', 'use', 'floor_area_bin'], observed=False).agg(**agg_dict).round(3)

    # Reset index for clean DataFrame
    stats.reset_index(inplace=True)
    return stats


	
def main()
	for country in country_names:
			pv_stats = []
			dummy_gpd = []

			country_name = country
			glob_pattern = os.path.join(data_fld, f"{country}-v2-ghsl-e_y-*.fgb")
			iso3id = pycountry.countries.get(name=country_name).alpha_3  # iso3 code country
			files_tiles = os.listdir(os.path.join(tiles_country_path, country))

			results = joblib_loop(files_tiles, country_name)
			for result in results:
				dummy_gpd.append(result)
			####################### save stats per country ##############
			country_data = pd.concat(dummy_gpd)
			country_all_buildings = check_valid_geometries(country_data)
			country_all_buildings.drop_duplicates(subset=['geometry','unique_id'], inplace=True)

			stats = build_nuts3_use_statistics(country_all_buildings)
			output_csv = os.path.join(output_fld,f'{country_name}_nuts3_stats{today}.csv')
			stats.to_csv(output_csv, index=False)
			pv_stats.append(stats)

			country_all_buildings.to_file(os.path.join(output_fld, f'dbsm-{country_name}-pvpotential-R2025.gpkg'),driver='GPKG')
    if all_pv_stas:
		pv_stats_all=pd.concat(pv_stats)
		pv_stats_all.to_csv(os.path.join(output_fld, f'dbsm-alleu27-nuts3-use-floorarea-pvpotential-R2025.csv'))

		
if __name__ == "__main__":
    main()
