"""

Author: Georgia kakoulaki (georgia.kakoulaki@ec.europa.eu or georgia.kakoulaki@gmail.com)
Version: 1.0
Initial Creation: July 2024
Last Modification: April 2025

Copyright (c) 2025 European Union  
Licensed under the EUPL v1.2 or later: https://joinup.ec.europa.eu/collection/eupl/eupl-text-eup

Description:
Code to detect overlapping buildings in DBSM v2, assign unique IDs, and attach NUTS IDs to all polygons.
This step is tun after step)  (main conflation code)
As main inout here is the .gpkgs created in step0.
More details can be found in https://publications.jrc.ec.europa.eu/repository/handle/JRC142133

"""

import os
import json
import warnings
import logging
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from tqdm import tqdm

# Configure Logging
FORMAT = "[%(funcName)s:%(lineno)s - %(asctime)s] %(message)s"
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Paths (use environment variables or config for flexibility)

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/path/to/output/")
water_gdf = GHS-LAND_DIR #here the user needs the Global Human Settlement Layer-LAND raster layer vectorized https://human-settlement.emergency.copernicus.eu/ghs_land2022.php
clc_forest_gdf=CLC_GPKG_DIR #here the user need the Corine land cover layer from Copernicus https://land.copernicus.eu/en/products/corine-land-cover

# Column Data Types
COLUMN_DATA_TYPES = {
    'source': str,
    'eub-height': float,
    'eub-age': float,
    'eub-type': str,
    'osm-building': str,
    'osm-levels': str,
    'osm-roof-shape': str,
    'osm-height': str,
    'osm-date': str,
    'msb_height': float,
    'area': float,
    'unique_id': str,
    'flag': str,
    'eub_json_attrib': str,
    'osm_json_attrib': str,
    'msb_json_attrib': str,
}



def setup_logger():
"""Set up a logger for each process"""
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def row_to_json(row, columns_to_condense):
    """
    Convert specific attribute fields in JSON format.

    Args:
        row (pd.Series): A row from the GeoDataFrame.
        columns_to_condense (list): List of columns to include in the JSON.

    Returns:
        str: JSON string of selected columns.
    """
    row_dict = {col: (row[col] if pd.notna(row[col]) else " ") for col in columns_to_condense}
    return json.dumps(row_dict)

def save_attributes_as_json_col(gdf):
    """
    Add JSON attributes for EUB, OSM, and MSB columns in a GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: Updated GeoDataFrame with JSON attribute columns.
    """
    eub_columns = [col for col in gdf.columns if col.startswith('eub-')]
    osm_columns = [col for col in gdf.columns if col.startswith('osm-')]
    msb_columns = [col for col in gdf.columns if col.startswith('msb-')]

    if eub_columns:
        gdf['eub_json_attrib'] = gdf.apply(lambda row: row_to_json(row, eub_columns), axis=1)
    if osm_columns:
        gdf['osm_json_attrib'] = gdf.apply(lambda row: row_to_json(row, osm_columns), axis=1)
    if msb_columns:
        gdf['msb_json_attrib'] = gdf.apply(lambda row: row_to_json(row, msb_columns), axis=1)

    return gdf


def calculate_and_remove_overlaps(df, threshold=0.95):
    """
    Identifies and removes overlapping polygons from a GeoDataFrame.
    """
    # Perform union overlay to identify overlaps
    u = gpd.overlay(df1=df, df2=df, how="union", keep_geom_type=True, make_valid=True)
    u = u.loc[u.id_1 != u.id_2]  # Drop self-unions
    u["union_area"] = u.area  # Calculate area after union (intersection area)
    u["overlap_ratio"] = u["union_area"] / u["area_1"]  # Calculate overlap ratio

    # Identify polygons to drop
    to_drop = u.loc[u["overlap_ratio"] >= threshold]["id_1"]
    cleaned_df = df.loc[~df.id.isin(to_drop)]  # Remove overlapping polygons

    return cleaned_df

def assign_nuts_id_and_generate_stats(cleaned_chunk, nuts_gpkg, idx):
    """
    Assign NUTS ID to buildings and generate statistics.

    Parameters:
    tile_to_read (GeoDataFrame): Input buildings data
    nuts_gpkg (GeoDataFrame): NUTS boundaries data
    country_name (str): Country name
    dbsm_v (str): DBSM version
    output_path (str): Output path for the statistics CSV file
    today_day (str): Today's date
    """
    # Perform a spatial join to find the buildings that intersect with the NUTS boundaries
    overlapping_buildings = gpd.sjoin(cleaned_chunk, nuts_gpkg, how='inner', predicate='intersects')

    # Find the buildings that do not overlap with any NUTS boundary
    non_overlapping_buildings = cleaned_chunk[~cleaned_chunk.index.isin(overlapping_buildings.index)]

    if not non_overlapping_buildings.empty:
        # Perform a nearest neighbor search to find the closest NUTS boundary to each non-overlapping building
        nearest_nuts = gpd.sjoin_nearest(non_overlapping_buildings, nuts_gpkg, how='left', distance_col='distance')
        # Assign the NUTS ID to each non-overlapping building
        df_with_nuts_id = gpd.GeoDataFrame(pd.concat([overlapping_buildings, nearest_nuts], ignore_index=True))
        df_with_nuts_id['nuts_id'] = df_with_nuts_id['NUTS_ID']
        print('there were some buildings with no NUTS, should be fixed now')
    else:
        df_with_nuts_id = overlapping_buildings.copy()
        df_with_nuts_id = df_with_nuts_id.rename(columns={'NUTS_ID': 'nuts_id'})
        print(f'All buildings have NUTS in chunk {idx}')
    # Clean up the data
    df_with_nuts_id.drop(columns=['index_right', 'area_nuts'], inplace=True, errors='ignore')
    df_with_nuts_id['nuts_id'] = df_with_nuts_id['nuts_id'].fillna('unknown')
    df_with_nuts_id.drop_duplicates(subset='geometry', inplace=True)
   
    return df_with_nuts_id

def assign_gisco_id(chunk_with_nuts, filter_grid,idx):
    """
    Assign Gisco ID to buildings.
    Parameters:
    overlapping_buildings (GeoDataFrame): Buildings data
    grid_file_10Km (GeoDataFrame): Grid file data

    Returns:
    df_with_giscoid (GeoDataFrame): Buildings data with Gisco ID
    """

    # Perform a spatial join to find the buildings that intersect with the grid file
    gisco_buildings = gpd.sjoin(chunk_with_nuts, filter_grid, how='inner', predicate='intersects')

    # Find the buildings that do not overlap with any grid file
    non_overl_gisco_buildings = chunk_with_nuts[~chunk_with_nuts.index.isin(gisco_buildings.index)]

    if not non_overl_gisco_buildings.empty:
        # Perform a nearest neighbor search to find the closest grid file to each non-overlapping building
        nearest_gisco = gpd.sjoin_nearest(non_overl_gisco_buildings, filter_grid, how='left', distance_col='distance')
        # Assign the Gisco ID to each non-overlapping building
     
        # Concatenate the two GeoDataFrames
        df_with_giscoid = gpd.GeoDataFrame(pd.concat([gisco_buildings, nearest_gisco], ignore_index=True))
        df_with_giscoid.drop_duplicates(subset='geometry', inplace=True)
    else:
        df_with_giscoid = gisco_buildings
        df_with_giscoid.drop_duplicates(subset='geometry', inplace=True)

        print(f'All buildings have Gisco in chunk {idx}')
    print(f'{df_with_giscoid.gisco_grid.unique()} in {idx} ')
    return df_with_giscoid

def assign_unique_id(chunk_with_gisco):
    """
       Assign unique ID to buildings.
    # Building id: NUTS2024 (from buildings) + gisco_grid + latlon building"""
    chunk_with_gisco.drop_duplicates(subset='geometry', inplace=True)
    chunk_with_gisco['lat'] = chunk_with_gisco.geometry.centroid.y
    chunk_with_gisco['lon'] = chunk_with_gisco.geometry.centroid.x
    chunk_with_gisco['gisco_grid']=chunk_with_gisco['gisco_grid'].str.replace('0','',regex=False)
    chunk_with_gisco['lat_part']=chunk_with_gisco['lat'].astype(str).str[3:].astype(float)
    chunk_with_gisco['lon_part']=chunk_with_gisco['lon'].astype(str).str[3:].astype(float)
    chunk_with_gisco['unique_id'] = chunk_with_gisco.apply(
        lambda x: f"{x['nuts_id']}_{x['gisco_grid']}_Y{x['lat_part']:.4f}_X{x['lon_part']:.4f}", axis=1)

    return chunk_with_gisco


# Function to process a single tile
def process_tile(country_name, tile_to_read, data_fld, tiles_country_path, dbsm_v, nuts_fld, grid_file_10km, threshold=0.95, chunk_size=10000,large_build_thresh=20000):
    """
    Process a single tile for overlaps and save the cleaned version.
    large_build_thresh=20000 in m2
    """
    logger = setup_logger()  # each worker gets its own logger


    stats = {
        'country': country_name,
        'tile': tile_to_read,
        'num_poly_before': 0,
        'num_poly_after': 0,
        'total_area_before': 0.0,
        'total_area_after': 0.0
    }
    cleaned_chunks = []
    cleaned_tile=[]

    try:
        # Load tile GeoPackage
        tile_path = os.path.join(tiles_country_path, country_name, tile_to_read)
        tile_gpkg = gpd.read_file(tile_path, engine='pyogrio')

        iso2id = pycountry.countries.get(name=country_name).alpha_2  # ISO3 country code
        if iso2id == 'GR':
            iso2id='EL'

        nuts3_2024 = gpd.read_file(os.path.join(NUTS_fld, 'nuts3_2024_3035_100k.gpkg'), engine='pyogrio', 
                                   where=f"NUTS_ID LIKE '{iso2id}%'", use_arrow=True)


        nuts_gpkg = nuts3_2024[['geometry', 'NUTS_ID']]
        nuts_gpkg['area_nuts'] = nuts_gpkg.geometry.area
        nuts_gpkg = nuts_gpkg.rename(columns={'geometry': 'geometry_nuts'})
        nuts_gpkg.set_geometry('geometry_nuts', inplace=True)


        filter_grid=grid_file_10km[grid_file_10km['CNTR_ID'].str.contains(iso2id)]
        filter_grid['gisco_grid'] = filter_grid['GRD_ID'].str.split('m', expand=True)[1]
        filter_grid.drop(columns=['DIST_BORD', 'TOT_P_2018', 'TOT_P_2006', 'GRD_ID', 'TOT_P_2011',
                                     'Y_LLC', 'NUTS2016_3', 'NUTS2016_2', 'NUTS2016_1',
                                     'NUTS2016_0', 'LAND_PC', 'X_LLC', 'NUTS2021_3', 'NUTS2021_2',
                                     'DIST_COAST', 'NUTS2021_1'], inplace=True, errors='ignore')
        # check crs of buildings and nuts
        if nuts_gpkg.crs != tile_gpkg.crs:
            nuts_gpkg = nuts_gpkg.to_crs(tile_gpkg.crs)

        if filter_grid.crs != tile_gpkg.crs.name:
            filter_grid.to_crs(tile_gpkg.crs, inplace=True)

        # Define bounding box for the tile
        xmin, ymin, xmax, ymax = tile_gpkg.total_bounds
        bbox = (xmin, ymin, xmax, ymax)
        logger.info(f"LOADING: {tile_to_read}")
        water_gdf = gpd.read_file(WATER_MASK_GPKG_PATH, bbox=bbox,engine='pyogrio')
        
        clc_forest_gdf=gpd.read_file(os.path.join(CLC_GPKG_DIR, 'clc_311-324.gpkg'),
                                     bbox=bbox,engine='pyogrio')
        ## Load the final DBSMproduct data within the bounding box
        #iso3id = pycountry.countries.get(name=country_name).alpha_3  # ISO3 country code
        # load the merged dbsm for the country i want and then read tile
        glob_pattern = os.path.join(data_fld, f"{dbsm_v}-{country_name}-merge_*.gpkg")

        buildings_file = glob.glob(glob_pattern)[0]
        buildgs_all_tile_read = gpd.read_file(buildings_file, bbox=bbox, engine='pyogrio')

        if buildgs_all_tile_read is not None or buildgs_all_tile_read.empty:
            buildgs_all_tile_read['geometry']=buildgs_all_tile_read.normalize()
            buildings_less5m2 = buildgs_all_tile_read[buildgs_all_tile_read['area']<=5].copy()
            buildings_over5m2 = buildgs_all_tile_read[buildgs_all_tile_read['area']>5].copy() # keep the buildings >5m2
            buildings_over5m2.drop(columns=['NUTS_ID'], inplace=True, errors='ignore')

            stats['num_poly_before'] = len(buildgs_all_tile_read)
            stats['total_area_before'] = buildgs_all_tile_read.geometry.area.sum()

      
     


        # Break the tile into smaller chunks of polygons
            num_chunks = len(buildings_over5m2) // chunk_size + 1  # Number of chunks
            chunks = [buildings_over5m2.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]


        # Process each chunk of polygons for overlap removal=
            for idx,chunk in enumerate(tqdm(chunks,desc='Processing chunk'),start=1):
                print(f'{idx}')
                if chunk is not None and not chunk.empty:
                    chunk.loc[:, 'id'] = chunk.index  # Ensure unique ID for each polygon, df_test['id']=df_test.index
                    if 'area' not in chunk.columns:
                        chunk['area']=chunk.geometry.area

                    cleaned_chunk = calculate_and_remove_overlaps(chunk, threshold=threshold)
                    cleaned_chunk['geometry'] = cleaned_chunk.normalize()
                    cleaned_chunk.drop_duplicates(subset='geometry', keep='first', inplace=True)
                    chunk_after_drop = cleaned_chunk
                    #chunk.loc[:, 'area_1'] = chunk.geometry.area  # Add area field for overlap calculation

                    chunk_with_nuts = assign_nuts_id_and_generate_stats(chunk_after_drop, nuts_gpkg, idx)
                    chunk_with_nuts['geometry'] = chunk_with_nuts.normalize()
                    chunk_with_nuts.drop_duplicates(subset='geometry',keep='first', inplace=True)
                    print(f'len nuts chunk:{len(chunk_with_nuts)}')

                    chunk_with_gisco = assign_gisco_id(chunk_with_nuts, filter_grid, idx)
                    chunk_with_gisco['geometry'] = chunk_with_gisco.normalize()
                    chunk_with_gisco.drop_duplicates(subset='geometry', keep='first', inplace=True)
                    logger.info(f'len gisco chunk:{len(chunk_with_gisco)}')

                    chunk_with_unique_id = assign_unique_id(chunk_with_gisco)
                    chunk_with_unique_id.drop(columns=['index_right', 'NUTS2021_0','id','lat','lon','gisco_grid','distance','lat_part', 'NUTS_ID','lon_part'],inplace=True,errors='ignore')
                    chunk_with_unique_id['geometry'] = chunk_with_unique_id.normalize()
                    chunk_with_unique_id.drop_duplicates(subset='geometry',keep='first',inplace=True)
                    chunk_with_unique_id.drop_duplicates(subset='unique_id', keep='first', inplace=True)
                    print(f'final {len(chunk_with_unique_id)}')

                    ############################
                    chunk_filtered = chunk_with_unique_id[chunk_with_unique_id['area'] >= large_build_thresh]

                    if not chunk_filtered.empty:
                        # check overalaping with forest from clc 311 to 324 categories and create a new column called flag
                        chunk_forest_joined = gpd.sjoin(chunk_filtered, clc_forest_gdf, how='inner',
                                                        predicate='intersects')
                        forest_intersecting_indices = chunk_forest_joined.index
                        if not forest_intersecting_indices.empty:
                            chunk_with_unique_id.loc[forest_intersecting_indices, 'flag'] = 'potential forest overlay'  # flag the buildngs with forest intersecting
                            chunk_with_unique_id.loc[forest_intersecting_indices].to_file(os.path.join(r'/scratch/kakouge/',f'forest_interc{country_name}-{tile_to_read}.gpkg'), driver='GPKG')
                        else:
                            chunk_with_unique_id['flag'] = None

                        chunk_joined = gpd.sjoin(chunk_filtered, water_gdf, how='inner', predicate='intersects')
                        water_intersecting_indices = chunk_joined.index
                        if water_intersecting_indices is not None and not water_intersecting_indices.empty:
                            intersect_builds = chunk_with_unique_id.loc[water_intersecting_indices]  # Get  the intersecting  with water
                            chunk_with_unique_id.loc[water_intersecting_indices, 'flag'] = 'potential water overlay'  # flag the buildngs with forest intersecting


                            #chunk_after_drop = cleaned_chunk.drop(intersecting_indices)  # Delete the intersecting buildings from the original GeoDataFrame
                            intersect_builds.to_file(os.path.join(r'/scratch/kakouge/', f'water_interc{country_name}-{tile_to_read}-{idx}.gpkg'),driver='GPKG')
                            logger.info(f'Itersecting bodies or building more than >20000 in: {idx} chunk: {tile_to_read}')
                    else:
                        logger.info(f'No intesecting bodies or building more than >20000 in {idx} chunk {tile_to_read}')
                        chunk_with_unique_id['flag'] = None
                    print(f'original len chunk:{len(chunk)} and after drop {len(chunk_after_drop)}')
                    ############################
                    chunk_with_unique_id = save_attributes_as_json_col(chunk_with_unique_id)
                    cleaned_chunks.append(chunk_with_unique_id)

                cleaned_chunks = [chunk.drop_duplicates(subset='unique_id', keep='first') for chunk in
                                  cleaned_chunks]

                logger.info(f"completed:{idx} of {num_chunks} in {tile_to_read}")


            cleaned_tile_conc = pd.concat(cleaned_chunks)
            unique_ids = cleaned_tile_conc['unique_id'].drop_duplicates().index
            unique_geometries = cleaned_tile_conc['geometry'].drop_duplicates().index
            # Find the intersection of these indices if you want to keep only rows where both are unique
            unique_indices = list(set(unique_ids) & set(unique_geometries))
            cleaned_tile = cleaned_tile_conc.loc[unique_indices]

            stats['num_poly_after'] = len(cleaned_tile)
            stats['total_area_after'] = cleaned_tile.geometry.area.sum()
        ###HERE add nuts id and GISCO id for each building
        # Save cleaned tile
            logger.info(f"CLEANED TILE SAVED: {tile_to_read}")
        else:
            stats['num_poly_before'] = 0
            stats['total_area_before'] = 0
            stats['num_poly_after'] = 0
            stats['total_area_after'] = 0

        return stats,cleaned_chunks

    except Exception as e:
        logger.error(f"Error processing {tile_to_read}: {e}")
    return stats, cleaned_chunks


def check_columns(gdf,country_name,column_data_types):
    # Check if all expected columns exist
    field_names = gdf.columns.tolist()

    for col in column_data_types:
        if col not in field_names:
            print(f"Warning: Column '{col}' is missing from {country_name}")

    # Delete any extra columns
    extra_columns = [col for col in field_names if col not in column_data_types and col !='geometry']
    if extra_columns:
        print(f"Deleting extra columns from {country_name}: {extra_columns}")
        gdf.drop(columns=extra_columns, inplace=True)
     
    else:
        print(f"No extra columns found ")
    return gdf





def process_tiles_multiprocessing(country_names, tiles_country_path, data_fld, output_path, dbsm_v, today_day, nuts_fld, grid_file_10km):
    """
       Function to handle multiprocessing per tileper country
       and have as an output the clean buildings without duplicates and with the unique id code
       also, get some stats per tile and per country but also at nuts level
        """
    all_stats = []
    all_countries_nuts_stats=[]
    all_countries_nuts_stats=[]

    for country_name in country_names:
            to_merge_country=[]
            to_merge_stats=[]
      
            print(country_name)

            files_tiles = os.listdir(os.path.join(tiles_country_path, country_name))
            logger = setup_logger()

            # Create a pool of workers for multiprocessing
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.starmap(process_tile, [
                    (country_name, tile_to_read, data_fld, tiles_country_path, dbsm_v, nuts_fld,
                                 grid_file_10km)
                    for tile_to_read in files_tiles
                ])

            for result in results:
                try:
                    stats_tile, df_tile = result
                #print(f"Stats: {stats_tile}, Dataframe: {df_tile}")
                    if not df_tile:
                        print('empty df_tile')
                        continue

                    merge_tile_df = pd.concat(df_tile, ignore_index=True)
                    merge_tile_df.drop(columns=['CNTR_ID'],axis=1,errors='ignore',inplace=True)

                    to_merge_country.append(merge_tile_df) #merge gpkgs per tile to save it per country
                    to_merge_stats.append(stats_tile)
                    to_save_country_conc = pd.concat(to_merge_country,ignore_index=True)
                  
                    to_save_country = to_save_country_conc.drop_duplicates(subset='unique_id')
                    to_save_stats = pd.DataFrame(to_merge_stats)
                except Exception as e:
                    print(f"Error processing result {result}: {e}")



            # Get the count of polygons for each nuts_id using groupby
            filtered_data_50m2 = to_save_country[to_save_country['area']<50]
            num_build_less50m2 = filtered_data_50m2.groupby('nuts_id').size().reset_index(name='num_build_less<50m2')
            total_area_less50m2 = round(filtered_data_50m2.groupby('nuts_id').area.sum())

            stats_nuts = to_save_country.groupby('nuts_id').agg(
                num_buildings=('geometry','size'), #count buildings
                total_area=('area','sum'),
                mean_build_area=('area', 'mean'),
                max_build_area=('area', 'max'),
                min_build_area = ('area', 'min')
               # mean_height=('height','mean'),
               # mean_shapefactor=('shapefactor','mean'),
            ).reset_index().round()
            stats_nuts = stats_nuts.merge(num_build_less50m2, on='nuts_id', how='left')
            stats_nuts = stats_nuts.merge(total_area_less50m2, on='nuts_id', how='left')
            stats_nuts.rename(columns={'area':'total area of build <50m2'},inplace=True)


            # Save the statistics to a CSV file
            stats_nuts.to_csv(os.path.join(output_path, f"{country_name}_{dbsm_v}{today_day}_nuts_stats.csv"), index=True)
            logger.info(f"NUTS id assigned stats saved: {country_name}")

            all_countries_nuts_stats.append(stats_nuts)


            to_save_country = check_columns(to_save_country,country_name,column_data_types)
            to_save_country_path = os.path.join(output_path, f'{country_name}_{dbsm_v}_{today_day}_clean_overlay.gpkg')
            to_save_country.to_file(to_save_country_path, driver="GPKG")

            to_save_stats_path = os.path.join(output_path,f'{country_name}_{dbsm_v}_{today_day}_overlay_stats.csv')
            to_save_stats.to_csv(to_save_stats_path, index=False)


        

            logger.info(f"Process completed: {country_name}")

    to_save_all_stats_nuts=pd.concat(all_countries_nuts_stats)
    to_save_all_stats_nuts.to_csv(os.path.join(output_path, f"all_countries_{dbsm_v}{today_day}_nuts_stats.csv"),index=True)
    logger.info(f"stats all NUTS id saved")

  


def main():
    today_day = date.today().strftime("%Y-%m-%d")
    dbsm_v = 'dbsm-v2'

    data_fld = os.getenv(DBSM R2025 Conflation output PATH)
    country_names = [f.split('_')[0] for f in filter(lambda f: f.endswith('.gpkg'), os.listdir(data_fld))]

    GRID_FILE_10KM_PATH = os.getenv("GRID_FILE_10KM_PATH", "/path/to/geographical/grid/") #gpkg
    NUTS_fld= os.getenv("nuts3_2024_3035_100k", "/path/to/geographical/grid/") #gpkg

    tiles_country_path = os.getenv(PATH with gpkgs of EACH COUNTRY /country-tiles/'
    grid_file_10km = gpd.read_file( GRID_FILE_10KM_PATH ), engine='pyogrio')

    process_tiles_multiprocessing(country_names, tiles_country_path, data_fld, output_path, dbsm_v, today_day, nuts_fld, grid_file_10km)


if __name__ == "__main__":
    main()
