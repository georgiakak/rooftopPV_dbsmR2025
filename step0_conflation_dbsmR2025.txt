"""
Author: Georgia kakoulaki (georgia.kakoulaki@ec.europa.eu or georgia.kakoulaki@gmail.com)
Version: 1.0
Initial Creation: July 2024
Last Modification: April 2025

Copyright (c) 2025 European Union  
Licensed under the EUPL v1.2 or later: https://joinup.ec.europa.eu/collection/eupl/eupl-text-eup

Description:
----------
Python based tool for the creation of DBSM (Digital Building Stock Model) and processing version R2025.
The scriprts support:
-Cleaning and harmonization of the the 3 input databases (EUBOCCO(EUB), OpenStreetMap (OSM), Microsoft Buildings (MSB).
-The whole process is performed per country and tiling using the GISCO grid.
-Output as GeoPackage files.

Define coordinate system use : 'EPSG:3035' # all the input datasets were reprojected

Dataset DOI: https://doi.org/?
Code repository: https://code.europa.eu/?

For support, contact: ?@ec.europa.eu

"""


import os
import logging
import warnings
import glob
from datetime import date
import configparser

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon
from shapely.errors import ShapelyDeprecationWarning
from shapely.validation import make_valid

# Configure Logging
FORMAT = "[%(funcName)s:%(lineno)s - %(asctime)s] %(message)s"
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Configuration
config = configparser.ConfigParser()
config.read("config.ini")  # Load paths from a config file

# Paths (replace hardcoded paths with config options or environment variables)
EUB_FOLDER = os.getenv("EUB_FOLDER", "/path/to/eub/")
OSM_FOLDER = os.getenv("OSM_FOLDER", "/path/to/osm/")
MSB_FOLDER = os.getenv("MSB_FOLDER", "/path/to/msb/")
NUTS_GPKG = os.getenv("NUTS_GPKG", "/path/to/nuts.gpkg")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/path/to/output/")
PER_TILE_OUTPUT = os.getenv("PER_TILE_OUTPUT", "/path/to/tile/output/")

# Country Groups
COUNTRIES_EUB_OSM = ['austria', 'germany', 'italy', 'czechia', 'austria', 'belgium']
COUNTRIES_ONLY_OSM = ['bulgaria', 'greece', 'hungary', 'croatia', 'ireland', 'latvia', 'portugal', 'sweden', 'romania']

# DBSM Version
VERSION_DBSM = "dbsm-v2"

# NUTS GeoPackage Initialization
try:
    nuts_gpkg = gpd.read_file(NUTS_GPKG)
    nuts_gpkg = nuts_gpkg[['geometry', 'NUTS_ID']]  # Ensure only necessary columns are kept
    logger.info("Loaded NUTS GeoPackage successfully.")
except Exception as e:
    logger.error(f"Failed to load NUTS GeoPackage: {e}")
    nuts_gpkg = None


def check_crs_equivalence(eub_tile, osm_tile, msb_tile):
    """Check the CRS of each dataset to be the same, if not you need to convert

    Args:
        eub_tile GeoDataframe: Eubucco dataset
        osm_tile GeoDataframe: Open Street Map dataset
        msb_tile GeoDataframe: Microsoft dataset

    Returns:
        str: Boolean confirming or not projection of the 3 dataset
    """
    eub_crs = eub_tile.crs
    osm_crs = osm_tile.crs
    msb_crs = msb_tile.crs

    if eub_crs.equals(osm_crs) and eub_crs.equals(msb_crs):
        logger.info("All CRS are equivalent.")
        return True
    else:
        logger.info("CRS are not equivalent.")
        logger.info(f"EUB CRS: {eub_crs}")
        logger.info(f"OSM CRS: {osm_crs}")
        logger.info(f"MSB CRS: {msb_crs}")
        return False


def valid_data(clip_gfd, source_name):
    """_summary_
    clip the gpkg using the tile,check validity of each geometry feature

    Args:
        clip_gfd (_type_): geodataframe
        source_name (_type_): _description_

    Returns:
        None: missing return as IF is not valid
    """
    if clip_gfd is not None and not clip_gfd.empty:
        clip_gfd.drop_duplicates(subset='geometry', keep='first', inplace=True)
      
        # Create a column containing whether is a valid geometry or not
        clip_gfd['validity'] = clip_gfd.apply(lambda row: explain_validity(row.geometry), axis=1)
        # provides a valid representation of the geometry, if it is invalid
        clip_gfd['geometry'] = clip_gfd['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

        filter_clip = clip_gfd[clip_gfd.geom_type.isin(['Polygon','MultiPolygon'])]
        # Check if there are any MultiPolygon geometries
        contains_multipolygon = any(isinstance(geom, MultiPolygon) for geom in filter_clip.geometry)

        if contains_multipolygon:
            logger.info(f"{source_name} It has multipolygons")
            clip_to_save = filter_clip.explode(column='geometry', ignore_index=True)
        else:
            logger.info(f"{source_name} No multi-polygons")
            clip_to_save = filter_clip.copy()
    else:
        clip_to_save = clip_gfd.copy()

    clip_to_save['source'] = source_name
    return clip_to_save

def info_gdf(gdf1):

    """
    check basic info about a gdf
    Args: a geodataframe
    Returns:info_gdf_str: return a string with the info
    """
    contains_geometry = any('geometry' in column for column in gdf1.columns)
    duplicates = gdf1[gdf1.duplicated(subset='geometry', keep=False)]
    gdf_crs = gdf1.crs
    info_gdf_str = f'No buildings: {len(gdf1)}, Area km2: {round(gdf1.geometry.area.sum() / 1e6, 3)}, Duplicates: {len(duplicates)}'
    return info_gdf_str

def parse_properties(json_str):

    """read a json string a take the properties
    Args:
        json_str (_type_): _description_
    Returns:
        height, confidence: properties as heigh and confidence level from the MSB layers
    """
    try:
        properties = json.loads(json_str)
        height = properties.get('height', None)
        confidence = properties.get('confidence', None)
    except json.JSONDecodeError:
        height, confidence = None, None
    return height, confidence

def msb_layer_check(msb_valid):
    """Perfom multiple checks on Microsoft GlobalML Building Footprints,delete duplicates if exist
    Args:
        msb_valid GeoDataFrame: applied some validity check in output function valid_data
    Returns:
        GeoDataFrame: GeoDataFrame with some modification applied
    """
    logger.info(f'Started Checking Microsoft GlobalML Building')
    # Dropping duplicates
    msb_checked = msb_valid.drop_duplicates(subset='geometry')
    # Multiple Assignment
    msb_checked['height'], msb_checked['confidence'] = zip(*msb_checked['properties'].apply(parse_properties))
    # Remove column properties
    msb_checked.drop(['properties'], axis=1, errors='ignore', inplace=True)
    # Sets the GEOMETRY COLUMN
    msb_checked.set_geometry('geometry', inplace=True)
    logger.info(f'Checking Microsoft GlobalML Building Finished')
    return msb_checked

def create_df_stat(files_tiles):
    ##create an empty df for each country for al the tiles to store some basics stats
    index_for_stats = ['final_EUB_Total_Area_km2', 'final_OSM_Total_Area_km2', 'final_MSB_Total_Area_km2',
                      'original_EUB_Total_Area_km2', 'original_OSM_Total_Area_km2', 'original_MSB_Total_Area_km2',
                      'final_build_num_EUB', 'final_build_num_OSM', 'final_build_num_MSB',
                      'original_build_num_EUB', 'original_build_num_OSM', 'original_build_num_MSB']
    columns_for_stats = [tile.split('.')[0] for tile in files_tiles]
    df_stats = pd.DataFrame(index=index_for_stats, columns=columns_for_stats)
    return df_stats

def drop_columns_if_exist(df, column_names):
    """ Drops columns from the DataFrame if they exist.
    Args:
        param df: pandas.DataFrame to be modified.
        param column_names: A list of column names to drop.
    Returns:
        The DataFrame with the specified columns dropped if they existed.
    """
    # Ensure column_names is a list of strings (column names)
    assert all(isinstance(col, str) for col in column_names), "column_names must be a list of strings"

    # Determine which of the provided column names are in the DataFrame
    columns_to_drop = [col for col in column_names if col in df.columns]
    # If there are any columns to drop, drop them
    if columns_to_drop:
        df.drop(columns=columns_to_drop,  axis=1, errors='ignore', inplace=True)
    return df

def replace_column_suffix(gdf, old_suffix, new_suffix=''):
    """ Replace column suffixes in a GeoDataFrame.
       Args:
           gdf: GeoDataFrame to be modified.
           old suffix: suffix to be replaced
           new_suffix: the new suffix, default is ''
       Returns:
           the GeoDataFrame with the modified columns
       """
    # filtered_data.keys()
    for col in gdf.columns:
        if col.endswith(old_suffix):
            new_col = col.replace(old_suffix, '')
            gdf.rename(columns={col: new_col}, inplace=True)
    return gdf

def set_new_geomtry(gdf, geometry_column_name='eub_geomtry', new_geometry_name='geometry'):
    if geometry_column_name in gdf.columns:
        gdf['geometry'] = gdf[geometry_column_name]  # set the eub geometry
        gdf.set_geometry(new_geometry_name, inplace=True)  # set the new geometry as the active geometry
        gdf.drop(geometry_column_name, axis=1, inplace=True)
    else:
        raise KeyError(f'Column {geometry_column_name} not found in GeoDataFrame')
    return gdf


def areacomparisonoverlap(overlay_eubosm, eub_overlaid, percentage_overlap):
    """Compares two dataset and their area of overalp, using spatial join
    Args:
        gdf1 (geodataframe polygons): features osm overlaid to eub
        gdf2 (geodataframe polygons): eub dataset to be saved
        percentage_overlap (integer):set by the user, in our case we use 20 (%)
    Returns:
        selected_f_area_eubosm (geodataframe): geodatagrame containing only the selcted polygons not overalying
    """
    gdf1 = overlay_eubosm
    gdf1['osm_index'] = gdf1.index
    if 'osm_area' not in gdf1.columns:
        gdf1['osm-area'] =gdf1.geometry.area

    gdf2 = eub_overlaid
    gdf2.rename(columns={'geometry': 'eub_geometry'}, inplace=True)
    gdf2.set_geometry('eub_geometry', inplace=True)
    gdf2['eub-area'] = gdf2.eub_geometry.area
    gdf2['eub_index'] = gdf2.index

    def select_largest_overlap(group):
        #give the items with the largest overlap
        return group.loc[group['intersection_area'].idxmax()]


    sjoin_result = gpd.sjoin(gdf1, gdf2, how='inner', predicate='intersects')
    sjoin_result['intersection_area'] = sjoin_result.apply( lambda row: row.geometry.intersection(gdf2.loc[row['eub_index'], 'eub_geometry']).area, axis=1)
  
    # Calculate the overlap ratios
    sjoin_result['ratio_osm'] = round(100 * sjoin_result['intersection_area'] / sjoin_result['osm-area'], 2)
    sjoin_result['ratio_eub'] = round(100 * sjoin_result['intersection_area'] / sjoin_result['eub-area'], 2)
    intersection_result = sjoin_result[ (sjoin_result['ratio_osm'] >= percentage_overlap) | (sjoin_result['ratio_eub'] >= percentage_overlap)]
    non_intersection = sjoin_result[(sjoin_result['ratio_osm'] < percentage_overlap) & (sjoin_result['ratio_eub'] < percentage_overlap)]

    filtered_result=intersection_result.groupby('eub_index').apply(select_largest_overlap).reset_index(drop=True)
   
    eub_with_OSMtags = gdf2.merge(filtered_result, how='inner', left_index=True, right_on='eub_index', indicator=True)
    eub_with_OSMtags.drop('geometry', axis=1, inplace=True) #drop osm geometry
    eub_with_OSMtags['geometry'] = eub_with_OSMtags['eub_geometry'] #set the eub geometry
    eub_with_OSMtags.set_geometry('geometry', inplace=True)  # set the new geometry as the active geometry
    eub_with_OSMtags.drop('eub_geometry', axis=1, inplace=True)
    #remove duplicates and normalize geometry
    eub_with_OSMtags['fid'] = range(len(eub_with_OSMtags))
    eub_with_OSMtags['geometry'] = eub_with_OSMtags.normalize()
    eub_with_OSMtags.drop_duplicates(subset='geometry',keep='first',inplace=True)

    eub_with_OSMtags = eub_with_OSMtags[eub_with_OSMtags['_merge'] == 'both']
    eub_with_OSMtags.drop(columns=[ 'base_saveindex_x','eub_index_x', 'validity',
       'source_left', 'osm_index', 'index_right', 'source_right','base_saveindex_y',
                                'eub_index_y', 'eub_index','eub_area_y', 'eub-height_y','eub-age_y','intersection_area',
                                    'ratio_osm','ratio_eub', 'eub-type_y','_merge'],
                          errors='ignore', inplace=True, axis=1)

    eub_with_OSMtags = replace_column_suffix(eub_with_OSMtags,'_x')


    eub_withno_OSMtags = gdf2.merge(non_intersection, how='inner', left_index=True, right_on='eub_index', indicator=True)
    eub_withno_OSMtags.drop('geometry', errors='ignore', inplace=True, axis=1)  # drop osm geometry
    eub_withno_OSMtags['geometry'] = eub_withno_OSMtags['eub_geometry']  # set the eub geometry
    eub_withno_OSMtags.set_geometry('geometry', inplace=True)  # set the new geometry as the active geometry
    eub_withno_OSMtags.drop('eub_geometry', axis=1, inplace=True)
    #normalize and remove duplicates
    eub_withno_OSMtags['fid'] = range(len(eub_withno_OSMtags))
    eub_withno_OSMtags['geometry'] = eub_withno_OSMtags.normalize()
    eub_withno_OSMtags.drop_duplicates(subset='geometry', inplace=True)

    eub_withno_OSMtags = eub_withno_OSMtags[eub_withno_OSMtags['_merge'] == 'both']
    eub_withno_OSMtags.drop(columns=['base_saveindex_x', 'eub_index_x', 'validity',
                                   'source_left', 'osm_index', 'index_right', 'source_right', 'base_saveindex_y',
                                   'eub_index_y', 'eub_index', 'eub_area_y', 'eub-height_y', 'eub-age_y',
                                   'intersection_area', 'osm-building', 'osm-levels', 'osm-roof-shape', 'osm-height', 'osm-date', 'osm_area',
                                   'ratio_osm', 'ratio_eub', 'eub-type_y', '_merge'],
                          errors='ignore', inplace=True, axis=1)

    eub_withno_OSMtags = replace_column_suffix(eub_withno_OSMtags,'_x')
    return eub_with_OSMtags, eub_withno_OSMtags



def find_non_overlay_items(gfd_tobechecked, gfd_base):
        """Perfom spatial join between two geodataframes
           and identify which buildings do not exist
        Args:
            gfd_tobechecked: geodataframe you need to check
            gfd_base: your base geodataframe
        Returns:
            GeoDataFrame: non_overlay_items, overlay_items
        """
        if gfd_base is not None and not gfd_base.empty:
            gfd_tobechecked['saveindex'] = gfd_tobechecked.index
            sjoin_result = gfd_base.sjoin(gfd_tobechecked, how='inner')['saveindex']  # spatial join eub and osm

            non_overlay_polyg = gfd_tobechecked[~gfd_tobechecked.saveindex.isin(sjoin_result)].copy()  # osm items that do not exist in eub
            non_overlay_polyg.drop_duplicates(subset='geometry', keep='first', inplace=True)  ##delete duplicates
            non_overlay_polyg.drop(columns=['saveindex'],errors='ignore',axis=1, inplace=True)
        else:
            non_overlay_polyg = gfd_tobechecked

        return non_overlay_polyg

def identify_nonexisting(gfd_tobechecked, gfd_base, buffer_distance=0.001):
    """Perfom spatial join between two geodataframes
       and identify which buildings do not exist
    Args:
        gfd_tobechecked: geodataframe you need to check
        gfd_base: your base geodataframe
    Returns:
        GeoDataFrame: non_overlay_items, overlay_items
    """
    if gfd_base is not None and not gfd_base.empty:
        gfd_tobechecked['saveindex'] = gfd_tobechecked.index
        sjoin_result = gfd_base.sjoin(gfd_tobechecked, how='inner')['saveindex']  # spatial join eub and osm

        non_overlay_items = gfd_tobechecked[~gfd_tobechecked.saveindex.isin(sjoin_result)].copy()  # osm items that do not exist in eub
        non_overlay_items.drop_duplicates(subset='geometry', keep='first', inplace=True)  ##delete duplicates
   
        overlay_items = gfd_tobechecked[gfd_tobechecked.saveindex.isin(sjoin_result)].copy()  #### overlaying buildings between EUB and OSM, results OSM overlaid buildings
        overlay_items.drop_duplicates(subset='geometry', keep='first', inplace=True)
        overlay_items['fid'] = range(len(overlay_items))
     

        gfd_base['base_saveindex'] = gfd_base.index
        sjoin_result_base = overlay_items.sjoin(gfd_base, how='inner')  # spatial join, want to find teh eub that do not overlay with osm
        overlapping_indices_base = sjoin_result_base['base_saveindex'].unique()
        overlap_items_base = gfd_base[gfd_base.base_saveindex.isin(overlapping_indices_base)].copy()
       

        non_overlap_base = gfd_base[~gfd_base.base_saveindex.isin(overlapping_indices_base)].copy()  # eub items that do not overlay with osm
        non_overlap_base.drop_duplicates(subset='geometry', keep='first', inplace=True)
      
    else:
        non_overlay_items = gfd_tobechecked
        overlay_items = None
        non_overlap_base = gfd_base
    return non_overlay_items, overlay_items, overlap_items_base, non_overlap_base



def msb_selectheight(gfd, height_lim, confidence_lim):
    """_modifies the original MSB layer attribute table
    keeps the MSB height attribute only if the confidence_lim is satisfied (95%), read the MSB manual
    Args:
        msb_to_work (_type_): _description_
        height_lim (_type_): _description_
        confidence_lim (_type_): _description_

    Returns:
        _type_: _description_
    """
    gdf_filtered = gfd.copy()
    condition = ( gdf_filtered['height'] > height_lim) & ( gdf_filtered['confidence'] >= confidence_lim)
    gdf_filtered['msb_height'] = np.where(condition, gfd['height'], np.nan)
    gdf_filtered.drop_duplicates(subset='geometry', keep='first', inplace=True)
    return  gdf_filtered


def clean_msbstep(msb_to_work, merge_eubosm):

	"""_modifies the original MSB, check geometry validity, 
	keep MSB height values using a threshold on the confidece 
	level provided by the MSB source data
    """
  
    if not merge_eubosm is None:
        merge_eubosm.drop(columns=['validity', 'centroid', 'saveindex'], axis=1, errors='ignore', inplace=True)
    msb_clean_tile = pd.DataFrame()
    if not msb_to_work.empty:
        non_overlay_msb_eubosm = find_non_overlay_items(msb_to_work, merge_eubosm)
        if non_overlay_msb_eubosm is not None and not non_overlay_msb_eubosm.empty:
            msb_to_save = msb_selectheight(non_overlay_msb_eubosm, height_lim = 0, confidence_lim = 0.9)

            if msb_to_save is not None and not msb_to_save.empty:
                msb_to_save.drop(columns=['validity', 'height', 'confidence'], axis=1, errors='ignore', inplace=True)
                msb_clean_tile = msb_to_save
                merge_eubosmsb = pd.concat([merge_eubosm, msb_to_save], ignore_index=True)
            else:
                merge_eubosmsb = merge_eubosm
                merge_eubosmsb['msb-height'] = pd.Series(dtype='float')

    else:
        merge_eubosmsb = merge_eubosm
        merge_eubosmsb['msb-height'] = pd.Series(dtype='float')


        print('empty msb')
    return msb_clean_tile, merge_eubosmsb


def eub_osm_join(eub_tile, osm_tile):

    """Creates a new geodataframe using the EUBUCCO and OSM dataset,after applying overlaying and spatial join
    Args:
        eub_tile GeoDataframe: Eubucco dataset
        osm_tile GeoDataframe: Open Street Map dataset

    Returns:
        osm_clean_tile: OSM gedataframe
        eub_clean_tile: EUBUCCO geodataframe
        merge_eubosm: merged EUB and OSM, only buildings from OSM that do not exist in EUB
    """
   
    # CHECK GEOMETRY VALIDITY
    if not eub_tile.empty:
        eub_to_save = eub_tile
        # ROUND NUMBER TO TWO DECIMAL DIGITS
        eub_to_save['eub-height'] = round(eub_to_save['height'], 2).astype('float64')  # in cristiano's code name: tile '-a04aa-eub-height2dec.gpkg' and  '-a04a-eub-height.gpkg'
        # ROUND NUMBER TO TWO DECIMAL DIGITS
        eub_to_save['eub-age'] = round(eub_to_save['age'], 2).astype('float64')  # in cristiano's code name: tile '-a04b-eub-age.gpkg'
        eub_to_save['eub-type'] = (eub_to_save['type']).astype('string')  #
        # DROPS SEVERAL COLUMNS
        eub_to_save.drop(columns=['id', 'height', 'age', 'type', 'id_source', 'type_source', 'validity'], axis=1, errors='ignore', inplace=True)
        logger.info('Applied Modification to EUBUCCO dataset')
    else:
        eub_to_save = eub_tile
        logger.info('EUBUCCO dataset is empty')

    if not eub_to_save.empty:
        non_overlay_eubosm, overlay_eubosm, eub_overlaid, eub_not_overlaid = identify_nonexisting(osm_tile, eub_to_save,buffer_distance=0.001)  # non_overlay_eubosm=gives you the osm not overlaid with eub
    

        # ############ area comparison eub osm
        if not non_overlay_eubosm.empty:
            eub_with_OSMtags, eub_withno_OSMtags = areacomparisonoverlap(overlay_eubosm, eub_overlaid, 50)  # EUB/OSM using overlay area between osm_ratio and eub_ratio
            # ###################### merge all the  eub dataset with tags or no tags
            eub_with_or_no_OSMtags=gpd.GeoDataFrame(pd.concat([eub_not_overlaid, eub_with_OSMtags, eub_withno_OSMtags]))#.drop_duplicates())
            eub_with_or_no_OSMtags.drop_duplicates(subset='geometry',keep='first',inplace=True)
            eub_with_or_no_OSMtags['fid'] = range(len(eub_with_or_no_OSMtags))
            eub_with_or_no_OSMtags.drop(columns=['temp_index', 'saveindex','eub_index'],axis=1, errors='ignore', inplace=True)
            #merge al the eub data and the OSM that didnt overlap with eub

            merge_eubosm = gpd.GeoDataFrame(pd.concat([eub_with_or_no_OSMtags, non_overlay_eubosm]))#.drop_duplicates())
            merge_eubosm.drop_duplicates(subset='geometry', keep='first', inplace=True)
            merge_eubosm['fid'] = range(len(merge_eubosm))
           # merge_eubosm.to_file(r'/scratch/kakouge/test_duplicates_malta/merged_non_osm_all_eub.gpkg', driver='GPKG')

            merge_eubosm.drop(columns=['validity','base_saveindex','eub_index','temp_index', 'saveindex'], axis=1, errors='ignore', inplace=True)

            osm_clean_tile = non_overlay_eubosm # OSM only, not existing and
            osm_clean_tile.drop(columns=[ 'temp_index','saveindex' 'validity'], axis=1, errors='ignore', inplace=True)

            eub_clean_tile = eub_with_or_no_OSMtags.copy()
            logger.info('more than 1 building overlaping')
        else:
            eub_to_save.rename(columns={'eub_geometry': 'geometry'}, inplace=True)
            eub_to_save.set_geometry('geometry', inplace=True)
            merge_eubosm = pd.concat([eub_to_save, non_overlay_eubosm], ignore_index=True)
            merge_eubosm.drop_duplicates(subset='geometry', keep='first', inplace=True)
            osm_clean_tile = non_overlay_eubosm
            osm_clean_tile['source'] = 'osm'
            osm_clean_tile.drop_duplicates(subset='geometry', keep='first', inplace=True)
            eub_clean_tile = eub_to_save
            # osm_dummy.append(osm_clean_tile)
            # eub_dummy.append(eub_clean_tile)
            logger.info('no overlap OSM - EUB')
    else:
        osm_clean_tile = osm_tile
        osm_clean_tile['source'] = 'osm'
        #if not osm_clean_tile.empty:
        osm_clean_tile.drop_duplicates(subset='geometry', keep='first', inplace=True)
        merge_eubosm = pd.concat([eub_to_save, osm_clean_tile])
        eub_clean_tile = eub_to_save
        logger.info('OSM only, no overlap OSM - EUB')

    return osm_clean_tile, eub_clean_tile, merge_eubosm


def save_each_tile(tile_to_read, eub_data_save, osm_data_save, msb_data_save, output_gpkg_path, gpkg_all_tiles_name):
    """Save each tile and each source to a GeoPackage
    Args:
        tile (_type_): _description_
        eub_data_save (_type_): _description_
        osm_data_save (_type_): _description_
        msb_data_save (_type_): _description_
        output_gpkg_path (_type_): _description_
        gpkg_all_tiles_name (_type_): _description_
    """
    tile_base_name = os.path.splitext(os.path.basename(tile_to_read))[0]
    # Define layer names based on the tile and source
    eub_layer_name = f"dbsm_v3_{tile_base_name}_eub"
    osm_layer_name = f"dbsm_v3_{tile_base_name}_osm"
    msb_layer_name = f"dbsm_v3_{tile_base_name}_msb"

    full_output_path = os.path.join(output_gpkg_path, gpkg_all_tiles_name)
    # Function to save a layer if it's not empty and has a valid geometry column
    def save_layer(gdf, layer_name):
        max_attempts = 10
        retry_interval = 5
        attempt = 0
        while attempt < max_attempts:
            try:
                gdf.to_file(full_output_path, layer=layer_name, driver="GPKG")
                break
            except Exception as e:
                if 'database is locked' in str(e):
                    print(f'Database is locked, waiting for {retry_interval}seconds before retrying')
                    time.sleep(retry_interval)
                    retry_interval *= 2
                    attempt += 1
                    gdf.to_file(full_output_path, layer=layer_name, driver="GPKG")
                else:
                    #print(f'An error occurred: {e} {tile_to_read}')
                    separate_gpkg_name = os.path.join(output_gpkg_path, f'{tile_base_name}_separate.gpkg')
                    gdf.to_file(separate_gpkg_name, layer=layer_name, driver='GPKG')
                    print(f'Tile saved separately {e} {tile_to_read}')
    # Save each layer
    save_layer(eub_data_save, eub_layer_name)
    save_layer(osm_data_save, osm_layer_name)
    save_layer(msb_data_save, msb_layer_name)


def create_df_stat(tile_list):
    #create an empty df for each country for al the tiles to store some basics stats
    index_for_stats= ['final_EUB_Total_Area_km2', 'final_OSM_Total_Area_km2', 'final_MSB_Total_Area_km2',
                      'original_EUB_Total_Area_km2', 'original_OSM_Total_Area_km2', 'original_MSB_Total_Area_km2',
                      'final_build_num_EUB', 'final_build_num_OSM', 'final_build_num_MSB',
                      'original_build_num_EUB', 'original_build_num_OSM', 'original_build_num_MSB']
    columns_for_stats = [tile.split('.')[0] for tile in tile_list]
    df_stats = pd.DataFrame(index=index_for_stats, columns=columns_for_stats)
    return df_stats


def calculate_ms_errors(df_stats, country, tile_name, source1, source2):
    # Calculate the squared difference between the two area totals
    area_diff = df_stats.loc[f'final_{source1}_Total_Area_km2', tile_name] - df_stats.loc[f'final_{source2}_Total_Area_km2', tile_name]
    diff_rel = np.round(area_diff ** 2, 3)

    # Calculate the denominator for the variance
    var_denom = (df_stats.loc[f'final_build_num_{source1}', tile_name] +
                 df_stats.loc[f'final_build_num_{source2}', tile_name])

    # Ensure var_denom is treated as a scalar
    if isinstance(var_denom, pd.Series):
        var_denom = var_denom.iloc[0]  # Assuming you want the first element

    # Check for division by zero
    if var_denom != 0:
        ratio_rel = diff_rel / var_denom
        df_stats.loc[f'{country}_ms_err_{source1}', tile_name] = np.round(ratio_rel, 5)
    else:
        df_stats.loc[f'{country}_ms_err_{source1}', tile_name] = np.nan
        #print(f"Result stored in DataFrame: {df_stats.loc[f'{country}_ms_err_{source1}', tile_name]}")



def calculate_area_ratio(df_stats, country, tile_name, source1, source2):
# Extract values for both sources
    osm_value = df_stats.loc[f'final_{source1}_Total_Area_km2', tile_name]
    eub_value = df_stats.loc[f'final_{source2}_Total_Area_km2', tile_name]
# Handle the case where osm_value and eub_value are Series
    if isinstance(osm_value, pd.Series): osm_value = osm_value.iloc[0]
# Assuming you want the first element
    if isinstance(eub_value, pd.Series): eub_value = eub_value.iloc[0]
# Assuming you want the first element #
# Check for division by zero or missing values
    if osm_value != 0 and eub_value != 0:
        ratio_area = osm_value / eub_value
        df_stats.loc[f'{country}_ratio_area_{source1}', tile_name] = np.round(ratio_area, 3)
    else:
        df_stats.loc[f'{country}_ratio_area_{source1}', tile_name] = np.nan


def calculate_rel_errors(df_stats, country, tile_name, col_osm, col_msc, source):
# Calculate the difference between the two columns
    diff_data = df_stats.loc[col_osm, tile_name] - df_stats.loc[col_msc, tile_name]
    var_den_res = df_stats.loc[col_msc, tile_name]
# Ensure var_den_res is treated as a scalar
    if isinstance(var_den_res, pd.Series):
        var_den_res = var_den_res.iloc[0]
# Check for division by zero
    if var_den_res != 0:
        ratio_data = diff_data / var_den_res
        df_stats.loc[f'{country}_rel_err_{source}', tile_name] = np.round(ratio_data, 5)
    else:
        df_stats.loc[f'{country}_rel_err_{source}', tile_name] = np.nan


def calculate_rel_errors(df_stats, country, tile_name, col_osm, col_msc, source):
# Calculate the difference between the two columns
    diff_data = df_stats.loc[col_osm, tile_name] - df_stats.loc[col_msc, tile_name]
    var_den_res = df_stats.loc[col_msc, tile_name]
# Ensure var_den_res is treated as a scalar
    if isinstance(var_den_res, pd.Series):
        var_den_res = var_den_res.iloc[0]
# Check for division by zero
    if var_den_res != 0:
        ratio_data = diff_data / var_den_res
        df_stats.loc[f'{country}_rel_err_{source}', tile_name] = np.round(ratio_data, 5)
    else:
        df_stats.loc[f'{country}_rel_err_{source}', tile_name] = np.nan

def calculate_build_ratio(df_stats, country, tile_name, source1, source2):

# Extract building values for both sources
    osm_value = df_stats.loc[f'final_build_num_{source1}', tile_name]
    eub_value = df_stats.loc[f'final_build_num_{source2}', tile_name]
# Handle the case where osm_value and eub_value are Series
    if isinstance(osm_value, pd.Series):
        osm_value = osm_value.iloc[0]
# Assuming you want the first element
    if isinstance(eub_value, pd.Series):
        eub_value = eub_value.iloc[0]
# Assuming you want the first element # Check for division by zero or missing values
    if osm_value != 0 and eub_value != 0:
        ratio_build = osm_value / eub_value
        df_stats.loc[f'{country}_ratio_build_{source1}', tile_name] = np.round(ratio_build, 3)
    else:
        df_stats.loc[f'{country}_ratio_build_{source1}', tile_name] = np.nan

def calculate_tile_basic_stats(df_stats, tile_name, original_tile, clean_tile, source ):

    if original_tile is not None and not original_tile.empty:
        logger.info(f'{source}_original_tile is not NONE')
        df_stats.at[f'original_{source}_Total_Area_km2', tile_name] = round(original_tile.geometry.area.sum() / 1e6, 5)  # clipped tile osm original
        df_stats.at[f'original_build_num_{source}', tile_name] = len(original_tile)
    else:
        logger.info(f'{source}_original_tile is NONE')
        df_stats.at[f'original_{source}_Total_Area_km2', tile_name] = 0  # clipped tile osm original
        df_stats.at[f'original_build_num_{source}', tile_name] = 0

    if clean_tile is not None and not clean_tile.empty:
        logger.info(f'{source}_clean_tile is not NONE')
        df_stats.at[f'final_{source}_Total_Area_km2', tile_name] = round(clean_tile.geometry.area.sum() / 1e6, 5)  # final clean tile
        df_stats.at[f'final_build_num_{source}', tile_name] = len(clean_tile)
    else:
        logger.info(f'{source}_clean_tile is NONE')
        df_stats.at[f'final_{source}_Total_Area_km2', tile_name] = 0
        df_stats.at[f'final_build_num_{source}', tile_name] = 0


def stat_tile_country(df_stats, tile_name, country):
    """assign some basic stats for  each working tile
        Args:
           df_stats,tile_name,
           osm_tile,osm_clean_tile,eub_tile,
           eub_clean_tile,msb_tile,
           msb_clean_tile,country

        Returns:
            _type_: _description_
        """
    ####mean squared error
    calculate_ms_errors(df_stats, country, tile_name, 'OSM', 'EUB')
    calculate_ms_errors(df_stats, country, tile_name, 'MSB', 'EUB')
    #####realtive error
    calculate_rel_errors(df_stats, country, tile_name, 'final_MSB_Total_Area_km2', 'final_EUB_Total_Area_km2', 'MSB')
    calculate_rel_errors(df_stats, country, tile_name, 'final_OSM_Total_Area_km2', 'final_EUB_Total_Area_km2', 'OSM')

    #######area ratio
    calculate_area_ratio(df_stats, country, tile_name, 'OSM', 'EUB')
    calculate_area_ratio(df_stats, country, tile_name, 'MSB', 'EUB')
    #######building ratio
    calculate_build_ratio(df_stats, country, tile_name,'OSM','EUB')
    calculate_build_ratio(df_stats, country, tile_name,'MSB','EUB')

    df_stats.iloc[:, :] = df_stats.iloc[:, :].apply(pd.to_numeric, errors='coerce')
    # Sum horizontally for the first 12 rows and assign to 'Total' column
    #df_stats.loc[0:11, 'Total'] = df_stats.iloc[0:12].sum(axis=1).round(2)
    logger.info('all STATS added')
    #print(len(df_stats), tile_name)
    return df_stats

def read_country_per_tile(tile_to_read,country_name, gpkg_all_tiles_name,today_day,empty_df_stats,per_tile_output_path):

    try:
        iso3id = pycountry.countries.get(name=country_name).alpha_3  # iso3 code country
      
        tiles_country_path=r'/eos/jeodpp/data/projects/3D-BIG/data/processed_data/buildings_footprints/country-tiles/'
        tile_gpkg = gpd.read_file(os.path.join(tiles_country_path, country_name, tile_to_read),engine='pyogrio')
      
        xmin, ymin, xmax, ymax = tile_gpkg.total_bounds # Define the bounding box (extent) you are interested in TILE
        bbox = (xmin, ymin, xmax, ymax)
        logger.info(f"LOADING: {tile_to_read}")  # will not print anything
        eub_list_data = glob.glob(os.path.join(eub_fld, f'v0_1-{iso3id}*.gpkg'))

        if eub_list_data: #and country_name!='malta':
            eub_data = glob.glob(os.path.join(eub_fld, f'v0_1-{iso3id}*.gpkg'))[0]
            eub_tile_orig = gpd.read_file(eub_data, bbox=bbox, engine='pyogrio')
            eub_tile_orig.drop_duplicates(subset='geometry',keep='first',inplace=True)
        else:
            geometry_column = 'geometry'
            columns_eub = ['eub-height', 'eub-age', 'eub-type', 'eub-area', 'source']
            eub_tile_orig = gpd.GeoDataFrame(columns=columns_eub, geometry=[], crs='EPSG:3035')
            eub_tile_orig[geometry_column] = None
            eub_tile_orig.set_geometry(geometry_column)
            print(country_name)

        if country_name in countries_EUBOSM and not eub_tile_orig.empty:
            eub_tile_read = eub_tile_orig[~eub_tile_orig['id_source'].str.contains('latest',case=False, na=False)].copy() # keep only cadastral data
            print('1')
        elif country_name in countries_onlyOSM or eub_tile_orig.empty:
            geometry_column='geometry'
            columns_eub = ['eub-height','eub-age','eub-type','eub-area','source']
            eub_tile_read = gpd.GeoDataFrame(columns=columns_eub,geometry=[],crs='EPSG:3035')
            eub_tile_read[geometry_column]=None
            eub_tile_read.set_geometry(geometry_column)
            print(2)
        else:
            eub_tile_read = eub_tile_orig
            print('3')


        if not eub_tile_read.empty:
            #eub_tile_clipped = eub_tile_read.clip
            eub_tile_clipped = eub_tile_read[eub_tile_read.intersects(tile_gpkg.geometry.unary_union)]
            logger.info(f"EUB clipping: {tile_to_read}")  # will not print anything
        # valid_data and msb_layer_check ARE FUNCTIONS
            eub_tile = valid_data(eub_tile_clipped, 'eub')
            eub_info = info_gdf(eub_tile)
            logger.info(f"EUB clipping: {tile_to_read}")  # will not print anything
        else:
            logger.info(f"EUB empty: {tile_to_read}")  # will not print anything
            eub_tile = eub_tile_read
            eub_info = info_gdf(eub_tile)

        ###### read OSM data #######################################################
        osm_data = os.path.join(osm_dir, f'{country_name}-osm.fgb')
        osm_tile_read = gpd.read_file(osm_data, bbox=bbox,engine='pyogrio')

        if not osm_tile_read.empty:
            osm_tile_valid = valid_data(osm_tile_read, 'osm')  # check osm dataset
            osm_tile_clipped_step1 = osm_tile_valid[osm_tile_valid.intersects(tile_gpkg.geometry.unary_union)]
        # Filter out features that do not overlap with the tile
            osm_tile = osm_tile_clipped_step1[osm_tile_clipped_step1.intersects(tile_gpkg.geometry.iloc[0])].copy()
            osm_info = info_gdf(osm_tile)
            logger.info(f"OSM clipping: {tile_to_read}")  # will not print anything
        else:
            logger.info(f"OSM empty: {tile_to_read}")  # will not print anything
            osm_tile = osm_tile_read
            osm_info = info_gdf(osm_tile)

        ######### MSB data ################
        msb_data = os.path.join(msb_fld, f'msb2024-{country_name.casefold()}.gpkg')
        msb_tile_read = gpd.read_file(msb_data, bbox=bbox,engine='pyogrio')

        if not msb_tile_read.empty:
            msb_tile_valid = valid_data(msb_tile_read, 'msb')  # check geometries and make them
            msb_tile_clipped = msb_tile_valid[msb_tile_valid.intersects(tile_gpkg.geometry.unary_union)]
            #msb_tile_clipped=msb_tile_valid.clip(tile_gpkg)
            logger.info(f"MSB clipping: {tile_to_read}")  # will not print anything
            msb_tile = msb_layer_check(msb_tile_clipped)  # check duplicates
            msb_info = info_gdf(msb_tile)
        else:
            logger.info(f"MSB empty: {tile_to_read}")  # will not print anything
            msb_tile = msb_tile_read
            msb_info = info_gdf(msb_tile)

        check_crs_equivalence(eub_tile, osm_tile, msb_tile)
        logger.info(f" Original country dataset info:\n EUB {eub_info} \n OSM {osm_info} \n MSB {msb_info}")

        osm_clean_tile, eub_clean_tile, msb_clean_tile = None, None, None
        merge_eubosm = None

        if eub_tile.empty and osm_tile.empty and msb_tile.empty:
            eub_clean_tile = eub_tile.copy()
            osm_clean_tile = osm_tile.copy()
            msb_clean_tile = msb_tile.copy()

        if not eub_tile.empty or not osm_tile.empty:
            logger.info(f'EUB is not empty {tile_to_read}')
            osm_clean_tile, eub_clean_tile, merge_eubosm = eub_osm_join(eub_tile, osm_tile)
       

        if not msb_tile.empty:
            logger.info(f"MSB is not empty {tile_to_read}")
            msb_clean_tile, merge_eubosmsb = clean_msbstep(msb_tile, merge_eubosm)
           
        else:
            msb_clean_tile = msb_tile.copy()
            merge_eubosmsb = merge_eubosm.copy()
            merge_eubosmsb['msb-height']= pd.Series(dtype='float')
            logger.info(f"MSB is EMPTY {tile_to_read}")
        logger.info(f"{tile_to_read.split('-')[0].upper()} TILES saved")
    except Exception as e:
       
        logger.info(f'Error {e} - Working on {tile_to_read}')
        return None, None, None, None, None, None, None, tile_to_read, None

    if merge_eubosmsb is not None:
        logger.info('Merge of EUB-OSM-MSB is not empty')
      
        merge_eubosmsb.drop(columns=['centroid', 'saveindex', 'validity', 'type'], axis=1, errors='ignore', inplace=True)
    logger.info(f"Tile {tile_to_read} - DONE")

    if osm_clean_tile is not None and not osm_clean_tile.empty:
        osm_clean_tile = osm_clean_tile.copy()
        osm_clean_tile.drop(columns=['osm_index', 'area_joined', 'ratio_osm'], axis=1, errors='ignore', inplace=True)

    if msb_clean_tile is not None and not msb_clean_tile.empty:
        msb_clean_tile = msb_clean_tile.copy()
        msb_clean_tile.drop(columns=['save_index'], axis=1, errors='ignore', inplace=True)

    logger.info(f'Has Entered Post-Processing after process {tile_to_read}')
    tile_name = tile_to_read.split('.')[0]

    logger.info(f"Added stats for {tile_to_read}")

   
    calculate_tile_basic_stats(empty_df_stats, tile_name, eub_tile, eub_clean_tile, 'EUB' )
    calculate_tile_basic_stats(empty_df_stats, tile_name, osm_tile, osm_clean_tile, 'OSM' )
    calculate_tile_basic_stats(empty_df_stats, tile_name, msb_tile, msb_clean_tile, 'MSB' )


    df_stats = stat_tile_country(empty_df_stats, tile_name, country_name)
    df_stats.to_excel(os.path.join(output_dir,'dbsm_stats_all_countries_'+tile_to_read+today_day+'.xlsx'))

    if merge_eubosmsb is not None or not merge_eubosmsb.empty:
        merge_eubosmsb.drop(columns=['osm_index', 'area_joined', 'ratio_osm'], axis=1, errors='ignore', inplace=True)
        joined_nuts = gpd.sjoin(merge_eubosmsb, nuts_gpkg, how='left', predicate='within')
        joined_nuts.drop(columns=['centroid', 'type', 'validity', 'height', 'confidence', 'saveindex', 'index_right'],
                           axis=1, errors='ignore', inplace=True)
        joined_nuts['area m2'] = round(joined_nuts.geometry.area,2)
        # Reorder fields according to the specified order
        fields_order = ['geometry', 'source', 'eub-height', 'eub-age','eub-type', 'osm-building', 'osm-levels',
                        'osm-roof-shape','osm-height', 'osm-date', 'msb_height', 'NUTS_ID', 'area m2']
        # Define the expected data types for the missing columns
        column_data_types = {
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
            'area m2': float,
            'NUTS_ID': str,
        }

        for col, dtype in column_data_types.items():
            if col in joined_nuts.columns:
                joined_nuts[col] = joined_nuts[col].astype(dtype)
            else:
                joined_nuts[col] = pd.Series([np.nan]*len(joined_nuts),dtype=dtype)


        joined_nuts = joined_nuts[fields_order]

    return osm_clean_tile, eub_clean_tile, msb_clean_tile, merge_eubosmsb, eub_tile, osm_tile, msb_tile, tile_to_read, df_stats



def write_results(result,country_name,today_day,all_dummy):
    """write the final product merge and add the nuts id
    clean the geodataframes

       Args:
           result: the output from the multiprocessing running the  function read country per tile
           country_name,
           today_day,
           all_dummy: empty to store the merged

       Returns:
           _type_: _description_
       """
    osm_clean_tile, eub_clean_tile, msb_clean_tile, merge_eubosmsb, eub_tile, osm_tile, msb_tile, tile_to_read, df_stats = result

    if osm_clean_tile is not None and not osm_clean_tile.empty:
        osm_dummy.append(osm_clean_tile)
    else:
        pass


    if msb_clean_tile is not None and not msb_clean_tile.empty:
        msb_dummy.append(msb_clean_tile)
    else:
        pass

    if eub_clean_tile is not None and not eub_clean_tile.empty:
        eub_dummy.append(eub_clean_tile)
    else:
        pass

    if merge_eubosmsb is not None and not merge_eubosmsb.empty:
        logger.info('merge_eubosmsb is not NONE')
        merge_eubosmsb['area'] = merge_eubosmsb.geometry.area
        #merge_total_area = round(merge_eubosmsb['area'].sum() / 1e6, 3)
        logger.info('total eubosmsb computed')
        merge_eubosmsb.drop(columns=['osm_index', 'area_joined', 'ratio_osm'], axis=1, errors='ignore',inplace=True)
        all_dummy.append(merge_eubosmsb)

    logger.info(f'{result[7]} processed')

    return all_dummy


def joblib_loop(files_tiles, country_name):
    results = Parallel(n_jobs=20, verbose = 5)(delayed(read_country_per_tile)(tile_to_read, country_name, gpkg_all_tiles_name, today_day, empty_df_stats,per_tile_output_path)
                                               for tile_to_read in files_tiles)
    return results


if __name__ == "__main__":
    today_day = date.today().strftime("%Y-%m-%d")
    country_names = [f.split('-')[0] for f in filter(lambda f: f.endswith('.fgb'), os.listdir(osm_dir))]
   

    for country_name in country_names:

        files_tiles = os.listdir(os.path.join(tiles_country_path, country_name))

       
        gpkg_all_tiles_name = f"dbsm-v2-{country_name}-tiles-no-shifting_{today_day}.gpkg"
        empty_df_stats = create_df_stat(files_tiles)  ### create a df for the stats

        results = joblib_loop(files_tiles, country_name)
     

        print(country_name)

        eub_dummy = []
        osm_dummy = []
        msb_dummy = []
        all_dummy = []

        for result in results:
            print(result[7])
            merged_output = write_results(result, country_name, today_day, all_dummy)


        merge_all = pd.concat(merged_output)
        merge_all.drop(columns=['osm_index', 'area_joined', 'ratio_osm','eub-area','osm-area'], axis=1, errors='ignore', inplace=True)
        merge_all['geometry'] = merge_all.normalize()
        merge_all.drop_duplicates(subset='geometry', inplace=True)
        merge_all['fid'] = range(len(merge_all))

     
        geopackage_filename = f'{version_dbsm}-{country_name.casefold()}-{today_day}-R2025.gpkg'
        merge_all.to_file(os.path.join(output_dir, geopackage_filename), driver='GPKG')
        logger.info(f'saved merged gpkg for country {country_name.upper()}')






