import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from typing import Literal
import pickle
from collections import defaultdict, Counter
import itertools
from concurrent.futures import ThreadPoolExecutor
import time
import itertools
from collections import defaultdict
from itertools import product

from crossmatching_utils import crossmatch_skycoords, const_or_variable_radius_match_function, sort_col_order

#Testing / profiling
import cProfile
import pstats
import io


'''
Functions for batch crossmatching against the neural net source classifications in the PS1-STRM catalog

See Beck et al. 2021 (https://academic.oup.com/mnras/article/500/2/1633/5899759) for catalog details
'''


def read_hdf5(file, requested_cols, required_cols = ['raMean', 'decMean','class']):
	columns_to_read = list(set(requested_cols + required_cols))
	return pd.read_hdf(file, key='data', columns=columns_to_read)  



def read_hdf5_list(filenames, requested_cols, input_match_cat, match_radius, n_matches, n_threads, verbose, selection_function, **selection_function_kwargs):
	''' The hot loop for STRM cross matching'''

	match_tables = []
	match_indices = []
	match_distances = []

	strm_cat_start = time.perf_counter()
	with ThreadPoolExecutor(max_workers=n_threads) as executor:  
			# Map the read_hdf5 function to the list of files
			results = executor.map(lambda file: read_hdf5(file, requested_cols), filenames)

			# Collect the results into a list
			dataframes = list(results)

			# Concatenate all DataFrames into a single DataFrame
			combined_df = pd.concat(dataframes, ignore_index=True)

			# Filter combined_df to include only requested objects
			if selection_function is not None:
				if verbose:
					kwargs_str = ', '.join(f'{k} = {v}' for k, v in selection_function_kwargs.items())
					print(f'\nApplied selection function {selection_function.__name__} with kwargs {kwargs_str}')
				combined_df = selection_function(combined_df, verbose = verbose, **selection_function_kwargs)
				

	# Create SkyCoord object out of filtered dataframe
	strm_match_cat = SkyCoord(ra = np.array(combined_df['raMean'])*u.deg, dec = np.array(combined_df['decMean'])*u.deg)
	

	if verbose:
		strm_cat_end = time.perf_counter()
		print(f'STRM chunk data read + catalog creation time: {(strm_cat_end - strm_cat_start):.2f} seconds')

	match_tables = crossmatch_skycoords(
		input_cat = input_match_cat, 
		match_cat = strm_match_cat, 
		match_tab = combined_df, 
		n_matches = n_matches,
		hierarchical_optimization = True,
		match_function = const_or_variable_radius_match_function,
		match_radius = match_radius,
		verbose = verbose
		)

	if verbose:
		match_end = time.perf_counter()
		print(f'Matched catalogs in {(match_end - strm_cat_end):.4f} seconds')

	return match_tables




def STRM_type_filter(combined_df, objtype: Literal['galaxy','star','qso','unsure','all'], verbose = False):
	'''
	Input:
	- pandas table assembled from STRM hdf5 files
	- objtype: type of object to cross match (must be 'galaxy','star','qso','unsure', or 'all'). Unsure is a STRM category. All will cross match with any objtype.
	
	Output:
	- table with only the requested objtype
	'''

	if verbose:
		print(f'\nTotal number of STRM objects in chunk before filtering: {len(combined_df)}\n')
		class_count = Counter(combined_df['class'])
		for obj_class, obj_count in sorted(class_count.items(), key=lambda item: item[1], reverse=True):
			print(f'{obj_class}: {obj_count} --- {(obj_count/len(combined_df)*100):.2f}%')

	if objtype == 'all':
		if verbose:
			print(f'\nRequested objtype = "all" and no selection function provided: not filtering table.\n')
	else:
		combined_df = combined_df[combined_df['class'] == objtype.upper()]
		if verbose:
			print(f'\nMatching only objects of type {objtype.upper()}\n')

	if verbose:
		if objtype != 'all':
			print(f'Total number of STRM objects in chunk after filtering: {len(combined_df)}')
			class_count = Counter(combined_df['class'])
			for obj_class, obj_count in sorted(class_count.items(), key=lambda item: item[1], reverse=True):
				print(f'{obj_class}: {obj_count:.3e} --- {(obj_count/len(combined_df)*100):.2f}%')

	return combined_df



def fast_tile_partition2(matchable_cat,metadata_table, max_chunk_size):
	round_ra = np.floor(matchable_cat.ra.deg) + 0.5
	round_dec = np.floor(matchable_cat.dec.deg) + 0.5
	all_pairs = list(product(set(round_ra), set(round_dec)))

	nearest_tiles = pd.DataFrame({'ra':round_ra, 'dec':round_dec})

	partitions = nearest_tiles.reset_index(drop = True).groupby(['ra','dec']).agg(
			indices=('ra', lambda x: list(x.index))
		).reset_index()

	# Hacky way to determine the naming convention
	if metadata_table.loc[0,'filename'].endswith('_table.h5'):
		suffix = '_table.h5'
	elif metadata_table.loc[0,'filename'].endswith('.h5'):
		suffix = '.h5'

	# Construct the filename column in partitions DataFrame
	partitions['filename'] = partitions.apply(lambda row: f'chunk_ra_{row["ra"]}_dec_{row["dec"]}{suffix}', axis=1)

	# Merge partitions with metadata_table on the filename column
	merged_df = pd.merge(partitions, metadata_table, on='filename', how='left').sort_values('filesize_MB')

	# Partition the sorted DataFrame into chunks
	chunks = []
	current_filenames, current_indices = [], []
	current_chunk_size = 0

	for index, row in merged_df.iterrows():
		if current_chunk_size + row['filesize_MB'] > max_chunk_size:
			# If adding the next file exceeds the limit, start a new chunk
			chunks.append((current_filenames,current_chunk_size,np.array(current_indices)))
			current_filenames, current_indices = [], []
			current_chunk_size = 0

		# Add file to the current chunk
		current_filenames.append(row['filename'])
		current_indices += row['indices']
		current_chunk_size += row['filesize_MB']


	# Add the last chunk if not empty
	if current_filenames:
		chunks.append((current_filenames,current_chunk_size,np.array(current_indices)))

	return chunks


def STRM_out_of_field_filter(input_cat):
	not_matchable = np.where(input_cat.dec.deg < -32.5)
	if len(not_matchable[0]) > 0:
		print(f'Warning: {len(not_matchable)} out of input {len(input_cat)} coordinates ({len(not_matchable)/len(input_cat)*100}%) are outside of the Panstarrs coverage (dec < -32.5 deg)')
		matchable_cat = input_cat[~not_matchable]
	else:
		matchable_cat = input_cat
	return matchable_cat



def get_metadata(match_cat_name, requested_cols):

	if match_cat_name == 'STRM_base':
		metadata_table_name = 'STRM_base_metadata.csv'
		datadir = '/lustre/aoc/sciops/ddong/Catalogs/PS1_STRM/data/output_chunks/hdf5_tables/'
		possible_cols = [
    'objID', 'uniquePspsOBid', 'raMean', 'decMean', 'l', 'b', 'class', 'prob_Galaxy', 'prob_Star', 'prob_QSO',
    'extrapolation_Class', 'cellDistance_Class', 'cellID_Class', 'z_phot', 'z_photErr', 'z_phot0',
    'extrapolation_Photoz', 'cellDistance_Photoz', 'cellID_Photoz']


	elif match_cat_name == 'STRM_WISE':
		metadata_table_name = 'STRM_WISE_metadata.csv'
		datadir = '/lustre/aoc/sciops/ddong/Catalogs/STRM_WISE/data/output_chunks/'
		possible_cols = [
    'objID', 'raMean', 'raMeanErr', 'decMean', 'decMeanErr', 'l', 'b', 'distance_Deg', 'sqrErr_Arcsec', 
    'BayesFactor', 'cntr', 'HtmID', 'ra', 'dec', 'sigra', 'sigdec', 'sigradec', 'cc_flags', 'ext_flg', 
    'ph_qual', 'moon_lev', 'w1mpro', 'w1sigmpro', 'w1rchi2', 'w1sat', 'w1mag', 'w1sigm', 'w1flg', 
    'w1mag_1', 'w1sigm_1', 'w1flg_1', 'w1mag_4', 'w1sigm_4', 'w1flg_4', 'w1mag_7', 'w1sigm_7', 'w1flg_7', 
    'w2mpro', 'w2sigmpro', 'w2rchi2', 'w2sat', 'w2mag', 'w2sigm', 'w2flg', 'w2mag_1', 'w2sigm_1', 'w2flg_1', 
    'w2mag_4', 'w2sigm_4', 'w2flg_4', 'w2mag_7', 'w2sigm_7', 'w2flg_7', 'w3mpro', 'w3sigmpro', 'w3rchi2', 
    'w3sat', 'w3mag', 'w3sigm', 'w3flg', 'w3mag_1', 'w3sigm_1', 'w3flg_1', 'w3mag_4', 'w3sigm_4', 'w3flg_4', 
    'w3mag_7', 'w3sigm_7', 'w3flg_7', 'w4mpro', 'w4sigmpro', 'w4rchi2', 'w4sat', 'w4mag', 'w4sigm', 'w4flg', 
    'w4mag_1', 'w4sigm_1', 'w4flg_1', 'w4mag_4', 'w4sigm_4', 'w4flg_4', 'w4mag_7', 'w4sigm_7', 'w4flg_7', 
    'gFPSFMag', 'gFPSFMagErr', 'gFKronMag', 'gFKronMagErr', 'gFApMag', 'gFApMagErr', 'gFmeanMagR5', 
    'gFmeanMagR5Err', 'gFmeanMagR6', 'gFmeanMagR6Err', 'gFmeanMagR7', 'gFmeanMagR7Err', 'gnTotal', 
    'gnIncPSFFlux', 'gnIncKronFlux', 'gnIncApFlux', 'gnIncR5', 'gnIncR6', 'gnIncR7', 'gFlags', 'gE1', 'gE2', 
    'rFPSFMag', 'rFPSFMagErr', 'rFKronMag', 'rFKronMagErr', 'rFApMag', 'rFApMagErr', 'rFmeanMagR5', 
    'rFmeanMagR5Err', 'rFmeanMagR6', 'rFmeanMagR6Err', 'rFmeanMagR7', 'rFmeanMagR7Err', 'rnTotal', 
    'rnIncPSFFlux', 'rnIncKronFlux', 'rnIncApFlux', 'rnIncR5', 'rnIncR6', 'rnIncR7', 'rFlags', 'rE1', 'rE2', 
    'iFPSFMag', 'iFPSFMagErr', 'iFKronMag', 'iFKronMagErr', 'iFApMag', 'iFApMagErr', 'iFmeanMagR5', 
    'iFmeanMagR5Err', 'iFmeanMagR6', 'iFmeanMagR6Err', 'iFmeanMagR7', 'iFmeanMagR7Err', 'inTotal', 
    'inIncPSFFlux', 'inIncKronFlux', 'inIncApFlux', 'inIncR5', 'inIncR6', 'inIncR7', 'iFlags', 'iE1', 'iE2', 
    'zFPSFMag', 'zFPSFMagErr', 'zFKronMag', 'zFKronMagErr', 'zFApMag', 'zFApMagErr', 'zFmeanMagR5', 
    'zFmeanMagR5Err', 'zFmeanMagR6', 'zFmeanMagR6Err', 'zFmeanMagR7', 'zFmeanMagR7Err', 'znTotal', 
    'znIncPSFFlux', 'znIncKronFlux', 'znIncApFlux', 'znIncR5', 'znIncR6', 'znIncR7', 'zFlags', 'zE1', 'zE2', 
    'yFPSFMag', 'yFPSFMagErr', 'yFKronMag', 'yFKronMagErr', 'yFApMag', 'yFApMagErr', 'yFmeanMagR5', 
    'yFmeanMagR5Err', 'yFmeanMagR6', 'yFmeanMagR6Err', 'yFmeanMagR7', 'yFmeanMagR7Err', 'ynTotal', 
    'ynIncPSFFlux', 'ynIncKronFlux', 'ynIncApFlux', 'ynIncR5', 'ynIncR6', 'ynIncR7', 'yFlags', 'yE1', 'yE2', 
    'EBV_Planck', 'EBV_PS1', 'class', 'prob_Galaxy', 'prob_Star', 'prob_QSO', 'extrapolation_Class', 
    'cellDistance_Class', 'cellID_Class', 'z_phot', 'z_photErr', 'z_phot0', 'extrapolation_Photoz', 
    'cellDistance_Photoz', 'cellID_Photoz']


    # Make sure requested columns exist 
	col_diff = set(requested_cols) - set(possible_cols) 
	if len(col_diff) > 0:
		raise Exception(f'Requested columns {col_diff} do not exist for catalog {match_cat_name}! Column options: {possible_cols}')

	# Load metadata table
	metadata_table = pd.read_csv(metadata_table_name)

	return metadata_table, datadir


def STRM_crossmatch(
	input_cat: SkyCoord,
	requested_cols: list,
	match_cat_name = Literal['STRM_base','STRM_WISE'],
	colorder = ['class','distance_arcsec','position_angle_deg','input_ra_deg','input_dec_deg','raMean','decMean'],
	n_matches = 1,
	match_radius = None,
	max_chunk_size = 1*u.GB,
	n_threads = 10,
	verbose = False, 
	selection_function = None,
	**selection_function_kwargs
	):

	'''
	Bulk local cross matching of coords with STRM

	Required input: 

	- input_coords: astropy SkyCoord object, e.g., input_coords = SkyCoord(ra = [1,2,3], dec = [4,5,6], unit = (u.deg,u.deg))
	- requested_cols: list of colnames to output. 
		- Options:
			['objID','uniquePspsOBid','raMean','decMean','l','b','class','prob_Galaxy','prob_Star','prob_QSO',
			'extrapolation_Class','cellDistance_Class','cellID_Class','z_phot','z_photErr','z_phot0',
			'extrapolation_Photoz','cellDistance_Photoz','cellID_Photoz']

	Optional inputs:
	
	- colorder: the order you want your columns to be returned in
	- n_matches: int (number of nearest neighbor matches to return. e.g., if n_matches = 3, this function will output the 1st, 2nd, and 3rd nearest neighbor for all points so long as they are within match_radius)
	- match_radius: astropy Quantity object corresonding to an anglular separation, e.g. match_radius = 5*u.arcsec
		- if match_radius is not specified, will return the closest n_matches matches regardless of separation

	- max_chunk_size (depends on RAM, must have unit. Larger values can be faster depending on I/O efficiency.)
	- n_threads (a few more than the number of CPU cores you're willing to use)
	- verbose (print stats as cross match is happening)
	- selection function (optional function that takes as input a STRM pandas table and applies a selection function, returning a filtered pandas table)
		- must also input *selection_function_args for selection function

	- datadir: the directory of the hdf5 files 


	Output:
	- matched_index_list: indices (of input_coords) that were successfully matched within match_radius. If match_radius is None, returns all indices.
	- matched_distance_list: distances of the nth nearest neighbor within match_radius
	- matched_tab_list: STRM properties of matches
		
	'''
	if verbose:
		fstart = time.perf_counter()

	# Get metadata
	metadata_table, datadir = get_metadata(match_cat_name = match_cat_name, requested_cols = requested_cols)

	# Exclude points out of the panstarrs coverage
	matchable_cat = STRM_out_of_field_filter(input_cat)

	# Partition matchable points into chunks of size < max_chunk_size
	partition_start = time.perf_counter()
	chunks = fast_tile_partition2(matchable_cat,metadata_table, max_chunk_size.to(u.MB).value)
	partition_end = time.perf_counter()

	if verbose:
		print(f'Partitioned {len(matchable_cat)} points in to {len(chunks)} chunks of size < {max_chunk_size}')

	# Crossmatch each chunk
	for i, chunk in enumerate(chunks):
		loop_start = time.perf_counter()
		# Parse chunks
		filenames, size, input_indices = chunk
		filepaths = [os.path.join(datadir,x) for x in filenames]

		input_match_cat = matchable_cat[input_indices]

		if verbose:
			print('----------------------------------------------------------------------------------------------------------------------------------------------')
			print(f'\nProcessing chunk {i+1}/{len(chunks)} of size {size:.2f} MB containing {len(filenames)} files and {len(input_match_cat)} points to match')

		match_tables = read_hdf5_list(
			filenames = filepaths,
			requested_cols = requested_cols,
			input_match_cat = input_match_cat,
			match_radius = match_radius,
			n_matches = n_matches,
			n_threads = n_threads,
			verbose = verbose,
			selection_function = selection_function,
			**selection_function_kwargs
			)

		if verbose:
			loop_end = time.perf_counter()
			print(f'\nMatch loop runtime: {(loop_end - loop_start):.4f} seconds')
			print('----------------------------------------------------------------------------------------------------------------------------------------------')
		
		#Concatenate results
		if i == 0:
			matched_tab_list = match_tables
		else:
			matched_tab_list = [pd.concat([matched_tab_list[i], match_tables[i]]) for i in range(n_matches)]

	sorted_tab_list = [sort_col_order(tab, desired_order = colorder) for tab in matched_tab_list]

	if verbose:
		fend = time.perf_counter()
		print(f'------> Total crossmatch runtime: {(fend-fstart):.2f} seconds')
		print(f'------> Runtime per chunk: {(fend-fstart)/len(chunks):.2f} seconds')


	return sorted_tab_list



if __name__ == '__main__':
	
	#Test

	for nrand in [10_000_000]:
		rand_ra_start = 20.0
		rand_dec_start = -15.0

		rand_ra_end = rand_ra_start + 5
		rand_dec_end = rand_dec_start + 5

		n_matches = 5
		match_radius = 1*u.arcsec
		verbose = True
		max_chunk_size = 1*u.GB

		print('\n------------------------  Test  ------------------------')
		print(f'\nGenerating input skycoord of length n = {nrand:,} for:')
		print(f'Random RAs between {rand_ra_start} and {rand_ra_end} deg')
		print(f'Random DECs between {rand_dec_start} and {rand_dec_end} deg\n')

		rand_ra = np.random.uniform(rand_ra_start,rand_ra_end,nrand)
		rand_dec = np.random.uniform(rand_dec_start,rand_dec_end,nrand)
		rand_coords = SkyCoord(ra = rand_ra, dec = rand_dec, unit = (u.deg,u.deg))


	'''
	match_cat_name can be STRM_base or STRM_WISE
	'''
	