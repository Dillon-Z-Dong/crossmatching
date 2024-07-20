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

from crossmatching_utils import crossmatch_skycoords, const_or_variable_radius_match_function

'''
Functions for batch crossmatching against the neural net source classifications in the PS1-STRM catalog

See Beck et al. 2021 (https://academic.oup.com/mnras/article/500/2/1633/5899759) for catalog details
'''


def read_hdf5(file, requested_cols, required_cols = ['raMean', 'decMean','class']):
	columns_to_read = list(set(requested_cols + required_cols))
	return pd.read_hdf(file, key='data', columns=columns_to_read)  



def read_hdf5_list(filenames, datadir, requested_cols, input_match_cat, match_radius, n_matches, n_threads, verbose, selection_function, **selection_function_kwargs):
	''' The hot loop for STRM cross matching'''

	match_tables = []
	match_indices = []
	match_distances = []

	strm_cat_start = time.perf_counter()
	with ThreadPoolExecutor(max_workers=n_threads) as executor:  
			# Map the read_hdf5 function to the list of files
			results = executor.map(lambda file: read_hdf5(file, requested_cols), [os.path.join(datadir,x) for x in filenames])

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


def partition_files(nearest_tiles_names, nearest_tiles_sizes, total_file_size):
	''' Partitions the hdf5 files into chunks using the greedy algorithm'''

	# Combine indices, names, and sizes into a list of tuples
	files = list(zip(np.arange(len(nearest_tiles_names)), nearest_tiles_names, nearest_tiles_sizes))
	
	# Create a dictionary to track the files and their original indices
	file_dict = {}
	for idx, name, size in files:
		if name not in file_dict:
			file_dict[name] = {'size': size, 'indices': [idx]}
		else:
			file_dict[name]['indices'].append(idx)

	# Convert the dictionary back to a list of tuples (name, size, original indices)
	file_list = [(name, data['size'], data['indices']) for name, data in file_dict.items()]
	
	# Sort by size in descending order
	file_list.sort(key=lambda x: x[1], reverse=True)
	
	# Initialize chunks
	chunks = []
	current_chunk = []
	current_chunk_size = 0

	# Partition files into chunks
	for name, size, indices in file_list:
		if current_chunk_size + size > total_file_size:
			# If adding the next file exceeds the limit, start a new chunk
			chunks.append(current_chunk)
			current_chunk = []
			current_chunk_size = 0
		
		# Add file to the current chunk
		current_chunk.append((name, size, indices))
		current_chunk_size += size

	# Add the last chunk if not empty
	if current_chunk:
		chunks.append(current_chunk)

	return chunks


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

def STRM_crossmatch(
	input_cat: SkyCoord,
	requested_cols: list,
	n_matches = 1,
	match_radius = None,
	max_chunk_size = 1*u.GB,
	n_threads = 10,
	verbose = False, 
	datadir = '/lustre/aoc/sciops/ddong/Catalogs/PS1_STRM/data/output_chunks/hdf5_tables/',
	metadata_table = '/lustre/aoc/sciops/ddong/Catalogs/PS1_STRM/data/STRM_metadata.csv',
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

	- n_matches: int (number of nearest neighbor matches to return. e.g., if n_matches = 3, this function will output the 1st, 2nd, and 3rd nearest neighbor for all points so long as they are within match_radius)
	- match_radius: astropy Quantity object corresonding to an anglular separation, e.g. match_radius = 5*u.arcsec
		- if match_radius is not specified, will return the closest n_matches matches regardless of separation

	- max_chunk_size (depends on RAM, must have unit. Larger values can be faster depending on I/O efficiency.)
	- n_threads (a few more than the number of CPU cores you're willing to use)
	- verbose (print stats as cross match is happening)
	- selection function (optional function that takes as input a STRM pandas table and applies a selection function, returning a filtered pandas table)
		- must also input *selection_function_args for selection function

	- datadir: the directory of the hdf5 files 
	- metadata_table: file in datadir containing the filename, n_rows, filesize in MB, central RA, and central DEC of the chunks
	- metadata_skycoord: pickle of skycoord containing the metadata points

	Output:
	- matched_index_list: indices (of input_coords) that were successfully matched within match_radius. If match_radius is None, returns all indices.
	- matched_distance_list: distances of the nth nearest neighbor within match_radius
	- matched_tab_list: STRM properties of matches
		
	'''
	if verbose:
		fstart = time.perf_counter()

	# Load metadata table
	metadata_table = pd.read_csv(os.path.join(datadir,metadata_table))
	
	# Create skycoord from metadata table
	metadata_cat = SkyCoord(ra = np.array(metadata_table['central_ra']), dec = np.array(metadata_table['central_dec']), unit = (u.deg,u.deg))

	# Make sure requested cols are ok
	possible_cols = ['objID','uniquePspsOBid','raMean','decMean','l','b','class','prob_Galaxy','prob_Star','prob_QSO',\
	'extrapolation_Class','cellDistance_Class','cellID_Class','z_phot','z_photErr','z_phot0',\
	'extrapolation_Photoz','cellDistance_Photoz','cellID_Photoz']

	col_diff = set(requested_cols) - set(possible_cols) 
	if len(col_diff) > 0:
		raise Exception(f'\n\nRequested columns {col_diff} do not exist in STRM!\n\nColumn options: {possible_cols}')

	# Identify relevant tiles for matchable points, excluding points outside of Panstarrs coverage
	
	not_matchable = np.where(input_cat.dec.deg < -32.5)
	if len(not_matchable[0]) > 0:
		print(f'Warning: {len(not_matchable)} out of input {len(input_cat)} coordinates ({len(not_matchable)/len(input_cat)*100}%) are outside of the Panstarrs coverage (dec < -32.5 deg)')
		matchable_cat = input_cat[~not_matchable]
	else:
		matchable_cat = input_cat
	
	nearest_tiles_index, d2d, _ = matchable_cat.match_to_catalog_sky(metadata_cat)
	nearest_tiles_names = metadata_table.iloc[nearest_tiles_index].filename
	nearest_tiles_sizes = metadata_table.iloc[nearest_tiles_index].filesize_MB
	chunks = partition_files(nearest_tiles_names, nearest_tiles_sizes, max_chunk_size.to(u.MB).value)
	
	if verbose:
		print(f'Partitioned {len(set(nearest_tiles_index)):,} nearest tiles to the {len(matchable_cat)} matchable input coordinates into {len(chunks)} chunks of size < {max_chunk_size} in {(time.perf_counter()-fstart):.2f} seconds')

	for i, chunk in enumerate(chunks):
		if verbose:
			loop_start = time.perf_counter()

		filenames, sizes, input_indices = [list(x) for x in zip(*chunk)]
		input_indices = np.array(list(itertools.chain.from_iterable(input_indices))) #Flattening the list of lists takes a surprising amount of time. In the test case it was 8.1s using sum compared to 2.4s for itertools.chain
		input_match_cat = matchable_cat[input_indices]

		if verbose:
			print('----------------------------------------------------------------------------------------------------------------------------------------------')
			print(f'\nProcessing chunk {i+1}/{len(chunks)} of size {sum(sizes):.1f} MB containing {len(filenames)} files and {len(input_match_cat)} points to match')

		match_tables = read_hdf5_list(
			filenames = filenames,
			datadir = datadir,
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

		if verbose:
			fend = time.perf_counter()
			print(f'Total runtime: {(fend-fstart):.2f} seconds')

	return matched_tab_list



if __name__ == '__main__':
	
	#Test
	nrand = 10_000_000

	rand_ra_start = 20.0
	rand_dec_start = 15.0

	rand_ra_end = rand_ra_start + 5
	rand_dec_end = rand_dec_start + 5

	n_matches = 5
	match_radius = 1*u.arcsec
	verbose = True

	print('\n------------------------  Test  ------------------------')
	print(f'\nGenerating input skycoord of length n = {nrand:,} for:')
	print(f'Random RAs between {rand_ra_start} - {rand_ra_end}')
	print(f'Random DECs between {rand_dec_start} - {rand_dec_end}\n')

	rand_ra = np.random.uniform(rand_ra_start,rand_ra_end,nrand)
	rand_dec = np.random.uniform(rand_dec_start,rand_dec_end,nrand)
	rand_coords = SkyCoord(ra = rand_ra, dec = rand_dec, unit = (u.deg,u.deg))

	print('Doing crossmatch')
	run_start = time.time()
	out = STRM_crossmatch(rand_coords, verbose = verbose, match_radius = match_radius, n_matches = n_matches, requested_cols = ['objID','z_phot','z_photErr','z_phot0'], selection_function = STRM_type_filter, objtype = 'galaxy')
	run_end = time.time()
	
	

