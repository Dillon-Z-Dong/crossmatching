import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
import os
import pandas as pd
import time


########### Custom match functions

def normalized_match_function(idx: np.ndarray, d2d: np.ndarray, **match_function_kwargs):
	'''use for e.g. galaxy cross matching where you want to normalize by a value like the half light radius'''
	raise NotImplementedError('normalized match function is not yet implemented')


def const_or_variable_radius_match_function(idx: np.ndarray, d2d: np.ndarray, **match_function_kwargs):
	'''
	kwargs must include 'match_radius', which may be a single quantity or an array of the same size as the match catalog
	'''
	match_radius = match_function_kwargs['match_radius']

	ndim = np.ndim(match_radius)
	if ndim == 0: #match radius is a scalar
		matches = np.where(d2d < match_radius)

	elif ndim == 1: #match radius is a 1D array
		matches = np.where(d2d < match_radius[idx])

	return matches

def null_match_function(idx: np.ndarray, d2d: np.ndarray, **match_function_kwargs):
	'''
	Use this function if you don't want to do any selection on matches (i.e. you want to return the nearest neighbor regardless of distance)
	'''
	return np.arange(len(d2d))

def sort_col_order(tab: pd.DataFrame, desired_order = ['input_ra_deg', 'input_dec_deg', 'distance_arcsec', 'position_angle', 'catalog_ra', 'catalog_dec']):
	'''
	reorders the columns of dataframe so that the first columns are those in desired_order, and the other columns are in their original order
	'''

	remaining_columns = [col for col in tab.columns if col not in desired_order]
	new_order = desired_order + remaining_columns
	tab = tab.reindex(columns=new_order)

	return tab


########  General utils

def initialize_match_tab(match_tab=None, columns=['input_ra_deg', 'input_dec_deg', 'distance_arcsec', 'position_angle', 'catalog_ra_deg', 'catalog_dec_deg'], filler=np.nan):
	'''Creates a pandas dataframe with the requested columns, and filled with filler. If match_tab is provided, adds the initialized columns if not already present.'''
	
	if match_tab is None:
		# Create a new DataFrame if match_tab is not provided
		match_tab = pd.DataFrame(columns=columns)
	else:
		# Add missing columns with filler values
		for col in columns:
			if col not in match_tab.columns:
				match_tab[col] = filler
	
	match_tab = sort_col_order(match_cat)
	return match_tab


def crossmatch_skycoords(input_cat: SkyCoord, match_tab: pd.DataFrame, \
	n_matches: int = 1, match_cat: SkyCoord = None, hierarchical_optimization:bool = None, verbose:bool = False, match_function = null_match_function, **match_function_kwargs):
	'''General crossmatching between input_cat (your points) and either a user provided match_cat (catalog points) or one generated from match_tab
		- if using match_tab, the columns 'catalog_ra_deg', and 'catalog_dec_deg' must exist
	Input your favorite match function (e.g., const_or_variable_radius_match_function, or something fancier)
	Does a blind row selection on match_tab (catalog other info) and returns a list of match_tab rows (length n_matches) for each k-th neighbor
	match_tab must have 
	'''

	# Decide whether or not to use the hierarchical optimization for various match functions if not specified
	if hierarchical_optimization is None:

		hierarchical_optimization_mapping = {
			null_match_function: True,
			const_or_variable_radius_match_function: True,
			normalized_match_function: False
		}

		# Set hierarchical_optimization based on the match_function
		hierarchical_optimization = hierarchical_optimization_mapping.get(match_function, False)

	# Create match_cat from match_tab if one does not exist
	if match_cat is None:
		match_cat = SkyCoord(ra = np.array(match_tab['catalog_ra_deg']), dec = np.array(match_tab['catalog_dec_deg']))

	# Determine whether we should save the kd tree. Currently saving if we're making multiple searches on a tree of length > 1e6.
	if n_matches > 1 and len(match_cat) > 1e6:
		skdt = 'tree'
	else:
		skdt = False

	# Initialize the output array
	match_tables = []

	# Run the cross match

	if hierarchical_optimization:
		latest_matched_indices = np.arange(len(input_cat))

	if verbose:
		kwargs_str = ', '.join(f'{k} = {v}' for k, v in match_function_kwargs.items())
		print('----------------------')
		print(f'Running crossmatch_skycoord with with match function {match_function.__name__}, kwargs {kwargs_str}, and {hierarchical_optimization = }')
		print('----------------------')

	for k in range(1,n_matches+1):

		input_cat_k = input_cat[latest_matched_indices]
		if verbose:
			print(f'Crossmatching {len(input_cat_k)} input points against {len(match_cat)} catalog points looking for the {k = }-th nearest neighbor')
			start = time.perf_counter()

		idx,d2d,_ = match_coordinates_sky(input_cat_k, match_cat, nthneighbor=k, storekdtree = skdt) 
		matches = match_function(idx = idx, d2d = d2d, **match_function_kwargs)

		if hierarchical_optimization:
			latest_matched_indices = latest_matched_indices[matches]
			latest_match_tab = match_tab.iloc[idx[matches]].copy()
			
		else:
			latest_match_tab = match_tab.copy()

		latest_match_tab['input_ra_deg'] = input_cat_k[matches].ra.deg
		latest_match_tab['input_dec_deg'] = input_cat_k[matches].dec.deg
		latest_match_tab['distance_arcsec'] = d2d[matches].to(u.arcsec).value
		latest_match_tab['position_angle_deg'] = input_cat_k[matches].position_angle(match_cat[idx[matches]]).deg

		# Append results to output
		match_tables.append(latest_match_tab)

		if verbose:
			loop_runtime = time.perf_counter() - start
			print(f'Found {len(matches[0])} matches in {loop_runtime:.4f} seconds\n') 

	if verbose:
		print('----------------------')

	return match_tables