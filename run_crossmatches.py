import sys
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import time
import pandas as pd  

# Add the directory to the system path
sys.path.append('/export/home/rabbit2/VLASS_tools/crossmatching/Panstarrs_STRM')

# Import the STRM_crossmatch function
from STRM_crossmatch import STRM_crossmatch

def run_crossmatches(input_coords, match_radius=10*u.arcsec, n_matches=1):
    results = {}

    for match_cat_name in ['STRM_base', 'STRM_WISE']:
        print(f'\nStarting crossmatch with {match_cat_name}...')

        start_time = time.time()
        result_dfs = STRM_crossmatch(
            input_cat=input_coords,
            requested_cols=['objID', 'z_phot', 'z_photErr', 'z_phot0'],
            match_cat_name=match_cat_name,
            match_radius=match_radius,
            n_matches=n_matches,
            selection_function=STRM_type_filter,  # Assuming this is defined elsewhere
            objtype='galaxy',
            verbose=True
        )
        end_time = time.time()

        results[match_cat_name] = result_dfs
        print(f'Crossmatch with {match_cat_name} completed in {end_time - start_time:.2f} seconds.')

    return results

if __name__ == '__main__':
    # Test
    for nrand in [1_000_000]:
        rand_ra_start = 20.0
        rand_dec_start = -15.0

        rand_ra_end = rand_ra_start + 5
        rand_dec_end = rand_dec_start + 5

        print('\n------------------------  Test  ------------------------')
        print(f'\nGenerating input skycoord of length n = {nrand:,} for:')
        print(f'Random RAs between {rand_ra_start} and {rand_ra_end} deg')
        print(f'Random DECs between {rand_dec_start} and {rand_dec_end} deg\n')

        rand_ra = np.random.uniform(rand_ra_start, rand_ra_end, nrand)
        rand_dec = np.random.uniform(rand_dec_start, rand_dec_end, nrand)
        rand_coords = SkyCoord(ra=rand_ra, dec=rand_dec, unit=(u.deg, u.deg))

        print('Running crossmatches...')
        results = run_crossmatches(
            input_coords=rand_coords
        )

        # Each value in the results dictionary is a list of pandas DataFrames
        for match_cat_name, dfs in results.items():
            print(f'\nResults for {match_cat_name}:')
            for i, df in enumerate(dfs):
                print(f'DataFrame {i+1}:')
                print(df.head())  # Display the first few rows of each DataFrame

