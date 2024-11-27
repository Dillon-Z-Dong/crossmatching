import h5py
import numpy as np
import pandas as pd
from pathlib import Path

# Column groupings from previous message here
COLUMN_GROUPS = {
    # Essential position and identification information
    'core': [
        'objID',          # PS1 source identifier
        'raMean',         # PS1 RA 
        'decMean',        # PS1 Dec
        'raMeanErr',      # PS1 RA error
        'decMeanErr',     # PS1 Dec error
        'l',              # Galactic longitude
        'b',              # Galactic latitude
        'distance_Deg',   # PS1-WISE separation
        'sqrErr_Arcsec',  # Combined astrometric uncertainty
        'BayesFactor',    # Cross-match likelihood ratio
        'HtmID',          # Hierarchical triangular mesh index
        'ra',             # WISE RA
        'dec',            # WISE Dec
        'sigra',          # WISE RA uncertainty
        'sigdec',         # WISE Dec uncertainty
        'sigradec'        # WISE position error correlation
    ],

    # Neural network classifications and photometric redshifts
    'classification': [
        'class',                 # GALAXY/STAR/QSO/UNSURE
        'prob_Galaxy',           # Galaxy probability
        'prob_Star',             # Star probability
        'prob_QSO',              # Quasar probability
        'extrapolation_Class',   # Classification extrapolation flag
        'cellDistance_Class',    # Classification SOM cell distance
        'cellID_Class',          # Classification SOM cell ID
        'z_phot',               # Photometric redshift (Monte-Carlo)
        'z_photErr',            # Redshift error
        'z_phot0',              # Base photometric redshift
        'extrapolation_Photoz',  # Photo-z extrapolation flag
        'cellDistance_Photoz',   # Photo-z SOM cell distance
        'cellID_Photoz'         # Photo-z SOM cell ID
    ],

    # Primary WISE photometry measurements
    'wise_base': [
        'w1mpro', 'w1sigmpro', 'w1rchi2', 'w1sat',
        'w2mpro', 'w2sigmpro', 'w2rchi2', 'w2sat',
        'w3mpro', 'w3sigmpro', 'w3rchi2', 'w3sat',
        'w4mpro', 'w4sigmpro', 'w4rchi2', 'w4sat'
    ],

    # Extended WISE photometry (different apertures)
    'wise_extended': [
        # W1 measurements
        'w1mag', 'w1sigm', 'w1flg',           # Standard aperture
        'w1mag_1', 'w1sigm_1', 'w1flg_1',     # 5.5" radius
        'w1mag_4', 'w1sigm_4', 'w1flg_4',     # 13.75" radius
        'w1mag_7', 'w1sigm_7', 'w1flg_7',     # 22.0" radius
        # W2 measurements
        'w2mag', 'w2sigm', 'w2flg',
        'w2mag_1', 'w2sigm_1', 'w2flg_1',
        'w2mag_4', 'w2sigm_4', 'w2flg_4',
        'w2mag_7', 'w2sigm_7', 'w2flg_7',
        # W3 measurements
        'w3mag', 'w3sigm', 'w3flg',
        'w3mag_1', 'w3sigm_1', 'w3flg_1',
        'w3mag_4', 'w3sigm_4', 'w3flg_4',
        'w3mag_7', 'w3sigm_7', 'w3flg_7',
        # W4 measurements
        'w4mag', 'w4sigm', 'w4flg',
        'w4mag_1', 'w4sigm_1', 'w4flg_1',
        'w4mag_4', 'w4sigm_4', 'w4flg_4',
        'w4mag_7', 'w4sigm_7', 'w4flg_7'
    ],

    # Primary PS1 photometry measurements
    'ps1_base': [
        # PSF magnitudes for all bands
        'gFPSFMag', 'gFPSFMagErr',
        'rFPSFMag', 'rFPSFMagErr',
        'iFPSFMag', 'iFPSFMagErr',
        'zFPSFMag', 'zFPSFMagErr',
        'yFPSFMag', 'yFPSFMagErr'
    ],

    # Extended PS1 photometry and shape measurements
    'ps1_extended': [
        # g band extended measurements
        'gFKronMag', 'gFKronMagErr',
        'gFApMag', 'gFApMagErr',
        'gFmeanMagR5', 'gFmeanMagR5Err',
        'gFmeanMagR6', 'gFmeanMagR6Err',
        'gFmeanMagR7', 'gFmeanMagR7Err',
        'gnTotal', 'gnIncPSFFlux', 'gnIncKronFlux', 'gnIncApFlux',
        'gnIncR5', 'gnIncR6', 'gnIncR7',
        'gE1', 'gE2',
        # r band extended measurements
        'rFKronMag', 'rFKronMagErr',
        'rFApMag', 'rFApMagErr',
        'rFmeanMagR5', 'rFmeanMagR5Err',
        'rFmeanMagR6', 'rFmeanMagR6Err',
        'rFmeanMagR7', 'rFmeanMagR7Err',
        'rnTotal', 'rnIncPSFFlux', 'rnIncKronFlux', 'rnIncApFlux',
        'rnIncR5', 'rnIncR6', 'rnIncR7',
        'rE1', 'rE2',
        # i band extended measurements
        'iFKronMag', 'iFKronMagErr',
        'iFApMag', 'iFApMagErr',
        'iFmeanMagR5', 'iFmeanMagR5Err',
        'iFmeanMagR6', 'iFmeanMagR6Err',
        'iFmeanMagR7', 'iFmeanMagR7Err',
        'inTotal', 'inIncPSFFlux', 'inIncKronFlux', 'inIncApFlux',
        'inIncR5', 'inIncR6', 'inIncR7',
        'iE1', 'iE2',
        # z band extended measurements
        'zFKronMag', 'zFKronMagErr',
        'zFApMag', 'zFApMagErr',
        'zFmeanMagR5', 'zFmeanMagR5Err',
        'zFmeanMagR6', 'zFmeanMagR6Err',
        'zFmeanMagR7', 'zFmeanMagR7Err',
        'znTotal', 'znIncPSFFlux', 'znIncKronFlux', 'znIncApFlux',
        'znIncR5', 'znIncR6', 'znIncR7',
        'zE1', 'zE2',
        # y band extended measurements
        'yFKronMag', 'yFKronMagErr',
        'yFApMag', 'yFApMagErr',
        'yFmeanMagR5', 'yFmeanMagR5Err',
        'yFmeanMagR6', 'yFmeanMagR6Err',
        'yFmeanMagR7', 'yFmeanMagR7Err',
        'ynTotal', 'ynIncPSFFlux', 'ynIncKronFlux', 'ynIncApFlux',
        'ynIncR5', 'ynIncR6', 'ynIncR7',
        'yE1', 'yE2'
    ],

    # Quality flags and extinction
    'flags': [
        'cc_flags',    # WISE contamination flag
        'ext_flg',     # WISE extended source flag
        'ph_qual',     # WISE photometric quality
        'moon_lev',    # WISE moonlight contamination
        'gFlags',      # PS1 g-band flags
        'rFlags',      # PS1 r-band flags
        'iFlags',      # PS1 i-band flags
        'zFlags',      # PS1 z-band flags
        'yFlags',      # PS1 y-band flags
        'EBV_Planck',  # Planck dust map extinction
        'EBV_PS1'      # PS1 dust map extinction
    ]
}

def get_chunk_size(group_name, typical_rows=10):
    """
    Calculate chunk size based on group characteristics.
    Aim for chunks that contain ~10 rows but align with 4KB pages.
    """
    PAGE_SIZE = 4096  # 4KB memory pages
    
    # Base chunk sizes on group type
    if group_name in ['core', 'classification', 'wise_base', 'ps1_base', 'flags']:
        target_size = 32768  # 32KB for frequently accessed groups
    else:
        target_size = 131072  # 128KB for extended data
    
    # Round to nearest multiple of page size
    return (target_size // PAGE_SIZE) * PAGE_SIZE

def optimize_hdf5_file(input_file, output_file):
    """
    Create an optimized HDF5 file from input file.
    
    Args:
        input_file: Path to input HDF5 file
        output_file: Path to write optimized file
    """
    with h5py.File(input_file, 'r') as f_in:
        # Get input data
        table = f_in['data/table']
        n_rows = table.shape[0]
        
        # Create output file
        with h5py.File(output_file, 'w') as f_out:
            # Create a group for our data
            data_group = f_out.create_group('data')
            
            # Process each column group
            for group_name, columns in COLUMN_GROUPS.items():
                print(f"Processing group: {group_name}")
                # Calculate chunk size
                chunk_size = get_chunk_size(group_name)
                
                # Create dataset for this group
                # Note: We're using a compound dtype to keep related columns together
                dtypes = []
                for col in columns:
                    if col in table.dtype.names:
                        dtypes.append((col, table.dtype[col]))
                
                if not dtypes:
                    print(f"No columns found for group {group_name}")
                    continue  # Skip if no columns found for this group
                
                # Create the compound dtype
                group_dtype = np.dtype(dtypes)
                
                # Calculate chunks in rows (chunk_size / bytes_per_row)
                bytes_per_row = group_dtype.itemsize
                rows_per_chunk = max(1, chunk_size // bytes_per_row)
                
                print(f"  Bytes per row: {bytes_per_row}")
                print(f"  Rows per chunk: {rows_per_chunk}")
                
                # Create dataset
                dset = data_group.create_dataset(
                    group_name,
                    shape=(n_rows,),
                    dtype=group_dtype,
                    chunks=(rows_per_chunk,),
                    compression=None  # No compression for better memory mapping
                )
                
                # Copy data
                structured_array = np.empty(n_rows, dtype=group_dtype)
                for col in columns:
                    if col in table.dtype.names:
                        structured_array[col] = table[col]
                
                dset[:] = structured_array
                print(f"  Created dataset of size: {dset.nbytes / 1024 / 1024:.2f} MB")

def process_directory(input_dir, output_dir, pattern="chunk_ra_*_dec_*.h5"):
    """
    Process all matching files in input directory.
    
    Args:
        input_dir: Path to directory containing input files
        output_dir: Path to write optimized files
        pattern: Glob pattern to match input files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    for input_file in input_path.glob(pattern):
        # Create corresponding output filename
        output_file = output_path / input_file.name
        
        print(f"Processing {input_file.name}...")
        optimize_hdf5_file(str(input_file), str(output_file))
        print(f"Created optimized file: {output_file}")

# Example usage:
if __name__ == "__main__":
    input_dir = "/lustre/aoc/sciops/ddong/Catalogs/STRM_WISE/data/output_chunks"
    output_dir = "/lustre/aoc/sciops/ddong/Catalogs/STRM_WISE/data/optimized_chunks"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Test with a single file first
    input_file = f"{input_dir}/chunk_ra_227.5_dec_56.5.h5"
    output_file = f"{output_dir}/chunk_ra_227.5_dec_56.5.h5"
    optimize_hdf5_file(input_file, output_file)