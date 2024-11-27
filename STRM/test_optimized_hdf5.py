import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

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
        'ra',             # WISE RA
        'dec',            # WISE Dec
        'sigra',          # WISE RA uncertainty
        'sigdec',         # WISE Dec uncertainty
        'sigradec',       # WISE position error correlation
        'EBV_Planck',     # Planck dust map extinction
        'EBV_PS1'         # PS1 dust map extinction
    ],

    # Neural network classifications and photometric redshifts
    'classification': [
        'class',                # GALAXY/STAR/QSO/UNSURE
        'prob_Galaxy',          # Galaxy probability
        'prob_Star',            # Star probability
        'prob_QSO',             # Quasar probability
        'extrapolation_Class',  # Classification extrapolation flag
        'cellDistance_Class',   # Classification SOM cell distance
        'cellID_Class',         # Classification SOM cell ID
        'z_phot',              # Photometric redshift (Monte-Carlo)
        'z_photErr',           # Redshift error
        'z_phot0',             # Base photometric redshift
        'extrapolation_Photoz', # Photo-z extrapolation flag
        'cellDistance_Photoz',  # Photo-z SOM cell distance
        'cellID_Photoz'        # Photo-z SOM cell ID
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

    # Primary PS1 photometry measurements grouped by band
    'ps1_base': [
        # g band primary measurements
        'gFPSFMag', 'gFPSFMagErr',
        'gFKronMag', 'gFKronMagErr',
        'gFApMag', 'gFApMagErr',
        # r band primary measurements
        'rFPSFMag', 'rFPSFMagErr',
        'rFKronMag', 'rFKronMagErr',
        'rFApMag', 'rFApMagErr',
        # i band primary measurements
        'iFPSFMag', 'iFPSFMagErr',
        'iFKronMag', 'iFKronMagErr',
        'iFApMag', 'iFApMagErr',
        # z band primary measurements
        'zFPSFMag', 'zFPSFMagErr',
        'zFKronMag', 'zFKronMagErr',
        'zFApMag', 'zFApMagErr',
        # y band primary measurements
        'yFPSFMag', 'yFPSFMagErr',
        'yFKronMag', 'yFKronMagErr',
        'yFApMag', 'yFApMagErr'
    ],

    # Extended PS1 photometry and shape measurements
    'ps1_extended': [
        # g band extended measurements
        'gFmeanMagR5', 'gFmeanMagR5Err',
        'gFmeanMagR6', 'gFmeanMagR6Err',
        'gFmeanMagR7', 'gFmeanMagR7Err',
        'gnTotal', 'gnIncPSFFlux', 'gnIncKronFlux', 'gnIncApFlux',
        'gnIncR5', 'gnIncR6', 'gnIncR7',
        'gE1', 'gE2',
        # r band extended measurements
        'rFmeanMagR5', 'rFmeanMagR5Err',
        'rFmeanMagR6', 'rFmeanMagR6Err',
        'rFmeanMagR7', 'rFmeanMagR7Err',
        'rnTotal', 'rnIncPSFFlux', 'rnIncKronFlux', 'rnIncApFlux',
        'rnIncR5', 'rnIncR6', 'rnIncR7',
        'rE1', 'rE2',
        # i band extended measurements
        'iFmeanMagR5', 'iFmeanMagR5Err',
        'iFmeanMagR6', 'iFmeanMagR6Err',
        'iFmeanMagR7', 'iFmeanMagR7Err',
        'inTotal', 'inIncPSFFlux', 'inIncKronFlux', 'inIncApFlux',
        'inIncR5', 'inIncR6', 'inIncR7',
        'iE1', 'iE2',
        # z band extended measurements
        'zFmeanMagR5', 'zFmeanMagR5Err',
        'zFmeanMagR6', 'zFmeanMagR6Err',
        'zFmeanMagR7', 'zFmeanMagR7Err',
        'znTotal', 'znIncPSFFlux', 'znIncKronFlux', 'znIncApFlux',
        'znIncR5', 'znIncR6', 'znIncR7',
        'zE1', 'zE2',
        # y band extended measurements
        'yFmeanMagR5', 'yFmeanMagR5Err',
        'yFmeanMagR6', 'yFmeanMagR6Err',
        'yFmeanMagR7', 'yFmeanMagR7Err',
        'ynTotal', 'ynIncPSFFlux', 'ynIncKronFlux', 'ynIncApFlux',
        'ynIncR5', 'ynIncR6', 'ynIncR7',
        'yE1', 'yE2'
    ],

    # Quality flags
    'flags': [
        'HtmID',      # Hierarchical triangular mesh index
        'cc_flags',   # WISE contamination flag
        'ext_flg',    # WISE extended source flag
        'ph_qual',    # WISE photometric quality
        'moon_lev',   # WISE moonlight contamination
        'gFlags',     # PS1 g-band flags
        'rFlags',     # PS1 r-band flags
        'iFlags',     # PS1 i-band flags
        'zFlags',     # PS1 z-band flags
        'yFlags'      # PS1 y-band flags
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

def read_input_file(input_file):
    """
    Read input HDF5 file using pandas.
    
    Args:
        input_file: Path to input HDF5 file
    
    Returns:
        pandas DataFrame containing all data
    """
    try:
        # Read the HDF5 file - try different keys if 'data' doesn't work
        try:
            df = pd.read_hdf(input_file, key='data')
        except KeyError:
            df = pd.read_hdf(input_file, key='data/table')
        
        print(f"Successfully read {len(df)} rows from {input_file}")
        return df
    except Exception as e:
        print(f"Error reading {input_file}: {str(e)}")
        raise

def optimize_hdf5_file(input_file, output_file):
    """
    Create an optimized HDF5 file from input file using pandas.
    
    Args:
        input_file: Path to input HDF5 file
        output_file: Path to write optimized file
    """
    print(f"\nProcessing {input_file}")
    
    # Read input data
    df = read_input_file(input_file)
    
    # Create output file
    store = pd.HDFStore(output_file, mode='w')
    
    try:
        # Process each column group
        for group_name, columns in COLUMN_GROUPS.items():
            print(f"\nProcessing group: {group_name}")
            
            # Filter columns that exist in the dataframe
            available_cols = [col for col in columns if col in df.columns]
            if not available_cols:
                print(f"No columns found for group {group_name}")
                continue
            
            # Create dataframe for this group
            group_df = df[available_cols].copy()
            
            # Calculate chunk size in rows
            chunk_size = get_chunk_size(group_name)
            bytes_per_row = group_df.memory_usage(deep=True).sum() / len(group_df)
            rows_per_chunk = max(1, chunk_size // int(bytes_per_row))
            
            print(f"  Columns: {', '.join(available_cols)}")
            print(f"  Bytes per row: {bytes_per_row:.1f}")
            print(f"  Rows per chunk: {rows_per_chunk}")
            
            # Store the group
            store.put(
                group_name,
                group_df,
                format='table',
                complevel=None,  # No compression for better memory mapping
                chunksize=rows_per_chunk,
                # Additional performance options
                data_columns=True,  # Enable searching/indexing
                index=False  # Don't store the index
            )
            
            # Report size
            dataset_size = store.get_storer(group_name).nrows * bytes_per_row
            print(f"  Dataset size: {dataset_size / 1024 / 1024:.2f} MB")
            
    finally:
        store.close()
    
    print(f"\nCreated optimized file: {output_file}")

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
    
    # Get list of all input files
    input_files = list(input_path.glob(pattern))
    print(f"Found {len(input_files)} files to process")
    
    # Process each file
    for i, input_file in enumerate(input_files, 1):
        try:
            output_file = output_path / input_file.name
            print(f"\nProcessing file {i}/{len(input_files)}: {input_file.name}")
            optimize_hdf5_file(str(input_file), str(output_file))
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            continue

def verify_optimized_file(input_file, output_file):
    """
    Verify that the optimized file contains the same data as the input file.
    
    Args:
        input_file: Path to original input file
        output_file: Path to optimized output file
    """
    print(f"\nVerifying {output_file}")
    
    # Read original data
    input_df = read_input_file(input_file)
    
    # Read optimized data
    store = pd.HDFStore(output_file, mode='r')
    try:
        # Combine all groups back into one dataframe
        dfs = []
        for group in store.keys():
            group_name = group.lstrip('/')  # Remove leading slash
            df = store.get(group)
            print(f"Group {group_name}: {len(df)} rows, {len(df.columns)} columns")
            dfs.append(df)
        
        output_df = pd.concat(dfs, axis=1)
        
        # Compare number of rows and columns
        print(f"\nInput shape: {input_df.shape}")
        print(f"Output shape: {output_df.shape}")
        
        # Compare column values
        common_cols = set(input_df.columns) & set(output_df.columns)
        print(f"\nCommon columns: {len(common_cols)}")
        
        for col in common_cols:
            if not np.array_equal(input_df[col].values, output_df[col].values):
                print(f"Warning: Mismatch in column {col}")
    
    finally:
        store.close()

if __name__ == "__main__":
    # Suppress pandas warnings about performance
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    input_dir = "/lustre/aoc/sciops/ddong/Catalogs/STRM_WISE/data/output_chunks"
    output_dir = "/lustre/aoc/sciops/ddong/Catalogs/STRM_WISE/data/optimized_chunks"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Test with a single file first
    input_file = f"{input_dir}/chunk_ra_227.5_dec_56.5.h5"
    output_file = f"{output_dir}/chunk_ra_227.5_dec_56.5.h5"
    
    # Process the file
    optimize_hdf5_file(input_file, output_file)
    
    # Verify the output
    verify_optimized_file(input_file, output_file)