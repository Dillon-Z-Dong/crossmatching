import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

def partition_and_save(csv_gz_path, output_dir='output_chunks', log_file='log.csv', complevel=2):
    """
    Reads a .csv.gz file, assigns column names, partitions the DataFrame into 1x1 degree chunks with a buffer, 
    and saves each chunk as an HDF5 file. Also logs the number of rows, output file size, and central coordinates 
    of each HDF5 file created.
    
    Parameters:
    csv_gz_path (str): Path to the input .csv.gz file.
    output_dir (str): Path to the directory where the HDF5 files will be saved.
    log_file (str): Path to the log file where the number of rows, output file size, and central coordinates will be recorded.
    complevel (int): Compression level for HDF5 files.
    """
    # Column names
    column_names = ['objID', 'uniquePspsOBid', 'raMean', 'decMean', 'l', 'b', 'class', 
                    'prob_Galaxy', 'prob_Star', 'prob_QSO', 'extrapolation_Class', 
                    'cellDistance_Class', 'cellID_Class', 'z_phot', 'z_photErr', 
                    'z_phot0', 'extrapolation_Photoz', 'cellDistance_Photoz', 'cellID_Photoz']
    
    # Read the .csv.gz file into a pandas DataFrame
    df = pd.read_csv(csv_gz_path, compression='gzip', names=column_names, header=0)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the log DataFrame
    log_df = pd.DataFrame(columns=['filename', 'num_rows', 'filesize_MB', 'central_ra', 'central_dec'])

    # Determine the min and max values for raMean and decMean
    ra_min = int(np.floor(df['raMean'].min()))
    ra_max = int(np.ceil(df['raMean'].max()))
    dec_min = int(np.floor(df['decMean'].min()))
    dec_max = int(np.ceil(df['decMean'].max()))

    # Loop over the range of ra and dec to create 1x1 degree chunks with a buffer
    for ra in range(ra_min, ra_max):
        for dec in range(dec_min, dec_max):
            chunk = df[(df['raMean'] >= ra - 0.05) & (df['raMean'] < ra + 1.05) &
                       (df['decMean'] >= dec - 0.05) & (df['decMean'] < dec + 1.05)]
            if not chunk.empty:
                central_ra = ra + 0.5
                central_dec = dec + 0.5
                filename = f'{output_dir}/chunk_ra_{central_ra}_dec_{central_dec}.h5'
                chunk.to_hdf(filename, key='data', mode='w', complevel=complevel, complib='blosc',format='table')
                num_rows = len(chunk)
                filesize_MB = os.path.getsize(filename) / (1024 * 1024)
                new_log_entry = pd.DataFrame([{'filename': filename, 'num_rows': num_rows, 'filesize_MB': filesize_MB,
                                               'central_ra': central_ra, 'central_dec': central_dec}])
                log_df = pd.concat([log_df, new_log_entry], ignore_index=True)
                print(f'Saved {filename} with {num_rows} rows, size {filesize_MB:.2f} MB.')

    # Append the log DataFrame to the log file
    log_df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

def initialize_log(output_dir='output_chunks', log_file='log.csv'):
    """
    Initializes the log file with details of the files that are already in the output directory.
    
    Parameters:
    output_dir (str): Path to the directory where the HDF5 files are saved.
    log_file (str): Path to the log file where the number of rows, output file size, and central coordinates will be recorded.
    """
    log_df = pd.DataFrame(columns=['filename', 'num_rows', 'filesize_MB', 'central_ra', 'central_dec'])
    
    for filename in os.listdir(output_dir):
        if filename.endswith('.h5'):
            file_path = os.path.join(output_dir, filename)
            df = pd.read_hdf(file_path)
            num_rows = len(df)
            filesize_MB = os.path.getsize(file_path) / (1024 * 1024)
            central_ra = float(filename.split('_')[2])
            central_dec = float(filename.split('_')[4].split('.h5')[0])
            new_log_entry = pd.DataFrame([{'filename': file_path, 'num_rows': num_rows, 'filesize_MB': filesize_MB,
                                           'central_ra': central_ra, 'central_dec': central_dec}])
            log_df = pd.concat([log_df, new_log_entry], ignore_index=True)

    # Save the log DataFrame to the log file
    log_df.to_csv(log_file, mode='w', index=False)

def check_and_process_files():
    processed_files = set()
    
    # Initialize the log file with the existing files in the output directory
    initialize_log()

    while True:
        current_time = datetime.now()
        for filename in os.listdir('.'):
            if filename.endswith('.csv.gz'):
                file_path = os.path.join('.', filename)
                modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_path not in processed_files and (current_time - modified_time) > timedelta(minutes=5):
                    try:
                        partition_and_save(file_path)
                        processed_files.add(file_path)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        time.sleep(60)  # Check every minute

# Start the infinite loop to check and process files
check_and_process_files()
