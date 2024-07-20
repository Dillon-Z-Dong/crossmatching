import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

def partition_and_save(csv_gz_path, output_dir='output_chunks', log_file='STRM_WISE_metadata.csv', complevel=2):
    """
    Reads a .csv.gz file, assigns column names, partitions the DataFrame into 1x1 degree chunks with a buffer, 
    and saves each chunk as an HDF5 file. Also logs the number of rows, output file size, and central coordinates 
    of each HDF5 file created.
    
    Parameters:
    csv_gz_path (str): Path to the input .csv.gz file.
    output_dir (str): Path to the directory where the HDF5 files will be saved.
    log_file (str): Path to the log file where the number of rows, output file size, and central coordinates will be recorded.
    complevel (int): Compression level for HDF5 files.

    column names:
    ['objID', 'raMean', 'raMeanErr', 'decMean', 'decMeanErr', 'l', 'b', 'distance_Deg', 'sqrErr_Arcsec', 'BayesFactor', 'cntr', 'HtmID', 'ra', 'dec', 'sigra', 'sigdec', 'sigradec', 'cc_flags', 'ext_flg', 'ph_qual', 'moon_lev', 'w1mpro', 'w1sigmpro', 'w1rchi2', 'w1sat', 'w1mag', 'w1sigm', 'w1flg', 'w1mag_1', 'w1sigm_1', 'w1flg_1', 'w1mag_4', 'w1sigm_4', 'w1flg_4', 'w1mag_7', 'w1sigm_7', 'w1flg_7', 'w2mpro', 'w2sigmpro', 'w2rchi2', 'w2sat', 'w2mag', 'w2sigm', 'w2flg', 'w2mag_1', 'w2sigm_1', 'w2flg_1', 'w2mag_4', 'w2sigm_4', 'w2flg_4', 'w2mag_7', 'w2sigm_7', 'w2flg_7', 'w3mpro', 'w3sigmpro', 'w3rchi2', 'w3sat', 'w3mag', 'w3sigm', 'w3flg', 'w3mag_1', 'w3sigm_1', 'w3flg_1', 'w3mag_4', 'w3sigm_4', 'w3flg_4', 'w3mag_7', 'w3sigm_7', 'w3flg_7', 'w4mpro', 'w4sigmpro', 'w4rchi2', 'w4sat', 'w4mag', 'w4sigm', 'w4flg', 'w4mag_1', 'w4sigm_1', 'w4flg_1', 'w4mag_4', 'w4sigm_4', 'w4flg_4', 'w4mag_7', 'w4sigm_7', 'w4flg_7', 'gFPSFMag', 'gFPSFMagErr', 'gFKronMag', 'gFKronMagErr', 'gFApMag', 'gFApMagErr', 'gFmeanMagR5', 'gFmeanMagR5Err', 'gFmeanMagR6', 'gFmeanMagR6Err', 'gFmeanMagR7', 'gFmeanMagR7Err', 'gnTotal', 'gnIncPSFFlux', 'gnIncKronFlux', 'gnIncApFlux', 'gnIncR5', 'gnIncR6', 'gnIncR7', 'gFlags', 'gE1', 'gE2', 'rFPSFMag', 'rFPSFMagErr', 'rFKronMag', 'rFKronMagErr', 'rFApMag', 'rFApMagErr', 'rFmeanMagR5', 'rFmeanMagR5Err', 'rFmeanMagR6', 'rFmeanMagR6Err', 'rFmeanMagR7', 'rFmeanMagR7Err', 'rnTotal', 'rnIncPSFFlux', 'rnIncKronFlux', 'rnIncApFlux', 'rnIncR5', 'rnIncR6', 'rnIncR7', 'rFlags', 'rE1', 'rE2', 'iFPSFMag', 'iFPSFMagErr', 'iFKronMag', 'iFKronMagErr', 'iFApMag', 'iFApMagErr', 'iFmeanMagR5', 'iFmeanMagR5Err', 'iFmeanMagR6', 'iFmeanMagR6Err', 'iFmeanMagR7', 'iFmeanMagR7Err', 'inTotal', 'inIncPSFFlux', 'inIncKronFlux', 'inIncApFlux', 'inIncR5', 'inIncR6', 'inIncR7', 'iFlags', 'iE1', 'iE2', 'zFPSFMag', 'zFPSFMagErr', 'zFKronMag', 'zFKronMagErr', 'zFApMag', 'zFApMagErr', 'zFmeanMagR5', 'zFmeanMagR5Err', 'zFmeanMagR6', 'zFmeanMagR6Err', 'zFmeanMagR7', 'zFmeanMagR7Err', 'znTotal', 'znIncPSFFlux', 'znIncKronFlux', 'znIncApFlux', 'znIncR5', 'znIncR6', 'znIncR7', 'zFlags', 'zE1', 'zE2', 'yFPSFMag', 'yFPSFMagErr', 'yFKronMag', 'yFKronMagErr', 'yFApMag', 'yFApMagErr', 'yFmeanMagR5', 'yFmeanMagR5Err', 'yFmeanMagR6', 'yFmeanMagR6Err', 'yFmeanMagR7', 'yFmeanMagR7Err', 'ynTotal', 'ynIncPSFFlux', 'ynIncKronFlux', 'ynIncApFlux', 'ynIncR5', 'ynIncR6', 'ynIncR7', 'yFlags', 'yE1', 'yE2', 'EBV_Planck', 'EBV_PS1', 'class', 'prob_Galaxy', 'prob_Star', 'prob_QSO', 'extrapolation_Class', 'cellDistance_Class', 'cellID_Class', 'z_phot', 'z_photErr', 'z_phot0', 'extrapolation_Photoz', 'cellDistance_Photoz', 'cellID_Photoz']
    """

    # Get column names from readme
    a = pd.read_csv('readme.txt',sep = '\t+',skiprows=28,names = ['Name','_','Unit','Data_Type','Size','Description','_2'], engine = 'python')
    column_names = [x.strip() for x in list(a['Name'])]
    
    # Read the .csv.gz file into a pandas DataFrame
    df = pd.read_csv(csv_gz_path, compression='gzip', names=column_names, header=0)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the log DataFrame
    try:
        log_df = pd.read_csv(log_file)
    except:
        log_df = pd.DataFrame(columns=['filename', 'num_rows', 'filesize_MB', 'central_ra', 'central_dec'])

    # Determine the min and max values for raMean and decMean
    ra_min = int(np.floor(df['raMean'].min()))
    ra_max = int(np.ceil(df['raMean'].max()))
    dec_min = int(np.floor(df['decMean'].min()))
    dec_max = int(np.ceil(df['decMean'].max()))

    # Loop over the range of ra and dec to create 1x1 degree chunks with a buffer
    for ra in range(ra_min, ra_max):
        for dec in range(dec_min, dec_max):
            central_ra = ra + 0.5
            central_dec = dec + 0.5
            filename = f'{output_dir}/chunk_ra_{central_ra}_dec_{central_dec}.h5'
            if not os.path.exists(filename):
                chunk = df[(df['raMean'] >= ra - 0.05) & (df['raMean'] < ra + 1.05) &
                           (df['decMean'] >= dec - 0.05) & (df['decMean'] < dec + 1.05)]

                if not chunk.empty:
                    chunk.to_hdf(filename, key='data', mode='w', complevel=complevel, complib='blosc',format='table')
                    num_rows = len(chunk)
                    filesize_MB = os.path.getsize(filename) / (1024 * 1024)
                    new_log_entry = pd.DataFrame([{'filename': filename, 'num_rows': num_rows, 'filesize_MB': filesize_MB,
                                                   'central_ra': central_ra, 'central_dec': central_dec}])
                    log_df = pd.concat([log_df, new_log_entry], ignore_index=True)
                    print(f'Saved {filename} with {num_rows} rows, size {filesize_MB:.2f} MB.')

    # Append the log DataFrame to the log file
    log_df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

def initialize_log(output_dir='output_chunks', log_file='STRM_WISE_metadata.csv'):
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
    #initialize_log()

    while True:
        current_time = datetime.now()
        for filename in os.listdir('.'):
            if filename.endswith('.csv.gz'):
                file_path = os.path.join('.', filename)
                modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_path not in processed_files: 
                    if (current_time - modified_time) > timedelta(minutes=5):
                        try:
                            print(f'Processing {file_path}')
                            partition_and_save(file_path)
                            processed_files.add(file_path)
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
                    else:
                        print(f"File {file_path} last modified {(current_time - modified_time).total_seconds()} seconds ago")
        #time.sleep(60)  # Check every minute

# Start the infinite loop to check and process files
check_and_process_files()
