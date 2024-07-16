# crossmatching
Python crossmatching tools designed for automated workflows and large datasets. Current catalogs:

(1) STRM (neural net classifications ['galaxy',''star','qso','unsure'] for all objects detected in Panstarrs DR1.) 
For details see:
- Beck et al. 2021: https://academic.oup.com/mnras/article/500/2/1633/5899759
- MAST: https://archive.stsci.edu/hlsp/ps1-strm

Setup:
- download_STRM.py (automated download of the STRM catalog)
- convert_to_hdf5.py (creates local hdf5 database. Tunable tile sizes for I/O efficiency.)
- STRM_metadata.csv, STRM_metadata_skycoord.pkl (metadata used in identifying the relevant files for crossmatching in the NRAO database. Will need to be adjusted for other setups.)

Cross matching:
- STRM_crossmatch.py (I/O and cross matching)
  - Finds closest n_matches (optionally within match_radius) for all points in input_cat, returning:
    - indices of matched coordinates, distance to the n-th nearest neighbor, and user requested STRM table columns    
  - supports arbitrary user-created selection functions on the STRM catalog
  - adjustable n_threads and max_chunk_size depending on local CPU and RAM resources
  
