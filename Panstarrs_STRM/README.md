Setup:
- download_STRM[_WISE].py (automated download of the STRM catalogs.)
- convert_to_hdf5[_STRM_WISE].py (creates local hdf5 database. Tunable tile sizes for I/O efficiency.)

Cross matching:
- STRM_crossmatch.py (I/O and cross matching)
  - returns table with the nearest k matches in either STRM or STRM-WISE that satisfy your selection function / crossmatching criteria
  - supports arbitrary user-created selection functions on the STRM catalog
  - supports arbitrary user-created crossmatching criteria (see crossmatching_utils.py)
  - adjustable n_threads and max_chunk_size depending on local CPU and RAM resources

- Make sure to edit the paths in get_metadata()