import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
import os
import pandas as pd

import crossmatching_utils as cu


def estimate_false_alarm_rate(n_rand, match_function, match_cat, selection_function):
	''' Estimates the false alarm probability of a cross match '''

	raise NotImplementedError('false alarm rate is not yet implemented')