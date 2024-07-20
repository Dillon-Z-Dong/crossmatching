import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
import os
import pandas as pd

import crossmatching_utils as cu


def loadTNS():
	''' Reads the latest TNS classified SN cat from this directory '''

	return tns_cat


def loadZTF():
	''' Downloads latest ZTF bright transient survey catalog, return skycoord '''

	return ztf_cat


def loadOSC():
	''' OSC is no longer updating, loads final file, returns skycoord '''

	return osc_cat



def crossmatch_SNe(input_cat: SkyCoord):
	''' Thin wrapper on crossmatch_skycoords applied to the 3 supernova catalogs'''



