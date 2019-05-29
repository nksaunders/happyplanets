import sys
import os.path
import logging
import warnings
from . import PACKAGEDIR
from contextlib import contextmanager
from matplotlib.backends.backend_pdf import PdfPages

import copy
import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt
from lightkurve import MPLSTYLE
from astropy.table import Table

import corner
import pymc3 as pm
from fbpca import pca
import exoplanet as xo
import astropy.units as u
import theano.tensor as tt
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box1DKernel
from itertools import combinations_with_replacement as multichoose


class Star(object):

    def __init__(self, row):

        self.fetch_parameters(row)

    def fetch_parameters(self, row):
        """Store parameters read in from row."""

        # store stellar parameters
        self.host = np.atleast_1d(row['pl_hostname'])[0]
        self.st_mass = np.atleast_1d(row['st_mass'])[0] * u.solMass
        self.st_mass_err1 = np.atleast_1d(row['st_masserr1'])[0] * u.solMass
        self.st_mass_err2 = np.atleast_1d(row['st_masserr2'])[0] * u.solMass

        self.st_rad = np.atleast_1d(row['st_rad'])[0] * u.solRad
        self.st_rad_err1 = np.atleast_1d(row['st_raderr1'])[0] * u.solRad
        self.st_rad_err2 = np.atleast_1d(row['st_raderr2'])[0] * u.solRad

    @property
    def properties(self):
        """Returns dictionary of stellar parameters."""

        dict = {"Host Name":self.host, "Stellar Mass":self.st_mass,
                "Stellar Radius":self.st_rad}

        return dict

    @property
    def errors(self):
        """Returns a dictionary of errors on stellar parameters."""

        dict = {"Stellar Mass Error":[self.st_mass_err1,self.st_mass_err2],
                "Stellar Radius Error":[self.st_rad_err1,self.st_rad_err2]}

        return dict
