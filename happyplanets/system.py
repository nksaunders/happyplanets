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

from .utils import read_local_data

__all__ = ['System']


class System(object):

    def __init__(self, target_name=None, ind=None):
        ''' '''

        self.target_name = target_name
        self.ind = ind
        self.fetch_parameters()

    def fetch_parameters(self):
        ''' '''

        # read in corresponding rows from local csv data file
        row = read_local_data(target_name=self.target_name, ind=self.ind)

        # store planet parameters
        self.host = np.atleast_1d(row['pl_hostname'])[0]

        self.pl_period = np.array(row['pl_orbper'], dtype=float)
        self.pl_period_err1 = np.array(row['pl_orbpererr1'], dtype=float)
        self.pl_period_err2 = np.array(row['pl_orbpererr2'], dtype=float)

        self.pl_t0 = np.array(row['pl_tranmid'], dtype=float) - 2454833 # JD -> BKJD
        self.pl_t0_err1 = np.array(row['pl_tranmiderr1'], dtype=float)
        self.pl_t0_err2 = np.array(row['pl_tranmiderr2'], dtype=float)

        self.pl_rad = np.array(row['pl_radj'], dtype=float)
        self.pl_rad_err1 = np.array(row['pl_radjerr1'], dtype=float)
        self.pl_rad_err2 = np.array(row['pl_radjerr2'], dtype=float)

        # store stellar parameters
        self.st_mass = np.atleast_1d(row['st_mass'])[0]
        self.st_mass_err1 = np.atleast_1d(row['st_masserr1'])[0]
        self.st_mass_err2 = np.atleast_1d(row['st_masserr2'])[0]

        self.st_rad = np.atleast_1d(row['st_rad'])[0]
        self.rad_star_err1 = np.atleast_1d(row['st_raderr1'])[0]
        self.rad_star_err2 = np.atleast_1d(row['st_raderr2'])[0]

        self.rprs = np.array([((rp * u.jupiterRad) / (self.st_rad * u.solRad)).value
                              for rp in self.pl_rad], dtype = float)
