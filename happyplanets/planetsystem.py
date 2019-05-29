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

from .utils import read_local_data, time_mask
from .planet import Planet
from .star import Star

__all__ = ['PlanetSystem']


class PlanetSystem(object):
    """ """

    def __init__(self, target_name=None, ind=None, star=None, planets=[]):

        self.target_name = target_name
        self.ind = ind
        self.star = star # to be made into a class later
        self.planets = planets # samesies
        self._fetch_parameters()

    def __repr__(self):

        dict = {'n Planets':self.n_planets,
                'Host Name':self.target_name,
                'Period':self.pl_period,
                'Period err1':self.pl_period_err1,
                'Period err2':self.pl_period_err2,
                't0':self.pl_t0,
                't0 err1':self.pl_t0_err1,
                't0 err2':self.pl_t0_err2,
                'Planet Radius':self.pl_rad,
                'Planet Radius err1':self.pl_rad_err1,
                'Planet Radius err2':self.pl_rad_err2,
                'Stellar Radius':self.st_rad,
                'Stellar Radius err1':self.st_rad_err1,
                'Stellar Radius err2':self.st_rad_err2,
                'Stellar Mass':self.st_mass,
                'Stellar Mass err1':self.st_mass_err1,
                'Stellar Mass err2':self.st_mass_err2,
                'Rp/Rs':self.rprs}

        return pd.DataFrame(dict).T.__repr__()


    def _fetch_parameters(self):
        """ """

        # read in corresponding rows from local csv data file
        row = read_local_data(target_name=self.target_name)

        # store planet parameters
        self.host = np.atleast_1d(row['pl_hostname'])[0]

        self.pl_period = np.array(row['pl_orbper'], dtype=float)
        self.pl_period_err1 = np.array(row['pl_orbpererr1'], dtype=float)
        self.pl_period_err2 = np.array(row['pl_orbpererr2'], dtype=float)

        self.pl_t0 = np.array(row['pl_tranmid'], dtype=float) - 2454833 # JD -> BKJD
        self.pl_t0_err1 = np.array(row['pl_tranmiderr1'], dtype=float)
        self.pl_t0_err2 = np.array(row['pl_tranmiderr2'], dtype=float)

        self.pl_rad = np.array(row['pl_radj'], dtype=float) / 11.21
        self.pl_rad_err1 = np.array(row['pl_radjerr1'], dtype=float)
        self.pl_rad_err2 = np.array(row['pl_radjerr2'], dtype=float)

        # store stellar parameters
        self.st_mass = np.atleast_1d(row['st_mass'])[0]
        self.st_mass_err1 = np.atleast_1d(row['st_masserr1'])[0]
        self.st_mass_err2 = np.atleast_1d(row['st_masserr2'])[0]

        self.st_rad = np.atleast_1d(row['st_rad'])[0]
        self.st_rad_err1 = np.atleast_1d(row['st_raderr1'])[0]
        self.st_rad_err2 = np.atleast_1d(row['st_raderr2'])[0]

        self.rprs = np.array([((rp * u.jupiterRad) / (self.st_rad * u.solRad)).value
                              for rp in self.pl_rad], dtype = float)

        # array of planet labels
        self.n_planets = len(self.pl_period)
        self.letters = "bcdefghijklmnopqrstuvwxyz"[:self.n_planets]


    def build_star(self):
        """Instantiate a Star object with read in parameters."""
        star = Star(radius=self.st_rad, radius_err=[self.st_rad_err1, self.st_rad_err2],
                     mass=self.st_mass, mass_err=[self.st_mass_err1, self.st_mass_err2])

        return star

    def create_planet_mask(self, t):
        """Return cadences in t during transit given t0, period, duration."""
        mask = np.zeros_like(t, dtype=bool)
        for i in range(self.n_planets):
            mask |= time_mask(t, self.pl_t0[i], self.pl_period[i], 0.3)
        return mask
