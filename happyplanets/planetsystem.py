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

        self.star = star
        self.planets = planets

        # array of planet labels
        self.n_planets = len(self.planets)
        self.letters = "bcdefghijklmnopqrstuvwxyz"[:self.n_planets]
        self.fetch_parameters()


    def __repr__(self):

        dict = {'n Planets':self.n_planets,
                'Host Name':self.host,
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


    def fetch_parameters(self):
        """Store stellar and planetary parameters as variables of PlanetSystem."""

        self.pl_period = np.array([p.pl_period.value for p in self.planets])
        self.pl_period_err1 = np.array([p.pl_period_err1.value for p in self.planets])
        self.pl_period_err2 = np.array([p.pl_period_err2.value for p in self.planets])

        self.pl_t0 = np.array([p.pl_t0.value for p in self.planets])
        self.pl_t0_err1 = np.array([p.pl_t0_err1.value for p in self.planets])
        self.pl_t0_err2 = np.array([p.pl_t0_err2.value for p in self.planets])

        self.pl_rad = np.array([p.pl_rad.value for p in self.planets])
        self.pl_rad_err1 = np.array([p.pl_rad_err1.value for p in self.planets])
        self.pl_rad_err2 = np.array([p.pl_rad_err2.value for p in self.planets])

        self.rprs = np.array([p.rprs for p in self.planets])
        self.rprs_err1 = np.array([p.rprs_err1 for p in self.planets])
        self.rprs_err2 = np.array([p.rprs_err2 for p in self.planets])

        # store stellar parameters
        self.st_mass = self.star.st_mass.value
        self.st_mass_err1 = self.star.st_mass_err1.value
        self.st_mass_err2 = self.star.st_mass_err2.value

        self.st_rad = self.star.st_rad.value
        self.st_rad_err1 = self.star.st_rad_err1.value
        self.st_rad_err2 = self.star.st_rad_err2.value


    def build_star(self, rows):
        """Instantiate a Star object."""

        return Star(rows)


    def build_planets(self, rows, host):
        """Instantiate a Planet object for each row containing the host star."""

        planets = []

        for i in range(len(rows)):
            planets.append(Planet(rows.iloc[i], host))

        return planets


    def create_planet_mask(self, t, n_dur_mask=2):
        """Return cadences in t during transit given t0, period, duration."""

        mask = np.zeros_like(t, dtype=bool)
        for p in self.planets:
            mask |= time_mask(t, p.pl_t0.value, p.pl_period.value, n_dur_mask*p.duration.value)
        return mask


def create_planet_system(target_name):
    """ """
    rows = read_local_data(target_name=target_name)
    star = Star(rows)

    planets = []
    for i in range(len(rows)):
        planets.append(Planet(rows.iloc[i], star))

    return PlanetSystem(star=star, planets=planets)
