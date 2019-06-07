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

__all__ = ['PlanetSystem', 'create_planet_system']


class PlanetSystem(object):
    """A class to hold the Star and Planet objects for a given system."""

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
                'Period':self.pl_orbper,
                'Period err1':self.pl_orbpererr1,
                'Period err2':self.pl_orbpererr2,
                't0':self.pl_tranmid,
                't0 err1':self.pl_tranmiderr1,
                't0 err2':self.pl_tranmiderr2,
                'Planet Radius':self.pl_radj,
                'Planet Radius err1':self.pl_radjerr1,
                'Planet Radius err2':self.pl_radjerr2,
                'Stellar Radius':self.st_rad,
                'Stellar Radius err1':self.st_raderr1,
                'Stellar Radius err2':self.st_raderr2,
                'Stellar Mass':self.st_mass,
                'Stellar Mass err1':self.st_masserr1,
                'Stellar Mass err2':self.st_masserr2,
                'Rp/Rs':self.rprs}

        return pd.DataFrame(dict).T.__repr__()


    def fetch_parameters(self):
        """Store stellar and planetary parameters as variables of PlanetSystem."""

        self.pl_orbper = np.array([p.pl_orbper.value for p in self.planets])
        self.pl_orbpererr1 = np.array([p.pl_orbpererr1.value for p in self.planets])
        self.pl_orbpererr2 = np.array([p.pl_orbpererr2.value for p in self.planets])

        self.pl_tranmid = np.array([p.pl_tranmid.value for p in self.planets])
        self.pl_tranmiderr1 = np.array([p.pl_tranmiderr1.value for p in self.planets])
        self.pl_tranmiderr2 = np.array([p.pl_tranmiderr2.value for p in self.planets])

        self.pl_radj = np.array([p.pl_radj.value for p in self.planets])
        self.pl_radjerr1 = np.array([p.pl_radjerr1.value for p in self.planets])
        self.pl_radjerr2 = np.array([p.pl_radjerr2.value for p in self.planets])

        self.rprs = np.array([p.rprs for p in self.planets])
        self.rprserr1 = np.array([p.rprserr1 for p in self.planets])
        self.rprserr2 = np.array([p.rprserr2 for p in self.planets])

        # store stellar parameters
        self.st_mass = self.star.st_mass.value
        self.st_masserr1 = self.star.st_masserr1.value
        self.st_masserr2 = self.star.st_masserr2.value

        self.st_rad = self.star.st_rad.value
        self.st_raderr1 = self.star.st_raderr1.value
        self.st_raderr2 = self.star.st_raderr2.value

        self.st_teff = self.star.st_teff.value
        self.st_tefferr1 = self.star.st_tefferr1.value
        self.st_tefferr2 = self.star.st_tefferr2.value


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
            mask |= time_mask(t, p.pl_tranmid.value, p.pl_orbper.value, n_dur_mask*p.duration.value)
        return mask


def create_planet_system(target_name):
    """ """
    rows = read_local_data(target_name=target_name)
    star = Star(rows)

    planets = []
    for i in range(len(rows)):
        planets.append(Planet(rows.iloc[i], star))

    return PlanetSystem(star=star, planets=planets)
