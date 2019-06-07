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
from astropy.constants import G
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box1DKernel
from itertools import combinations_with_replacement as multichoose


class Planet(object):

    def __init__(self, row, host):

        self.fetch_parameters(row, host)


    def fetch_parameters(self, row, host):
        """Read in parameters from local file."""

        # store planet parameters
        self.host = host
        self.letter = row['pl_letter']

        self.pl_orbper = row['pl_orbper'] * u.day
        self.pl_orbpererr1 = row['pl_orbpererr1'] * u.day
        self.pl_orbpererr2 = row['pl_orbpererr2'] * u.day

        self.pl_tranmid = (row['pl_tranmid'] - 2454833) * u.day
        self.pl_tranmiderr1 = row['pl_tranmiderr1'] * u.day
        self.pl_tranmiderr2 = row['pl_tranmiderr2'] * u.day

        self.pl_radj = (row['pl_radj'] * u.jupiterRad).to(u.earthRad)
        self.pl_radjerr1 = (row['pl_radjerr1']* u.jupiterRad).to(u.earthRad)
        self.pl_radjerr2 = (row['pl_radjerr2']* u.jupiterRad).to(u.earthRad)

        self.rprs = (self.pl_radj.to(u.solRad) / self.host.st_rad).value
        self.rprserr1 = (self.pl_radjerr1.to(u.solRad) / self.host.st_rad).value
        self.rprserr2 = (self.pl_radjerr2.to(u.solRad) / self.host.st_rad).value


    @property
    def separation(self):
        sep = (((G*self.host.st_mass/(4*np.pi**2)) * (self.pl_orbper)**2)**(1/3)).to(u.solRad)
        separation = (sep / self.host.st_mass).value

        return separation

    @property
    def duration(self):
        self.inclination = 90 # check this param
        b = self.separation * np.cos(self.inclination * np.pi/180)
        l = ((self.host.st_rad + self.pl_radj.to(u.solRad))**2 + (b * self.host.st_rad)**2)**0.5
        l /= (self.separation*self.host.st_rad)

        return (np.arcsin(l.value) * self.pl_orbper/np.pi)
