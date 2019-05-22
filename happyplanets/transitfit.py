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
import theano.tensor as tt
import astropy.units as unit
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box1DKernel
from itertools import combinations_with_replacement as multichoose

from .utils import silence

__all__ = ['TransitFit']

class TransitFit(object):

    def __init__(self, target_name=None, ind=None):
        # read in CSV file with planet parameters
        datadir = os.path.abspath(os.path.join(PACKAGEDIR, os.pardir, 'data'))
        self.kois = pd.read_csv(os.path.join(datadir,
                                'planets_2019.04.02_11.43.16.csv'),
                                skiprows=range(81))

        # set target name and find indices
        if target_name is not None:
            self.target_name = target_name
        elif ind is not None:
            self.target_name = self.kois['pl_hostname'][ind]
        else:
            raise ValueError("Please specify either `target_name` or `ind`.")

        self.ind = np.where(self.kois['pl_hostname'] == self.target_name)[0]
        self.prelim_model_built = False
