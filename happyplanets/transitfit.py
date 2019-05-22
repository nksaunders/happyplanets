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

from .utils import download_files, silence
from .system import System

__all__ = ['TransitFit']

class TransitFit(object):

    def __init__(self, target_name=None, ind=None):
        ''' '''

        self.system = System(target_name=target_name, ind=ind)
        print(self.system.host)
        self.tpf_collection = download_files(self.system.host)

    def test_fit(self):
        ''' '''
        pass
