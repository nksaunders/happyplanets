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

from .correct import Corrector


class TimeSeries(object):
    """ """

    def __init__(self, target_name, system):

        self.target_name = target_name
        self.system = system
        self.global_aperture = 'pipeline'


    def return_lightcurve(self, n_obs='all'):
        """ """

        tpf_collection = self.download_files(self.target_name, n_obs)
        return self.detrend(tpf_collection)


    def detrend(self, tpf_collection):
        """ """

        # assign variables
        x = np.array([], np.float64)
        y = np.array([], np.float64)
        yerr = np.array([], np.float64)

        for tpf in tpf_collection:
            mask = self.system.create_planet_mask(tpf.time)

            # make sure the pipeline aperture has at least 3 pixels
            if np.sum(tpf._parse_aperture_mask('pipeline'), axis=(0,1)) >= 3:
                aperture_mask = tpf._parse_aperture_mask(self.global_aperture)
            else:
                aperture_mask = tpf._parse_aperture_mask('threshold')
            time, flux, error, gp = Corrector().PyMC_PLD(tpf, self.system, mask, aperture_mask)

            x = np.append(x, time)
            y = np.append(y, flux)
            yerr = np.append(yerr, error)

        lc = lk.LightCurve(time=x, flux=y, flux_err=yerr)

        '''
        Make gp_lc a lk.LightCurve object and set it as a self variable.
        gp_lc = lk.LightCurve(time=x, flux=gp)
        '''

        return lc


    def download_files(self, target_name, n_obs='all'):
        """ """

        search_result = lk.search_targetpixelfile(target_name)
        if n_obs == 'all':
            tpf_collection = search_result.download_all(quality_bitmask='hardest')
        else:
            tpf_collection = search_result[:n_obs].download_all(quality_bitmask='hardest')

        return tpf_collection
