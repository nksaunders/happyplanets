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

from .correct import PyMC_PLD


class LightCurve(object):

    def __init__(self):
        ''' '''
        pass


    def download_files(target_name):
        ''' '''

        search_result = lk.search_targetpixelfile(target_name)
        tpf_collection = search_result.download_all(quality_bitmask='hardest')

        return tpf_collection


    def detrend(self, tpf_collection):
        ''' '''

        for tpf in tpf_collection:
            mask = np.zeros_like(tpf.time, dtype=bool)
            for i in range(self.n_planets):
                mask |= self.get_transit_mask(tpf.time, self.pl_t0[i], self.pl_period[i], 0.3)
            # make sure the pipeline aperture has at least 3 pixels
            if np.sum(tpf._parse_aperture_mask('pipeline'), axis=(0,1)) >= 3:
                aperture_mask = tpf._parse_aperture_mask(self.global_aperture)
            else:
                aperture_mask = tpf._parse_aperture_mask('threshold')
            time, flux, error, gp = self.PyMC_PLD(tpf, mask, aperture_mask)

            x = np.append(x, time)
            y = np.append(y, flux)
            yerr = np.append(yerr, error)

        return x, y, yerr, gp
