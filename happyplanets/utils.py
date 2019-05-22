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

@contextmanager
def silence():
    '''Suppreses all output'''
    logger = logging.getLogger()
    logger.disabled = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout


def download_files(target_name):
    ''' '''

    search_result = lk.search_targetpixelfile(target_name)
    tpf_collection = search_result.download_all(quality_bitmask='hardest')

    return tpf_collection



def read_local_data(target_name=None, ind=None):
    ''' '''
    # read in CSV file with planet parameters
    datadir = os.path.abspath(os.path.join(PACKAGEDIR, os.pardir, 'data'))
    kois = pd.read_csv(os.path.join(datadir, 'planets_2019.04.02_11.43.16.csv'),
                       skiprows=range(81))

    # set target name and find indices
    if ind is None and target_name is None:
        raise ValueError("Please specify either `target_name` or `ind`.")
    if ind is not None:
        target_name = kois['pl_hostname'][ind]

    ind = np.where(kois['pl_hostname'] == target_name)[0]

    return kois.iloc[ind]
