from pdb import set_trace as pdb

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

from .correct import Corrector
from .planetsystem import PlanetSystem, create_planet_system
from .plotting import preview, pdf_summary
from .utils import silence

__all__ = ['TransitFitter']

class TransitFitter(object):
    """ """

    def __init__(self, target_name=None, ind=None, aperture_mask='pipeline', n_obs='all'):

        self.target_name = target_name
        self.aperture_mask = aperture_mask

        self.system = create_planet_system(target_name)
        self.lightcurve, self.tpf_collection = generate_light_curve(self.target_name, self.system,
                                                                    aperture_mask=aperture_mask,
                                                                    n_obs=n_obs)
        self._initial_fit_generated = False


    def test_fit(self):
        """Optimize parameters with PyMC3 and plot initial fit."""
        self.model = self.fit(self.lightcurve.time, self.lightcurve.flux, self.lightcurve.flux_err)
        self._initial_fit_generated = True
        preview(self.model, self.system, self.target_name)


    def full_fit(self):
        """Optimize parameters, sample posteriors, and generate a pdf summary of
        results."""
        if not self._initial_fit_generated:
            self.model = self.fit(self.lightcurve.time, self.lightcurve.flux, self.lightcurve.flux_err)
        model, light_curves, trace = self.sample(self.model)
        
        pdf_summary(model, light_curves, trace, self.system, self.aperture_mask,
                    self.tpf_collection, self.target_name)

    def fit(self, x, y, yerr):
        """ """

        # build_model should only take lc and system
        # model = build_model(lc, system)

        def build_model(x, y, yerr, period_prior, t0_prior, rprs_prior, start=None):
            """Build an exoplanet model for a dataset and set of planets

            Args:
                x: The time series (in days); this should probably be centered
                y: The relative fluxes (in parts per thousand)
                yerr: The uncertainties on ``y``
                period_prior: The periods of the planets (in days)
                t0_prior: The phases of the planets in the same coordinates as ``x``
                depths: The depths of the transits in parts per thousand
                start: A dictionary of model parameters where the optimization
                    should be initialized

            Returns:
                A PyMC3 model specifying the probabilistic model for the light curve

            """

            period_prior = np.atleast_1d(period_prior)
            t0_prior = np.atleast_1d(t0_prior)
            rprs_prior = np.atleast_1d(rprs_prior)

            with pm.Model() as model:

                # Set model variables
                model.x = x
                model.y = y
                model.yerr = (yerr + np.zeros_like(x))

                # The baseline (out-of-transit) flux for the star in ppt
                mean = pm.Normal("mean",
                                 mu=0.0,
                                 sd=10.0)

                r_star = pm.Normal("r_star",
                                   mu=self.system.st_rad,
                                   sd=self.system.st_rad_err1)

                m_star = pm.Normal("m_star",
                                   mu=self.system.st_mass,
                                   sd=self.system.st_mass_err1)

                # Prior to require physical parameters
                pm.Potential("r_star_prior",
                             tt.switch(r_star > 0, 0, -np.inf))

                # The time of a reference transit for each planet
                t0 = pm.Normal("t0",
                               mu=t0_prior,
                               sd=self.system.pl_t0_err1,
                               shape=self.system.n_planets)

                # quadratic limb darkening paramters
                u = xo.distributions.QuadLimbDark("u")

                # Orbital parameters
                b = pm.Uniform("b",
                               lower=0,
                               upper=1,
                               testval=0.5,
                               shape=self.system.n_planets)

                r_pl = pm.Uniform("r_pl",
                                  testval=self.system.rprs,
                                  lower=self.system.rprs+(10*self.system.rprs_err2),
                                  upper=self.system.rprs+(10*self.system.rprs_err1),
                                  shape=self.system.n_planets)

                rprs = pm.Deterministic("rprs", r_pl / r_star)

                period = pm.Uniform("period",
                                    testval=period_prior,
                                    lower=period_prior+(2*self.system.pl_period_err2),
                                    upper=period_prior+(2*self.system.pl_period_err1),
                                    shape=self.system.n_planets)

                # factor * 10**logg / r_star = rho
                # factor = 5.141596357654149e-05
                # rho_star = pm.Deterministic("rho_star", factor * 10**logg_star / r_star)
                # logg_star = pm.Normal("logg_star", mu=logg, sd=logg_err)

                # Set up a Keplerian orbit for the planets
                model.orbit = xo.orbits.KeplerianOrbit(
                    period=period, t0=t0, b=b, r_star=r_star, m_star=m_star)# rho_star=rho_star)

                # Compute the model light curve using starry
                model.light_curves = xo.StarryLightCurve(u).get_light_curve(
                                        orbit=model.orbit, r=r_pl, t=model.x)

                model.light_curve = pm.math.sum(model.light_curves, axis=-1) * 1e3 + mean

                # Jitter and likelihood function
                logs2 = pm.Normal("logs2",
                                  mu=np.log(np.mean(model.yerr)),
                                  sd=10)

                pm.Normal("obs",
                          mu=model.light_curve,
                          sd=tt.sqrt(model.yerr**2+tt.exp(logs2)),
                          observed=model.y)

                # Fit for the maximum a posteriori parameters, I've found that I can get
                # a better solution by trying different combinations of parameters in turn
                if start is None:
                    start = model.test_point
                map_soln = xo.optimize(start=start, vars=[period, t0])
                map_soln = xo.optimize(start=map_soln, vars=[rprs, mean])
                map_soln = xo.optimize(start=map_soln, vars=[r_star])
                map_soln = xo.optimize(start=map_soln, vars=[period, t0, mean])
                map_soln = xo.optimize(start=map_soln, vars=[rprs, mean])
                map_soln = xo.optimize(start=map_soln)
                model.map_soln = map_soln

            return model

        # build our initial model and store a static version of the output for plotting
        model = build_model(x, y, yerr, self.system.pl_period, self.system.pl_t0, self.system.rprs)
        with model:
            mean = model.map_soln["mean"]
            static_lc = xo.utils.eval_in_model(model.light_curves, model.map_soln)

        return model


    def sample(self, model, ndraws=1000):
        """Sample the posterior from optimization."""
        # sample the model for our posterior parameters
        np.random.seed(42)
        sampler = xo.PyMC3Sampler()
        with model:
            burnin = sampler.tune(tune=250, start=model.map_soln, step_kwargs=dict(target_accept=0.9))
            trace = sampler.sample(draws=ndraws)

        # store outputs and save summary as a csv
        output_dir = os.path.join(PACKAGEDIR, os.pardir, 'outputs', self.target_name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        out = pm.summary(trace)
        out.to_csv(os.path.join(output_dir, '{}_fit_parameters.csv'.format(self.target_name)))

        # compute fits for out light curves from the sampled parameters
        periods = [self.system.pl_period]
        with model:
            light_curves = np.empty((500, len(model.x), len(self.system.pl_period)))
            func = xo.utils.get_theano_function_for_var(model.light_curves)
            for i, sample in enumerate(xo.utils.get_samples_from_trace(
                    trace, size=len(light_curves))):
                light_curves[i] = func(*xo.utils.get_args_for_theano_function(sample))

        return model, light_curves, trace


def generate_light_curve(target_name, system, aperture_mask='pipeline', n_obs=1):
    """Downloads target pixel files, detrends, and returns a LightCurve object.

    Parameters
    ----------
    `target_name` : str
        Name of host star for the desired system.
    `system` : happyplanets.System
        A System object containing host star and planets.
    `aperture_mask` : str or array-like
        Aperture mask used to generate light curve.
    `n_obs` : int or 'all'
        Number of observations to download, detrend, and stitch together. 1 by
        default.
    """

    search_result = lk.search_targetpixelfile(target_name)
    if n_obs == 'all':
        tpf_collection = search_result.download_all(quality_bitmask='hardest')
    else:
        tpf_collection = search_result[:n_obs].download_all(quality_bitmask='hardest')

    # assign variables
    x = np.array([], np.float64)
    y = np.array([], np.float64)
    yerr = np.array([], np.float64)

    corrector = Corrector()

    for tpf in tpf_collection:
        planet_mask = system.create_planet_mask(tpf.time)

        # make sure the pipeline aperture has at least 3 pixels
        if np.sum(tpf._parse_aperture_mask('pipeline'), axis=(0,1)) >= 3:
            aperture_mask = tpf._parse_aperture_mask(aperture_mask)
        else:
            warnings.warn('Pipeline aperture contains fewer than 3 pixels.'
                          'Using threshold aperture mask instead.')
            aperture_mask = tpf._parse_aperture_mask('threshold')
        time, flux, error, gp = corrector.PyMC_PLD(tpf, system, planet_mask, aperture_mask)

        x = np.append(x, time)
        y = np.append(y, flux)
        yerr = np.append(yerr, error)

    lc = lk.LightCurve(time=x, flux=y, flux_err=yerr)
    return lc, tpf_collection
