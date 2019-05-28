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

from .timeseries import TimeSeries
from .planetsystem import PlanetSystem
from .utils import silence

__all__ = ['TransitFitter']

class TransitFitter(object):
    """ """

    def __init__(self, target_name=None, ind=None, n_obs='all'):

        self.system = PlanetSystem(target_name=target_name, ind=ind)
        self.target_name = self.system.host
        self.lightcurve = TimeSeries(self.target_name, self.system).return_lightcurve(n_obs)


    def test_fit(self):
        """ """
        model = self.fit(self.lightcurve.time, self.lightcurve.flux, self.lightcurve.flux_err)
        self.preview(model)


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
                                  testval=rprs_prior*self.system.st_rad,
                                  lower=rprs_prior*self.system.st_rad+(3*self.system.st_rad_err2),
                                  upper=rprs_prior*self.system.st_rad+(3*self.system.st_rad_err1),
                                  shape=self.system.n_planets)

                rprs = pm.Deterministic("rprs", r_pl / r_star)

                period = pm.Uniform("period",
                                    testval=period_prior,
                                    lower=period_prior+3*self.system.pl_period_err2,
                                    upper=period_prior+3*self.system.pl_period_err1,
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


    def preview(self, model):
        """Plot the initial fit."""

        with model:
            mean = model.map_soln["mean"]
            static_lc = xo.utils.eval_in_model(model.light_curves, model.map_soln)

        plt.figure(figsize=(10,4))
        plt.plot(model.x, model.y - mean, "k.", label="data")
        for n, l in enumerate(self.system.letters):
            plt.plot(model.x, 1e3 * static_lc[:, n], label="planet {0}".format(l), zorder=100-n)
        plt.xlabel("time [days]")
        plt.ylabel("flux [ppt]")
        plt.title("{} initial fit".format(self.target_name))
        plt.xlim(model.x.min(), model.x.max())
        plt.legend(fontsize=10);
