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


class Corrector(object):
    """ """

    def __init__(self):
        '''
        Take tpf in constructor
        call `correct` from `Corrector` class
        '''
        pass

    def PyMC_PLD(self, tpf, system, planet_mask, aperture, sigma=5, ndraws=1000, pld_order=3, n_pca_terms=10, tlsing=False):
        """ """

        time = np.asarray(tpf.time, np.float64)
        flux = np.asarray(tpf.flux, np.float64)
        flux_err = np.asarray(tpf.flux_err, np.float64)
        aper = np.asarray(aperture, bool)

        raw_flux = np.asarray(np.nansum(flux[:, aper], axis=(1)),  np.float64)
        raw_flux_err = np.asarray(np.nansum(flux_err[:, aper]**2, axis=(1))**0.5,  np.float64)

        raw_flux_err /= np.median(raw_flux)
        raw_flux /= np.median(raw_flux)
        raw_flux -= 1

        # create nan mask
        nanmask = np.isfinite(tpf.time)
        nanmask &= np.isfinite(raw_flux)
        nanmask &= np.isfinite(raw_flux_err)
        nanmask &= np.abs(raw_flux_err) > 1e-12

        # mask out nan values
        raw_flux = raw_flux[nanmask]
        raw_flux_err = raw_flux_err[nanmask]
        flux = flux[nanmask]
        flux_err = flux_err[nanmask]
        time = time[nanmask]

        # Setting to Parts Per Thousand keeps us from hitting machine precision errors...
        raw_flux *= 1e3
        raw_flux_err *= 1e3

        # Build the first order PLD basis
        saturation = (np.nanpercentile(flux, 100, axis=0) > 175000)
        X1 = np.reshape(flux[:, aper & ~saturation], (len(tpf.flux), -1))

        extra_pld = np.zeros((len(time), np.any(saturation, axis=0).sum()))
        idx = 0
        for column in saturation.T:
            if column.any():
                extra_pld[:, idx] = np.sum(flux[:, column, :], axis=(1, 2))
                idx += 1
        X1 = np.hstack([X1, extra_pld])

        # Remove NaN pixels
        X1 = X1[:, ~((~np.isfinite(X1)).all(axis=0))]
        X1 = X1 / np.sum(flux[:, aper], axis=-1)[:, None]

        # higher order PLD design matrices
        X_sections = [X1]
        for i in range(2, pld_order+1):
            f2 = np.product(list(multichoose(X1.T, pld_order)), axis=1).T

            components, _, _ = pca(f2, n_pca_terms)
            X_n = components[:, :n_pca_terms]
            X_sections.append(X_n)

        X_pld = np.concatenate(X_sections, axis=1)

        def build_model(mask=None, start=None):
            ''' Build a PYMC3 model

            Parameters
            ----------
            mask : np.ndarray
                Boolean array to mask cadences. Cadences that are False will be excluded
                from the model fit
            start : dict
                MAP Solution from exoplanet

            Returns
            -------
            model : pymc3.model.Model
                A pymc3 model
            map_soln : dict
                Best fit solution
            '''

            if mask is None:
                mask = np.zeros(len(time), dtype=bool)

            with pm.Model() as model:
                # GP
                # --------
                diag = np.array(raw_flux_err[~mask]**2)
                # diag[mask] += 1e12 # Temporarily increase the in-transit error bars
                logs2 = pm.Normal("logs2", mu=np.log(np.var(raw_flux[~mask])), sd=4)
                logsigma = pm.Normal("logsigma", mu=np.log(np.std(raw_flux[~mask])), sd=4)
                logrho = pm.Normal("logrho", mu=np.log(150), sd=4)

                kernel = xo.gp.terms.Matern32Term(log_sigma=logsigma, log_rho=logrho)
                gp = xo.gp.GP(kernel, time[~mask], diag + tt.exp(logs2))

                # Motion model
                #------------------
                A = tt.dot(X_pld[~mask].T, gp.apply_inverse(X_pld[~mask]))
                B = tt.dot(X_pld[~mask].T, gp.apply_inverse(raw_flux[~mask][:, None]))
                C = tt.slinalg.solve(A, B)
                motion_model = pm.Deterministic("motion_model", tt.dot(X_pld[~mask], C)[:, 0])

                # Likelihood
                #------------------
                pm.Potential("obs", gp.log_likelihood(raw_flux[~mask] - motion_model))

                # gp predicted flux
                gp_pred = gp.predict()
                pm.Deterministic("gp_pred", gp_pred)
                pm.Deterministic("weights", C)

                # Optimize
                #------------------
                if start is None:
                    start = model.test_point
                map_soln = xo.optimize(start=start, vars=[logsigma])
                map_soln = xo.optimize(start=start, vars=[logrho, logsigma])
                map_soln = xo.optimize(start=start, vars=[logsigma])
                map_soln = xo.optimize(start=start, vars=[logrho, logsigma])
                map_soln = xo.optimize(start=map_soln, vars=[logs2])
                map_soln = xo.optimize(start=map_soln, vars=[logrho, logsigma, logs2])
                return model, map_soln, gp

        # Compute the transit mask
        mask = system.create_planet_mask(time)

        # First rough correction
        with silence():
            model0, map_soln0, gp = build_model(mask=mask)

        # Remove outliers, make sure to remove a few nearby points incase of flares.
        with model0:
            motion = np.dot(X_pld, map_soln0['weights']).reshape(-1)
            stellar = xo.eval_in_model(gp.predict(time), map_soln0)
            corrected = raw_flux - motion - stellar

        '''
        # Optimize PLD
        with silence():
            model, map_soln, gp = build_model(mask, map_soln0)

        # Burn in
        sampler = xo.PyMC3Sampler()
        with model:
            burnin = sampler.tune(tune=np.max([int(ndraws*0.3), 150]), start=map_soln,
                                  step_kwargs=dict(target_accept=0.9),
                                  chains=4)
        # Sample
        with model:
            trace = sampler.sample(draws=ndraws, chains=4)

        varnames = ["logrho", "logsigma", "logs2"]
        pm.traceplot(trace, varnames=varnames);

        samples = pm.trace_to_dataframe(trace, varnames=varnames)
        corner.corner(samples);

        # Generate 50 realizations of the prediction sampling randomly from the chain
        N_pred = 50
        pred_mu = np.empty((N_pred, len(time)))
        pred_motion = np.empty((N_pred, len(time)))
        with model:
            pred = gp.predict(time)
            for i, sample in enumerate(tqdm(xo.get_samples_from_trace(trace, size=N_pred), total=N_pred)):
                pred_mu[i] = xo.eval_in_model(pred, sample)
                pred_motion[i, :] = np.dot(X_pld, sample['weights']).reshape(-1)

        star_model = np.mean(pred_mu + pred_motion, axis=0)
        star_model_err = np.std(pred_mu + pred_motion, axis=0)
        corrected = raw_flux - star_model

        # remove outliers and store the mask if it's the first fit
        if not tlsing:
            time, corrected, raw_flux_err, mask = self.remove_outliers(time, corrected, raw_flux_err, mask)
            self.old_mask = mask
        '''
        return time, corrected, raw_flux_err, gp
        # else:
        #     return time, corrected, raw_flux_err, gp
