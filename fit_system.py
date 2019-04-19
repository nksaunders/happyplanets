import sys
import os.path
import logging
import warnings
from contextlib import contextmanager
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.table import Table

import corner
import pymc3 as pm
import exoplanet as xo
import theano.tensor as tt
import astropy.units as unit
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box1DKernel
from itertools import combinations_with_replacement as multichoose

def duck(name='the checkpoint', fail=False):
    shut_up_duck = False
    if not shut_up_duck:
        if not fail:
            print('Rubber duck floated past {}.'.format(name))
        else:
            print('The duck is dead.')

# read in CSV file with planet parameters
kois = pd.read_csv('data/planets_2019.04.02_11.43.16.csv', skiprows=range(81));

# set target name and find indices
target_name = 'K2-222'
ind = np.where(kois['pl_hostname'] == target_name)[0]

# store planet parameters
host = np.atleast_1d(kois['pl_hostname'][ind])[0]
pl_period = np.array(kois['pl_orbper'][ind], dtype=float)
pl_period_err = np.array(kois['pl_orbpererr1'][ind], dtype=float)
pl_t0 = np.array(kois['pl_tranmid'][ind], dtype=float) - 2454833 # JD -> BKJD
pl_t0_err = np.array(kois['pl_tranmiderr1'][ind], dtype=float)
r_ratio = np.array([(kois['pl_radj'][i]*unit.jupiterRad / kois['st_rad'][i]*unit.solRad).value for i in ind], dtype=float)

# store stellar parameters
mass_star = np.atleast_1d(kois['st_mass'][ind])[0]
mass_star_err = np.atleast_1d(kois['st_masserr1'][ind])[0]
rad_star = np.atleast_1d(kois['st_rad'][ind])[0]
rad_star_err = np.atleast_1d(kois['st_raderr1'][ind])[0]

# array of planet labels
n_planets = len(pl_period)
letters = "bcdefghijklmnopqrstuvwxyz"[:n_planets]

# set aperture
global_aperture = 'pipeline'

def get_transit_mask(t, t0, period, duration):
    """Return cadences in t during transit given t0, period, duration."""
    hp = 0.5*period
    return np.abs((t-t0+hp) % period - hp) < 0.5*duration

duck('the easy stuff')
# download tpf
# NOTE: only the first tpf is downloaded for now
tpf_collection = lk.search_targetpixelfile(host)[0].download_all(quality_bitmask='hardest')

duck('downloading')

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

def PyMC_PLD(tpf, planet_mask, aperture, sigma=5, ndraws=1000, pld_order=3, n_pca_terms=10):
    """ """
    duck('PyMC PLD being called')

    time = np.asarray(tpf.time, np.float64)
    flux = np.asarray(tpf.flux, np.float64)
    flux_err = np.asarray(tpf.flux_err, np.float64)
    aper = np.asarray(aperture, bool)

    raw_flux = np.asarray(np.nansum(flux[:, aper], axis=(1)),  np.float64)
    raw_flux_err = np.asarray(np.nansum(flux_err[:, aper]**2, axis=(1))**0.5,  np.float64)

    raw_flux_err /= np.median(raw_flux)
    raw_flux /= np.median(raw_flux)
    raw_flux -= 1

    # Setting to Parts Per Thousand keeps us from hitting machine precision errors...
    raw_flux *= 1e6
    raw_flux_err *= 1e6

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

    duck('the first order design matrix with shape {}'.format(X1.shape))
    # higher order PLD design matrices
    X_sections = [X1]
    for i in range(2, pld_order+1):
        duck('higher order pld loop')
        f2 = np.product(list(multichoose(X1.T, pld_order)), axis=1).T
        duck('multichoose')
        try:
            # We use an optional dependency for very fast PCA (fbpca).
            # If the import fails we will fall back on using the slower `np.linalg.svd`
            from fbpca import pca
            duck('fbpca import')
            components, _, _ = pca(f2, n_pca_terms)
        except ImportError:
            duck(fail=True)
            components, _, _ = np.linalg.svd(f2)
        X_n = components[:, :n_pca_terms]
        X_sections.append(X_n)
    duck('the higher order matrices')
    # Create the design matrix X by stacking X1 and higher order components, and
    # adding a column vector of 1s for numerical stability (see Luger et al.).
    # X has shape (n_components_first + n_components_higher_order + 1, n_cadences)
    X_pld = np.concatenate(X_sections, axis=1)
    '''
    # Build the second order PLD basis and run PCA to reduce the number of dimensions
    X2_pld = np.reshape(X_pld[:, None, :] * X_pld[:, :, None], (len(flux), -1))
    # Remove NaN pixels
    X2_pld = X2_pld[:, ~((~np.isfinite(X2_pld)).all(axis=0))]
    U, _, _ = np.linalg.svd(X2_pld, full_matrices=False)
    X2_pld = U[:, :X_pld.shape[1]]

    # Construct the design matrix and fit for the PLD model
    X_pld = np.concatenate((np.ones((len(flux), 1)), X_pld, X2_pld), axis=-1)
    '''

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
            mask = np.ones(len(time), dtype=bool)

        with pm.Model() as model:
            duck('beginning of the model')
            # GP
            # --------
            diag = np.array(raw_flux_err**2)
            diag[mask] += 1e12 # Temporarily increase the in-transit error bars
            logs2 = pm.Normal("logs2", mu=np.log(np.var(raw_flux)), sd=4)
            logsigma = pm.Normal("logsigma", mu=np.log(np.std(raw_flux)), sd=4)
            logrho = pm.Normal("logrho", mu=np.log(30), sd=4)

            kernel = xo.gp.terms.Matern32Term(log_sigma=logsigma, log_rho=logrho)
            gp = xo.gp.GP(kernel, time, diag + tt.exp(logs2))

            # Motion model
            #------------------
            import pdb; pdb.set_trace()
            A = tt.dot(X_pld.T, gp.apply_inverse(X_pld))
            B = tt.dot(X_pld.T, gp.apply_inverse(raw_flux[:, None]))
            C = tt.slinalg.solve(A, B)
            motion_model = pm.Deterministic("motion_model", tt.dot(X_pld, C)[:, 0])

            # Likelihood
            #------------------
            pm.Potential("obs", gp.log_likelihood(raw_flux - motion_model))

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

    # First rough correction
    with silence():
        model0, map_soln0, gp = build_model(mask=planet_mask)

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
    '''

    return time, corrected * 1e-3, raw_flux_err * 1e-3

x = np.array([], np.float64)
y = np.array([], np.float64)
yerr = np.array([], np.float64)
outliers = np.array([], bool)

for tpf in tpf_collection:
    mask = np.zeros_like(tpf.time, dtype=bool)
    for i in range(n_planets):
        mask |= get_transit_mask(tpf.time, pl_t0[i], pl_period[i], 0.7)
    # aperture_mask = tpf._parse_aperture_mask(global_aperture)
    aperture_mask = tpf.pipeline_mask
    time, flux, error = PyMC_PLD(tpf, mask, aperture_mask)
    x = np.append(x, time)
    y = np.append(y, flux)
    yerr = np.append(yerr, error)
    outliers = np.append(outliers, ~sigma_clip(y[mask], sigma=5).mask)

def build_model(x, y, yerr, periods, t0s, r_ratio, mask=None, start=None):
    """Build an exoplanet model for a dataset and set of planets

    Args:
        x: The time series (in days); this should probably be centered
        y: The relative fluxes (in parts per thousand)
        yerr: The uncertainties on ``y``
        periods: The periods of the planets (in days)
        t0s: The phases of the planets in the same coordinates as ``x``
        depths: The depths of the transits in parts per thousand
        mask: A boolean mask with the same shape as ``x`` indicating which
            data points should be included in the fit
        start: A dictionary of model parameters where the optimization
            should be initialized

    Returns:
        A PyMC3 model specifying the probabilistic model for the light curve

    """
    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    periods = np.atleast_1d(periods)
    t0s = np.atleast_1d(t0s)
    r_ratio = np.atleast_1d(r_ratio)

    with pm.Model() as model:

        # Extract the un-masked data points
        model.x = x[mask]
        model.y = y[mask]
        model.yerr = (yerr + np.zeros_like(x))[mask]
        model.mask = mask

        # The baseline (out-of-transit) flux for the star in ppt
        mean = pm.Normal("mean", mu=0.0, sd=10.0)
        # logg_star = pm.Normal("logg_star", mu=logg, sd=logg_err)
        r_star = pm.Normal("r_star", mu=rad_star, sd=rad_star_err)
        m_star = pm.Normal("m_star", mu=mass_star, sd=mass_star_err)

        # Prior to require physical parameters
        pm.Potential("r_star_prior", tt.switch(r_star > 0, 0, -np.inf))

        # The time of a reference transit for each planet
        t0 = pm.Normal("t0", mu=t0s, sd=pl_t0_err, shape=n_planets)

        # quadratic limb darkening paramters
        u = xo.distributions.QuadLimbDark("u")

        # Orbital parameters
        b = pm.Uniform("b", lower=0, upper=1, testval=0.5, shape=n_planets)
        logr = pm.Normal("logr", sd=0.5, mu=np.log(r_ratio)+np.log(rad_star), shape=n_planets)
        r_pl = pm.Deterministic("r_pl", tt.exp(logr))
        rprs = pm.Deterministic("rprs", r_pl / r_star) # rp/rs
        logP = pm.Normal("logP", mu=np.log(periods), sd=pl_period_err, shape=n_planets)
        period = pm.Deterministic("period", tt.exp(logP))

        # factor * 10**logg / r_star = rho
        factor = 5.141596357654149e-05
        # rho_star = pm.Deterministic("rho_star", factor * 10**logg_star / r_star)

        # Set up a Keplerian orbit for the planets
        model.orbit = xo.orbits.KeplerianOrbit(
            period=period, t0=t0, b=b, r_star=r_star, m_star=m_star)# rho_star=rho_star)

        # Compute the model light curve using starry
        model.light_curves = xo.StarryLightCurve(u).get_light_curve(
            orbit=model.orbit, r=r_pl, t=model.x)
        model.light_curve = pm.math.sum(model.light_curves, axis=-1) * 1e3 + mean

        # Jitter and likelihood function
        logs2 = pm.Normal("logs2", mu=np.log(np.mean(model.yerr)), sd=10)
        pm.Normal("obs", mu=model.light_curve, sd=tt.sqrt(model.yerr**2+tt.exp(logs2)),
                  observed=model.y)

        # Fit for the maximum a posteriori parameters, I've found that I can get
        # a better solution by trying different combinations of parameters in turn
        if start is None:
            start = model.test_point
        map_soln = xo.optimize(start=start, vars=[logP, t0])
        map_soln = xo.optimize(start=map_soln, vars=[rprs, mean])
        map_soln = xo.optimize(start=map_soln, vars=[r_star])
        map_soln = xo.optimize(start=map_soln, vars=[logP, t0, mean])
        map_soln = xo.optimize(start=map_soln, vars=[rprs, mean])
        map_soln = xo.optimize(start=map_soln)
        model.map_soln = map_soln

    return model

# build our initial model and store a static version of the output for plotting
model = build_model(x, y, yerr, pl_period, pl_t0, r_ratio)
with model:
    mean = model.map_soln["mean"]
    static_lc = xo.utils.eval_in_model(model.light_curves, model.map_soln)

# sample the model for our posterior parameters
np.random.seed(42)
sampler = xo.PyMC3Sampler()
with model:
    burnin = sampler.tune(tune=250, start=model.map_soln, step_kwargs=dict(target_accept=0.9))
    trace = sampler.sample(draws=1000)

# store outputs and save summary as a csv
if not os.path.isdir(target_name):
    os.mkdir(target_name)
out = pm.summary(trace)
out.to_csv('{}/{}_fit_parameters.csv'.format(target_name, target_name))

# compute fits for out light curves from the sampled parameters
periods = [pl_period]
with model:
    light_curves = np.empty((500, len(model.x), len(pl_period)))
    func = xo.utils.get_theano_function_for_var(model.light_curves)
    for i, sample in enumerate(xo.utils.get_samples_from_trace(
            trace, size=len(light_curves))):
        light_curves[i] = func(*xo.utils.get_args_for_theano_function(sample))

# save informative figures to a pdf summary
with PdfPages('{}/{}_summary.pdf'.format(target_name, target_name)) as pdf:

    '''Plot the tpf.'''
    plt.figure(figsize=(3,3))
    tpf.plot(aperture_mask=global_aperture)
    plt.title('TPF')
    pdf.savefig()
    plt.close()

    '''Plot the initial fit.'''
    plt.figure(figsize=(10,4))
    plt.plot(model.x, model.y - mean, "k.", label="data")
    for n, l in enumerate(letters):
        plt.plot(model.x, 1e3*static_lc[:, n], label="planet {0}".format(l), zorder=100-n)
    plt.xlabel("time [days]")
    plt.ylabel("flux [ppt]")
    plt.title("initial fit")
    plt.xlim(model.x.min(), model.x.max())
    plt.legend(fontsize=10);
    pdf.savefig()
    plt.close()

    '''Plot each of the individual transit fits.'''
    for n, letter in enumerate(letters):
        plt.figure(figsize=(10,4))

        # Compute the GP prediction
        mean_mod = np.median(trace["mean"][:, None])

        # Get the posterior median orbital parameters
        p = np.median(trace["period"][:, n])
        t0 = np.median(trace["t0"][:, n])

        # Compute the median of posterior estimate of the contribution from
        # the other planet. Then we can remove this from the data to plot
        # just the planet we care about.
        inds = np.arange(len(pl_period)) != n
        others = np.median(1e3*np.sum(light_curves[:, :, inds], axis=-1), axis=0)

        # Plot the folded data
        x_fold = (model.x - t0 + 0.5*p) % p - 0.5*p
        plt.plot(x_fold, model.y - mean_mod - others, ".k", label="data", zorder=-1000)

        # Plot the folded model
        inds = np.argsort(x_fold)
        inds = inds[np.abs(x_fold)[inds] < 0.6]
        pred = 1e3 * light_curves[:, inds, n]
        pred = np.percentile(pred, [16, 50, 84], axis=0)
        plt.plot(x_fold[inds], pred[1], color="C1", label="model")
        art = plt.fill_between(x_fold[inds], pred[0], pred[2], color="C1", alpha=0.5,
                               zorder=1000)
        art.set_edgecolor("none")

        # Annotate the plot with the planet's period
        txt = "period = {0:.4f} +/- {1:.4f} d".format(
            np.mean(trace["period"][:, n]), np.std(trace["period"][:, n]))
        plt.annotate(txt, (0, 0), xycoords="axes fraction",
                     xytext=(5, 5), textcoords="offset points",
                     ha="left", va="bottom", fontsize=12)

        plt.legend(fontsize=10, loc=4)
        plt.xlim(-0.5*p, 0.5*p)
        plt.xlabel("time since transit [days]")
        plt.ylabel("de-trended flux")
        plt.title("{} {}".format(target_name, letter));
        plt.xlim(-0.6, 0.6)
        pdf.savefig()
        plt.close()

    '''Orbital parameters corner plot.'''
    plt.figure(figsize=(7,7))

    # Convert to Earth radii
    r_pl = trace["r_pl"] * 109.07637070600963

    samples = np.concatenate((trace["r_pl"], trace["b"]), axis=-1)
    labels = ["$R_{{\mathrm{{Pl}},{0}}}$ [$R_\oplus$]".format(i) for i in letters]
    labels += ["impact param {0}".format(i) for i in letters]

    corner.corner(samples, labels=labels,
                  show_titles=True, title_kwargs=dict(fontsize=10));
    pdf.savefig()
    plt.close()

    '''Orbital parameters 2 corner plot.'''
    plt.figure(figsize=(7,7))

    labels = ["$P_{{{0}}}$ [days]".format(i) for i in letters]
    labels += ["$t0_{{{0}}}$ [TBJD]".format(i) for i in letters]
    samples = np.concatenate((trace["period"], trace["t0"]), axis=-1)

    corner.corner(samples, labels=labels,
                  show_titles=True, title_fmt=".5f",
                  title_kwargs=dict(fontsize=10));
    pdf.savefig()
    plt.close()

    '''Stellar parameters corner plot.'''
    plt.figure(figsize=(7,7))

    labels = ["$R_\mathrm{star}$ [$R_\odot$]", r"$M_\mathrm{star}$ "]
    samples = np.vstack((trace["r_star"], trace["m_star"])).T

    corner.corner(samples, labels=labels,
                  show_titles=True,
                  title_kwargs=dict(fontsize=10));
    pdf.savefig()
    plt.close()
