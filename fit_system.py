import os.path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import lightkurve as lk

import corner
import pymc3 as pm
import exoplanet as xo
import theano.tensor as tt
import astropy.units as unit

# read in CSV file with planet parameters
kois = pd.read_csv('data/planets_2019.04.02_11.43.16.csv', skiprows=range(81));

# set target name and find indices
target_name = 'K2-80'
ind = np.where(kois['pl_hostname'] == target_name)[0]

# store planet parameters
host = np.atleast_1d(kois['pl_hostname'][ind])[0]
pl_period = np.array(kois['pl_orbper'][ind], dtype=float)
pl_t0 = np.array(kois['pl_tranmid'][ind], dtype=float) - 2454833 # JD -> BKJD
r_ratio = np.array([(kois['pl_radj'][i]*unit.jupiterRad / kois['st_rad'][i]*unit.solRad).value for i in ind], dtype=float)

# store stellar parameters
logg = np.atleast_1d(kois['st_logg'][ind])[0]
logg_err = np.atleast_1d(kois['st_loggerr1'][ind])[0]
rad_star = np.atleast_1d(kois['st_rad'][ind])[0]
rad_star_err = np.atleast_1d(kois['st_raderr1'][ind])[0]

# array of planet labels
n_planets = len(pl_period)
letters = "bcdefghijklmnopqrstuvwxyz"[:n_planets]

def get_transit_mask(t, t0, period, duration):
    """Return cadences in t during transit given t0, period, duration."""
    hp = 0.5*period
    return np.abs((t-t0+hp) % period - hp) < 0.5*duration

# download tpf
# NOTE: only the first tpf is downloaded for now
tpf_collection = lk.search_targetpixelfile(host)[0].download_all(quality_bitmask='hardest')

# Run PLD on each TPF to extract the light curves
lc_collection = []
for tpf in tpf_collection:
    mask = np.zeros_like(tpf.time, dtype=bool)
    for i in range(len(pl_period)):
        mask |= get_transit_mask(tpf.time, pl_t0[i], pl_period[i], 0.7)
    pld = tpf.to_corrector("pld")
    lc = pld.correct(aperture_mask="pipeline", cadence_mask=~mask, use_gp=False, pld_order=2)
    lc_collection.append(lc.normalize())

# Normalize and stitch the sectors
lc = lc_collection[0]
if len(lc_collection) > 1:
    lc = lc.append([next_lc for next_lc in lc_collection[1:]])

# Remove outliers
_, outliers = lc.flatten(mask=mask).remove_outliers(return_mask=True)
outliers[mask] = False
mask = mask[~outliers]
lc = lc[~outliers]

# Extract the data and convert to parts per thousand
x = np.ascontiguousarray(lc.time, dtype=np.float64)
y = np.ascontiguousarray((lc.flux - 1.0) * 1e3, dtype=np.float64)
yerr = np.ascontiguousarray(lc.flux_err * 1e3, dtype=np.float64)

# Temporarily increase the in transit error bars substantially
diag = np.array(yerr**2)
diag[mask] += 10000.0

# Build a GP model to remote long-term variability
with pm.Model() as model:
    logs2 = pm.Normal("logs2", mu=np.log(1e-4*np.var(y)), sd=10)
    logsigma = pm.Normal("logsigma", mu=np.log(np.std(y)), sd=10)
    logrho = pm.Normal("logrho", mu=np.log(150.0), sd=10.0)

    kernel = xo.gp.terms.Matern32Term(log_sigma=logsigma, log_rho=logrho)
    gp = xo.gp.GP(kernel, x, diag + tt.exp(logs2), J=0)
    pm.Potential("loglike", gp.log_likelihood(y))

    map_soln = xo.optimize()
    pred = xo.utils.eval_in_model(gp.predict(), map_soln)

# Flatten the light curve
y -= pred

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

        # The baseline (out-of-transit) flux for the star in ppt. This
        # should be close to one because of how we normalized the data
        mean = pm.Normal("mean", mu=0.0, sd=10.0)
        logg_star = pm.Normal("logg_star", mu=logg, sd=logg_err)
        r_star = pm.Normal("r_star", mu=rad_star, sd=rad_star_err)

        # Prior to require physical parameters
        pm.Potential("r_star_prior", tt.switch(r_star > 0, 0, -np.inf))

        # The time of a reference transit for each planet
        t0 = pm.Normal("t0", mu=t0s, sd=1.0, shape=n_planets)

        # The Kipping (2013) parameterization for quadratic limb darkening paramters
        u = xo.distributions.QuadLimbDark("u")

        # Orbital parameters
        b = pm.Uniform("b", lower=0, upper=1, testval=0.5, shape=n_planets)
        logr = pm.Normal("logr", sd=1.0, mu=0.5*np.log(r_ratio[0])+np.log(rad_star), shape=n_planets)
        r_pl = pm.Deterministic("r_pl", tt.exp(logr))
        rprs = pm.Deterministic("rprs", r_pl / r_star) # rp/rs
        logP = pm.Normal("logP", mu=np.log(periods), sd=0.1, shape=n_planets)
        period = pm.Deterministic("period", tt.exp(logP))

        # This shouldn't make a huge difference, but I like to put a uniform
        # prior on the *log* of the radius ratio instead of the value. This
        # can be implemented by adding a custom "potential" (log probability).
        pm.Potential("r_prior", -pm.math.log(rprs))

        # factor * 10**logg / r_star = rho
        factor = 5.141596357654149e-05
        rho_star = pm.Deterministic("rho_star", factor * 10**logg_star / r_star)

        # Set up a Keplerian orbit for the planets
        model.orbit = xo.orbits.KeplerianOrbit(
            period=period, t0=t0, b=b, r_star=r_star, rho_star=rho_star)

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
        map_soln = start
        map_soln = xo.optimize(start=map_soln, vars=[logs2, mean])
        map_soln = xo.optimize(start=map_soln, vars=[rprs, mean])
        map_soln = xo.optimize(start=map_soln, vars=[logg_star])
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
    light_curves = np.empty((500, len(model.x), len(periods)))
    func = xo.utils.get_theano_function_for_var(model.light_curves)
    for i, sample in enumerate(xo.utils.get_samples_from_trace(
            trace, size=len(light_curves))):
        light_curves[i] = func(*xo.utils.get_args_for_theano_function(sample))

# save informative figures to a pdf summary
with PdfPages('{}/{}_summary.pdf'.format(target_name, target_name)) as pdf:

    '''Plot the tpf.'''
    plt.figure(figsize=(3,3))
    tpf.plot(aperture_mask='pipeline')
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
        inds = np.arange(len(periods)) != n
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
    # Convert to Earth radii
    '''
    plt.figure(figsize=(7,7))
    r_pl = trace["r_pl"] * 109.07637070600963
    samples = np.concatenate(([trace['r_pl']], [trace["b"]]), axis=0).T

    labels = ["$R_{{\mathrm{{Pl}},{0}}}$ [$R_\oplus$]".format(i) for i in letters]
    labels += ["impact param {0}".format(i) for i in letters]

    corner.corner(samples, labels=labels,
                  show_titles=True, title_kwargs=dict(fontsize=10));
    pdf.savefig()
    plt.close()
    '''
    '''Orbital parameters 2 corner plot.'''
    ''''
    plt.figure(figsize=(7,7))
    labels = ["$P_{{{0}}}$ [days]".format(i) for i in letters]
    labels += ["$t0_{{{0}}}$ [TBJD]".format(i) for i in letters]
    samples = np.concatenate((trace["period"], trace["t0"]), axis=-1)
    corner.corner(samples, labels=labels,
                  show_titles=True, title_fmt=".5f",
                  title_kwargs=dict(fontsize=10));
    pdf.savefig()
    plt.close()
    '''

    '''Stellar parameters corner plot.'''
    '''
    plt.figure(figsize=(7,7))
    labels = ["$R_\mathrm{star}$ [$R_\odot$]", "$\log g$ [cm/s$^2$]",
              r"$M_\mathrm{star}$ [$\rho_\odot$]"]
    samples = np.vstack((trace["r_star"], trace["logg_star"], trace["rho_star"])).T
    corner.corner(samples, labels=labels,
                  show_titles=True,
                  title_kwargs=dict(fontsize=10));

    pdf.savefig()
    plt.close()
    '''
