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

def preview(model, system, target_name):
    """Plot the initial fit."""

    with model:
      mean = model.map_soln["mean"]
      static_lc = xo.utils.eval_in_model(model.light_curves, model.map_soln)

    plt.figure(figsize=(10,4))
    plt.plot(model.x, model.y - mean, "k.", label="data")
    for n, l in enumerate(system.letters):
      plt.plot(model.x, 1e3 * static_lc[:, n], label="planet {0}".format(l), zorder=100-n)
    plt.xlabel("time [days]")
    plt.ylabel("flux [ppt]")
    plt.title("{} initial fit".format(target_name))
    plt.xlim(model.x.min(), model.x.max())
    plt.legend(fontsize=10);


def pdf_summary(model, light_curves, trace, system, aperture_mask, tpf_collection, target_name):

    with model:
        mean = model.map_soln["mean"]
        static_lc = xo.utils.eval_in_model(model.light_curves, model.map_soln)

    output_dir = os.path.join(PACKAGEDIR, os.pardir, 'outputs', target_name)
    out_fname = os.path.join(output_dir, '{}_summary.pdf'.format(target_name))
    with PdfPages(out_fname) as pdf:
        print('Generating summary for target {}.'.format(target_name))

        '''Plot the tpf.'''
        tpf = tpf_collection[0]
        plt.figure(figsize=(3,3))
        tpf.plot(aperture_mask=aperture_mask)
        plt.title('TPF')
        pdf.savefig()
        plt.close()

        '''Plot the raw flux light curve'''
        lc = tpf.to_lightcurve(aperture_mask=aperture_mask)
        lc.scatter()
        pdf.savefig()
        plt.close()

        '''Plot the initial fit.'''
        plt.figure(figsize=(10,4))
        plt.plot(model.x, model.y - mean, "k.", label="data")
        for n, l in enumerate(system.letters):
          plt.plot(model.x, 1e3 * static_lc[:, n], label="planet {0}".format(l), zorder=100-n)
        plt.xlabel("time [days]")
        plt.ylabel("flux [ppt]")
        plt.title("initial fit")
        plt.xlim(model.x.min(), model.x.max())
        plt.legend(fontsize=10);
        pdf.savefig()
        plt.close()

        '''Plot each of the individual transit fits.'''
        for n, letter in enumerate(system.letters):
            plt.figure(figsize=(10,4))

            # Compute the GP prediction
            mean_mod = np.median(trace["mean"][:, None])

            # Get the posterior median orbital parameters
            p = np.median(trace["period"][:, n])
            t0 = np.median(trace["t0"][:, n])

            # Compute the median of posterior estimate of the contribution from
            # the other planet. Then we can remove this from the data to plot
            # just the planet we care about.
            inds = np.arange(len(system.pl_orbper)) != n
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
        labels = ["$R_{{\mathrm{{Pl}},{0}}}$ [$R_\oplus$]".format(i) for i in system.letters]
        labels += ["impact param {0}".format(i) for i in system.letters]

        corner.corner(samples, labels=labels,
                    show_titles=True, title_kwargs=dict(fontsize=10));
        pdf.savefig()
        plt.close()

        '''Orbital parameters 2 corner plot.'''
        plt.figure(figsize=(7,7))

        labels = ["$P_{{{0}}}$ [days]".format(i) for i in system.letters]
        labels += ["$t0_{{{0}}}$ [TBJD]".format(i) for i in system.letters]
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

        '''Plot fit vs literature parameters in a table.
        plt.figure(figsize=(7,7))
        '''
