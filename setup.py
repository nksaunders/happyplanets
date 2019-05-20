import os
import sys
from setuptools import setup

setup(name='uniformplanets',
      version='0.0.1',
      description="Uniform Analysis of K2 Transiting Exoplanets",
      long_description="A python package to perform uniform parameter "
                       "estimates for transiting exoplanets. This software "
                       "utilizes the exoplanet, PyMC3, and lightkurve "
                       "packages.",
      author='Nicholas Saunders',
      author_email='saunders.nk@gmail.com',
      license='MIT',
      package_dir={'uniformplanets':'uniformplanets'},
      packages=['uniformplanets'],
      )
