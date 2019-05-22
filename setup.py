import os
import sys
from setuptools import setup

setup(name='happyplanets',
      version='0.0.1',
      description="Homogeneous Analysis of Parameters with Python: Planets",
      long_description="A python package to perform uniform parameter "
                       "estimates for transiting exoplanets. This software "
                       "utilizes the exoplanet, PyMC3, and lightkurve "
                       "packages.",
      author='Nicholas Saunders',
      author_email='saunders.nk@gmail.com',
      license='MIT',
      package_dir={'happyplanets':'happyplanets'},
      packages=['happyplanets'],
      )
