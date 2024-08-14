#!/usr/bin/env python

from distutils.core import setup

setup(
    name='odysim',
    version='0.1.1',
    description='ODYSEA mission science simulator.',
    author='A. Wineteer',
    author_email='wineteer@jpl.nasa.gov',
    package_dir={'odysea-simulator': 'odysim'},
    packages=[
        'odysim'],
    package_data={'odysim': ['orbit_files/*.npz', 'uncertainty_tables/*.npz','cartopy_files/*']}
)
