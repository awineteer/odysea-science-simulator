# odysea-science-simulator



<img src="https://github.com/awineteer/odysea-science-simulator/blob/main/examples/winds_daily_globe.png" align="right"
     alt="Daily Wind Sampling" width="250" height="250">
     
<img src="https://github.com/awineteer/odysea-science-simulator/blob/main/examples/currents_daily_globe.png" align="right"
     alt="Daily Current Sampling" width="250" height="250">
     
Simulation tools for the ODYSEA winds and currents mission. For more information about ODYSEA, see: https://odysea.ucsd.edu/

This set of tools is in active development as are the scope and capabilities for the ODYSEA mission. See the examples folder for examples of usage.

## ODYSEA
ODYSEA (Ocean DYnamics and Surface Exchange with the Atmosphere):  A revolutionary look at winds and surface currents

The ODYSEA satellite will bring into focus daily global surface currents and their interactions with winds to explore the Earth system and to improve weather and climate predictions.

## Installation

Installation is completed via pip. Clone this repository and navigate to odysea-science-simulator/ before issuing:

>pip install .

## Dependencies

>xarray,
netCDF4,
scipy,
numpy,
pandas,
pyyaml,
cartopy (for plotting only)


## What do these tools do?

These tools are primarily designed to simulate ODYSEA level 2 wind and current swath data. A pre-generated orbit nadir track is provided, along which this code will generate a swath of data. Also provided are templates for co-locating ocean/atmosphere model output to these swath data and functions for generating expected measurement uncertainties.

| ![A single swath of U-direction currents](https://github.com/awineteer/odysea-science-simulator/blob/main/examples/swath_currents.png) | 
|:--:| 
| *A single swath of U-direction currents* |

| ![Two orbits of currents, projected on a global map.](https://github.com/awineteer/odysea-science-simulator/blob/main/examples/projected_2_orbits.png) | 
|:--:| 
| *Two orbits of currents, projected on a global map.* |

## What do these tools not do?

While a swath is generated, the more complicated radar pulse and look geometries are not considered.

## To-do list:

- Update these tools as the mission evolves.
- Add measurement errors due to pointing uncertainties. 
- Add geophysical model function errors.
- Add more realistic/complex wind measurement errors.

## Configuration options

- The wacm_config.py file in /odysim/ contains important configuration parameters. In particular, the swath width is set.
- The /odysim/orbit_files/ folder contains orbit nadir tracks. These files can be specified as desired when calling OdyseaSwath(), but the default is a 590 km sun-synchronous orbit at 4AM/PM.
- The /odysim/uncertainty_tables/ folder contains a lookup table for expected measurement errors. This is the most in-flux part of the mission, and will be updated as performance is solidified. Check back often!

## Examples:


See [Here for basic swath generation.](https://github.com/awineteer/odysea-science-simulator/blob/main/examples/colocating_models_and_uncertainties_to_odysea_orbital_swaths.ipynb)

See [Here to co-locate models, generate uncertainties.](https://github.com/awineteer/odysea-science-simulator/blob/main/examples/colocating_models_and_uncertainties_to_odysea_orbital_swaths.ipynb)

See [Here to generate the logos/plots above.](https://github.com/awineteer/odysea-science-simulator/blob/main/examples/make_globe_maps.ipynb)