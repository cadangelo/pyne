#!/usr/bin/env python
import argparse
import ConfigParser
import os

import numpy as np
from pyne.mesh import Mesh, IMeshTag
from pyne.partisn import write_partisn_input, isotropic_vol_source
from pyne.dagmc import discretize_geom, load
from pyne import nucname
from pyne.bins import pointwise_collapse
from pyne.cccc import Atflux
from pyne.variancereduction import cadis
from pyne.mcnp import Wwinp


config_filename = 'config.ini'

config = \
"""
# Optional step to assess all materials in geometry for compatibility with 
# SNILB criteria
[step0]

# Prepare PARTISN input file for adjoint photon transport
[step1]
# Path to hdf5 geometry file
geom_file:
# ID of adjoint photon source cell
src_cell:
# Discretization in x-direction
# xmin, xmax, number of divisions
xmesh:
# Discretization in y-direction
# ymin, ymax, number of divisions
ymesh:
# Discretization in z-direction
# zmin, zmax, number of divisions
zmesh:
# Source intensity
intensities:
# Volume of source cell
src_vol:

# Calculate T matrix for each material
[step2]

# Calculate adjoint neutron source
[step3]

# Prepare PARTISN input for adjoint neutron transport
[step4]

# Generate Monte Carlo variance reduction parameters 
# (biased source and weight windows)
[step5]
# Path to adjoint neutron flux file.
atflux:
# Path to unbiased neutron source mesh file.
q_mesh:

"""



def setup():
    with open(config_filename, 'w') as f:
        f.write(config)
    print('File "{}" has been written'.format(config_filename))
    print('Fill out the fields in this file then run ">> gtcadis.py step1"')

def step5():
    # Parse config file
    config = ConfigParser.ConfigParser()
    config.read(config_filename)
    atflux = config.get('step5', 'atflux')    
    q_mesh = Mesh(structured=True, mesh=config.get('step5', 'q_mesh'))


    # Map atflux values to structured mesh 
    os.system('cp blank_mesh.h5m adj_n_mesh.h5m')
    adj_flux_mesh = Mesh(structured=True, mesh='adj_n_mesh.h5m')
    at = Atflux(atflux)
    at.to_mesh(adj_flux_mesh, "flux")

    adj_flux_mesh.flux = IMeshTag(217, float)
    adj_flux_mesh.flux2 = IMeshTag(175, float)

    a = adj_flux_mesh.flux[:]
    adj_flux_mesh.flux2[:] = a[:,42:]
    adj_flux_tag = "flux2"

    # Create source mesh tags
    q_tag = "source_density"
    q_bias_mesh = q_mesh
    q_bias_tag= "biased_source_density"
    # Create weight window tag
    ww_mesh = adj_flux_mesh
    ww_tag = "ww_n"
    
    # Use CADIS to generate biased source 
    cadis(adj_flux_mesh, adj_flux_tag, q_mesh, q_tag,
              ww_mesh, ww_tag, q_bias_mesh, q_bias_tag, beta=5)
    
    particle = 'n'
    tag_e_bounds = \
        ww_mesh.mesh.createTag('{0}_e_upper_bounds'.format(particle),
                                175, float)
    
    tag_e_bounds[ww_mesh.mesh.rootSet] = [1.00E-07, 4.14E-07, 5.32E-07, 6.83E-07,
    8.76E-07, 1.13E-06, 1.44E-06, 1.86E-06, 2.38E-06, 3.06E-06, 3.93E-06, 5.04E-06,
    6.48E-06, 8.32E-06, 1.07E-05, 1.37E-05, 1.76E-05, 2.26E-05, 2.90E-05, 3.73E-05,
    4.79E-05, 6.14E-05, 7.89E-05, 1.01E-04, 1.30E-04, 1.67E-04, 2.14E-04, 2.75E-04,
    3.54E-04, 4.54E-04, 5.83E-04, 7.49E-04, 9.61E-04, 1.23E-03, 1.58E-03, 2.03E-03,
    2.25E-03, 2.49E-03, 2.61E-03, 2.75E-03, 3.04E-03, 3.35E-03, 3.71E-03, 4.31E-03,
    5.53E-03, 7.10E-03, 9.12E-03, 1.06E-02, 1.17E-02, 1.50E-02, 1.93E-02, 2.19E-02,
    2.36E-02, 2.42E-02, 2.48E-02, 2.61E-02, 2.70E-02, 2.85E-02, 3.18E-02, 3.43E-02,
    4.09E-02, 4.63E-02, 5.25E-02, 5.66E-02, 6.74E-02, 7.20E-02, 7.95E-02, 8.25E-02,
    8.65E-02, 9.80E-02, 1.11E-01, 1.17E-01, 1.23E-01, 1.29E-01, 1.36E-01, 1.43E-01,
    1.50E-01, 1.58E-01, 1.66E-01, 1.74E-01, 1.83E-01, 1.93E-01, 2.02E-01, 2.13E-01,
    2.24E-01, 2.35E-01, 2.47E-01, 2.73E-01, 2.87E-01, 2.95E-01, 2.97E-01, 2.98E-01,
    3.02E-01, 3.34E-01, 3.69E-01, 3.88E-01, 4.08E-01, 4.50E-01, 4.98E-01, 5.23E-01,
    5.50E-01, 5.78E-01, 6.08E-01, 6.39E-01, 6.72E-01, 7.07E-01, 7.43E-01, 7.81E-01,
    8.21E-01, 8.63E-01, 9.07E-01, 9.62E-01, 1.00E+00, 1.11E+00, 1.16E+00, 1.22E+00,
    1.29E+00, 1.35E+00, 1.42E+00, 1.50E+00, 1.57E+00, 1.65E+00, 1.74E+00, 1.83E+00,
    1.92E+00, 2.02E+00, 2.12E+00, 2.23E+00, 2.31E+00, 2.35E+00, 2.37E+00, 2.39E+00,
    2.47E+00, 2.59E+00, 2.73E+00, 2.87E+00, 3.01E+00, 3.17E+00, 3.33E+00, 3.68E+00,
    4.07E+00, 4.49E+00, 4.72E+00, 4.97E+00, 5.22E+00, 5.49E+00, 5.77E+00, 6.07E+00,
    6.38E+00, 6.59E+00, 6.70E+00, 7.05E+00, 7.41E+00, 7.79E+00, 8.19E+00, 8.61E+00,
    9.05E+00, 9.51E+00, 1.00E+01, 1.05E+01, 1.11E+01, 1.16E+01, 1.22E+01, 1.25E+01,
    1.28E+01, 1.35E+01, 1.38E+01, 1.42E+01, 1.46E+01, 1.49E+01, 1.57E+01, 1.65E+01,
    1.69E+01, 1.73E+01, 1.96E+01]
    
    wwinp = Wwinp()
    wwinp.read_mesh(ww_mesh.mesh)
    
    wwinp.mesh.save("wwinp.h5m")
    q_mesh.mesh.save("biased_source.h5m")
    wwinp.write_wwinp("wwinp.out")

def main():

    gtcadis_help = ('This script automates the GT-CADIS process of \n'
                    'producing variance reduction parameters to optimize the\n'
                    'neutron transport step of the Rigorous 2-Step (R2S) method.\n')
    setup_help = ('Prints the file "config.ini" to be\n'
                  'filled in by the user.\n')
    step5_help = 'Creates the weight windows and biased source.'
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help=gtcadis_help, dest='command')

    setup_parser = subparsers.add_parser('setup', help=setup_help)
    step5_parser = subparsers.add_parser('step5', help=step5_help)

    args, other = parser.parse_known_args()
    if args.command == 'setup':
        setup()
    elif args.command == 'step5':
        step5()

if __name__ == '__main__':
    main()
