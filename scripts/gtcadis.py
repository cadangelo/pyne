#!/usr/bin/env python
import argparse
import ConfigParser
from os.path import isfile

import numpy as np
from pyne.mesh import Mesh, IMeshTag
from pyne.partisn import write_partisn_input, isotropic_vol_source
from pyne.dagmc import discretize_geom, load, cell_material_assignments
from pyne import nucname
from pyne.material import MaterialLibrary
from pyne.bins import pointwise_collapse
from pyne.alara import calc_T


config_filename = 'config.ini'

config = \
"""
## Optional step to assess all materials in geometry for compatibility with 
# SNILB criteria
[step0]

## Prepare PARTISN input file for adjoint photon transport
[step1]

## Calculate T matrix for each material
[step2]
# Path to hdf5 geometry file 
# (same as given in step 1, if running step 1)
geom_file: 
# Path to ALARA
alara_dir: 
# Single pulse irradiation time
irr_time:
# Single decay time of interest
decay_time: 

## Calculate adjoint neutron source
[step3]

## Prepare PARTISN input for adjoint neutron transport
[step4]

## Generate Monte Carlo variance reduction parameters 
# (biased source and weight windows)
[step5]


"""



def setup():
    with open(config_filename, 'w') as f:
        f.write(config)
    print('File "{}" has been written'.format(config_filename))
    print('Fill out the fields in these files then run ">> gtcadis.py step1"')

def step2():
    config = ConfigParser.ConfigParser()
    config.read(config_filename)

    # Get user input from config file
    geom = config.get('step2', 'geom_file')
    alara_dir = config.get('step2', 'alara_dir')
    irr_times = [config.getfloat('step2', 'irr_time')]
    decay_times = [config.getfloat('step2', 'decay_time')]

    # For a flat, 175 group neutron spectrum
    neutron_spectrum = [1]*175 # will be normalized
    flux_magnitudes = [1.75E14] # 1E12*175
    
    # Get materials from geometry file
    ml = MaterialLibrary(geom)
    mats = list(ml.values())

    T = calc_T(alara_dir, mats, neutron_spectrum, irr_times, flux_magnitudes, decay_times, remove=True)
    np.set_printoptions(threshold=np.nan)
    # save numpy array to be loaded by step 3 

def main():

    gtcadis_help = ('This script automates the GT-CADIS process of \n'
                    'producing variance reduction parameters to optimize the\n'
                    'neutron transport step of the Rigorous 2-Step (R2S) method.\n')
    setup_help = ('Prints the file "config.ini" to be\n'
                  'filled in by the user.\n')
    step2_help = 'Calculates the T matrix for each material in the geometry.'
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help=gtcadis_help, dest='command')

    setup_parser = subparsers.add_parser('setup', help=setup_help)
    step2_parser = subparsers.add_parser('step2', help=step2_help)

    args, other = parser.parse_known_args()
    if args.command == 'setup':
        setup()
    elif args.command == 'step2':
        step2()

if __name__ == '__main__':
    main()
