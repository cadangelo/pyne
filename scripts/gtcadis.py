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
# Path to hdf5 geometry file for adjoint neutron transport 
geom_file: 
# Path to processed nuclear data 
# (directory containing nuclib, fendl2.0bin.lib, fendl2.0bin.gam)
data_dir: 
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
    print('Fill out the fields in this file then run ">> gtcadis.py step1"')

def step2():
    config = ConfigParser.ConfigParser()
    config.read(config_filename)

    # Get user input from config file
    geom = config.get('step2', 'geom_file')
    data_dir = config.get('step2', 'data_dir')
    irr_times = [config.getfloat('step2', 'irr_time')]
    decay_times = [config.getfloat('step2', 'decay_time')]

    # For a flat, 175 group neutron spectrum, magnitude 1E12
    neutron_spectrum = [1]*175 # will be normalized
    flux_magnitudes = [1.75E14] # 1E12*175
    
    # Get materials from geometry file
    ml = MaterialLibrary(geom)
    mats = list(ml.values())
    print 'type mat list', type(ml.values[0])

    # Calculate T
    T = calc_T(data_dir, mats, neutron_spectrum, irr_times, flux_magnitudes, decay_times, remove=True)
    np.set_printoptions(threshold=np.nan)
    # Save numpy array that will be loaded by step 3
    np.save('tempT.npy', T)
    print 'T, datadir, mats, nspec, irr, fmag, dt, remove', type(T), type(data_dir), type(mats)
    print type(neutron_spectrum), type(irr_times), type(flux_magnitudes), type(decay_times)

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
