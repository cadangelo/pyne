#!/usr/bin/env python
import argparse
import ConfigParser
from os.path import isfile

import numpy as np
from pyne.mesh import Mesh
from pyne.partisn import write_partisn_input, isotropic_vol_source, mesh_to_isotropic_source
from pyne.dagmc import discretize_geom, load
from pyne import nucname
from pyne.bins import pointwise_collapse


config_filename = 'config.ini'

config = \
"""
# Optional step to assess all materials in geometry for compatibility with 
# SNILB criteria
[step0]

# Prepare PARTISN input file for adjoint photon transport
[step1]

# Calculate T matrix for each material
[step2]

# Calculate adjoint neutron source
[step3]

# Prepare PARTISN input for adjoint neutron transport
[step4]
# Path to neutron geometry hdf5 file
geom_file:
# Path to adj neutron source hdf5 file
adj_n_src_file:

# Generate Monte Carlo variance reduction parameters 
# (biased source and weight windows)
[step5]


"""



def setup():
    with open(config_filename, 'w') as f:
        f.write(config)
    print('File "{}" has been written'.format(config_filename))
    print('Fill out the fields in these filse then run ">> gtcadis.py step1"')

def _names_dict():
    names = ['h1', 'd', 'h3', 'he3', 'he4', 'li6', 'li7', 'be9', 'b10', 'b11',
    'c12', 'n14', 'n15', 'o16', 'f19', 'na23', 'mgnat', 'al27', 'si28', 'si29',
    'si30', 'p31', 'snat', 'cl35', 'cl37', 'knat', 'canat', 'ti46', 'ti47', 'ti48',
    'ti49', 'ti50', 'vnat', 'cr50', 'cr52', 'cr53', 'cr54', 'mn55', 'fe54', 'fe56',
    'fe57', 'fe58', 'co59', 'ni58', 'ni60', 'ni61', 'ni62', 'ni64', 'cu63', 'cu65',
    'ganat', 'zrnat', 'nb93', 'mo92', 'mo94', 'mo95', 'mo96', 'mo97', 'mo98',
    'mo100', 'snnat', 'ta181', 'w182', 'w183', 'w184', 'w186', 'au197', 'pb206',
    'pb207', 'pb208', 'bi209']

    names_formatted = ['h1', 'h2', 'h3', 'he3', 'he4', 'li6', 'li7', 'be9', 'b10', 'b11',
    'c12', 'n14', 'n15', 'o16', 'f19', 'na23', 'mg', 'al27', 'si28', 'si29',
    'si30', 'p31', 's', 'cl35', 'cl37', 'k', 'ca', 'ti46', 'ti47', 'ti48',
    'ti49', 'ti50', 'v', 'cr50', 'cr52', 'cr53', 'cr54', 'mn55', 'fe54', 'fe56',
    'fe57', 'fe58', 'co59', 'ni58', 'ni60', 'ni61', 'ni62', 'ni64', 'cu63', 'cu65',
    'ga', 'zr', 'nb93', 'mo92', 'mo94', 'mo95', 'mo96', 'mo97', 'mo98',
    'mo100', 'sn', 'ta181', 'w182', 'w183', 'w184', 'w186', 'au197', 'pb206',
    'pb207', 'pb208', 'bi209']

    names_dict = {nucname.id(x):y for x, y in zip(names_formatted, names)}

    return names_dict
 
def _cards(source):
    cards = {"block1": {"isn": 16,
                        "maxscm": '3E8',
                        "maxlcm": '6E8',
                       },
             "block3": {"lib": "xsf21-71", # name of cross section library
                       "lng":175,
                       "maxord": 5,
                       "ihm": 227,
                       "iht": 10,
                       "ihs": 11,
                       "ifido": 1,
                       "ititl": 1,
                       "i2lp1": 0,
                       "savbxs": 0,
                       "kwikrd": 0
                       },
            "block5": {"source": source,
                       "ith":1,
                       "isct":5}
            }
    return cards
   
def step4():

    config = ConfigParser.ConfigParser()
    config.read(config_filename)

    geom = config.get('step4', 'geom_file')
    print(geom)

    adj_n_src = config.get('step4', 'adj_n_src_file')
    print(adj_n_src)
    mesh = Mesh(structured=True, mesh=adj_n_src)

    source = mesh_to_isotropic_source(mesh, "adj_n_src")

    names_dict = _names_dict()

    ngroup = 217

    cards = _cards(source)

    write_partisn_input(mesh, geom, ngroup, cards=cards, names_dict=names_dict, data_hdf5path="/materials", nuc_hdf5path="/nucid", fine_per_coarse=1)

    print('Run PARTISN and then run gtcadis.py step5')

def main():

    gtcadis_help = ('This script automates the GT-CADIS process of \n'
                    'producing variance reduction parameters to optimize the\n'
                    'neutron transport step of the Rigorous 2-Step (R2S) method.\n')
    setup_help = ('Prints the file "config.ini" to be\n'
                  'filled in by the user.\n')
    step1_help = 'Creates the PARTISN input file for adjoint photon transport.'
    step4_help = 'Creates the PARTISN input file for adjoint neutron transport'
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help=gtcadis_help, dest='command')

    setup_parser = subparsers.add_parser('setup', help=setup_help)
    step1_parser = subparsers.add_parser('step1', help=step1_help)
    step4_parser = subparsers.add_parser('step4', help=step4_help)

    args, other = parser.parse_known_args()
    if args.command == 'setup':
        setup()
    elif args.command == 'step1':
        step1()
    elif args.command == 'step4':
        step4()
   
if __name__ == '__main__':
    main()
