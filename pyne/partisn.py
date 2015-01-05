#!/usr/bin/env python

""" Module for the production of PartiSn input decks. PartiSn is a discrete
ordinates code produced by Los Almos National Laboratory (LANL). Can be used
to produce neutron, photon, or coupled neutron photon prblems, adjoint or
forward or time dependent problems can be run.

Module is designed to work on 1D, 2D, or 3D Cartesian geometries.

If PyTaps not installed then this module will not work.
"""

from __future__ import print_function, division
import sys
import collections
import string
import struct
import math
import os
import linecache
import datetime
from warnings import warn
from pyne.utils import QAWarning
import itertools
from sets import Set

import numpy as np
import tables

from pyne import dagmc
from pyne.material import Material
from pyne.material import MultiMaterial
from pyne.material import MaterialLibrary

from pyne import nucname
from pyne.binaryreader import _BinaryReader, _FortranRecord

warn(__name__ + " is not yet QA compliant.", QAWarning)

# Mesh specific imports
try:
    from itaps import iMesh
    HAVE_PYTAPS = True
except ImportError:
    warn("the PyTAPS optional dependency could not be imported. "
                  "All aspects of the PartiSn module are not imported.",
                  VnVWarning)
    HAVE_PYTAPS = False

if HAVE_PYTAPS:
    from pyne.mesh import Mesh, StatMesh, MeshError, IMeshTag


#class PartisnRead(object):
""" This class reads all necessary attributes from a material-laden 
geometry file, a pre-made PyNE mesh object, and the nuclear data cross 
section library, and any optional inputs that are necessary for creating a 
PARTISN input file. Supported are 1D, 2D, and 3D geometries.

Parameters
----------
    mesh : mesh object, a premade mesh object that conforms to the geometry. 
        Bounds of the mesh must correspond to the desired PartiSn fine mesh. 
        One fine mesh per coarse mesh will be created. Can be 1-D, 2-D, or 3-D.
        Only Cartesian based geometries are currently supported.
    hdf5 : file, a material-laden dagmc geometry file.
    nucdata : file, nuclear data cross section library.
                note: only BXSLIB format is currently supported.
    nuc_names : dict, pyne element/isotope names to bxslib name assignment,
                keys are pyne nucids (int) and values are bxslib names (str)
    datapath : str, optional, The path in the heirarchy to the data table 
            in an HDF5 file. (for MaterialLibrary)
                default = material_library/materials
    nucpath : str, optional, The path in the heirarchy to the 
            nuclide array in an HDF5 file. (for MaterialLibrary)
                default = material_library/nucid
"""

def read_hdf5_mesh(mesh, hdf5, nucdata, nuc_names, **kwargs):
    dagmc.load(hdf5)
    # optional inputs
    datapath = kwargs['datapath'] if 'datapath' in kwargs else '/material_library/materials'
    nucpath = kwargs['nucpath'] if 'nucpath' in kwargs else '/material_library/nucid'

    # get coordinate system and mesh bounds from mesh       
    coord_sys, bounds = _read_mesh(mesh)
    
    # Read the materials from the hdf5 and convert to correct naming convention
    mat_lib = _get_materials(hdf5, datapath, nucpath, nuc_names)
    
    # Assign materials to cells   
    mat_assigns = _materials_to_cells(hdf5)
    
    # determine the zones
    zones, zone_voxel = _define_zones(mesh, mat_assigns)
    for key, item in zones.iteritems():
        print(key, item)
    
    for key, item in zone_voxel.iteritems():
        print(key, item)
    
    # read nucdata
    xs_names = _read_bxslib(nucdata)

    return coord_sys, bounds, mat_lib, zones, xs_names
 
    
def _read_mesh(mesh):
    # determines the system geometry (1-D, 2-D, or 3-D Cartesian)
    # currently cartesian is only supported
    nx = len(mesh.structured_get_divisions("x"))
    ny = len(mesh.structured_get_divisions("y"))
    nz = len(mesh.structured_get_divisions("z"))
    
    # Check for dimensions with >1 voxel (>2 bounds)
    # This determines 1-D, 2-D, or 3-D
    dim = 0
    i = False
    j = False
    k = False
    if nx > 2:
        dim += 1
        i = "x"
    if ny > 2:
        dim += 1
        if not i:
            i = "y"
        else:
            j = "y"
    if nz > 2:
        dim += 1
        if not i:
            i = "z"
        elif not j:
            j = "z"
        else:
            k = "z"
    
    # coordinate system data
    if dim == 1:
        coord_sys = [i]
    elif dim == 2:
        coord_sys = [i, j]
    elif dim == 3:
        coord_sys = [i, j, k]
        
    # collect values of mesh boundaries for each coordinate
    bounds = {}
    fine = {}
    for i in coord_sys:
        bounds[i] = mesh.structured_get_divisions(i)
        #fine[i] = [1]*(len(bounds[i]) - 1)
    
    return coord_sys, bounds

 
def _read_bxslib(nucdata):
    # read entire file
    #binary_file = _BinaryReader(nucdata, mode='rb')
    #record = binary_file.get_fortran_record()
    #print(binary_file)
    ##print(record.get_double())
    #print(record.get_string(28))   
        
    bxslib = open(nucdata, 'rb')
    string = ""
    edits = ""
    xs_names=[]
    # 181st byte is the start of xsnames
    bxslib.seek(180)
    done = False
    while not done:
        for i in range(0,8):
            bytes = bxslib.read(1)
            pad1=struct.unpack('s',bytes)[0]
            if '\x00' in pad1:
                done = True
                return xs_names
            string += pad1
        xs_names.append(string.strip(" "))
        string=""
    

def _get_materials(hdf5, datapath, nucpath, nuc_names):
    # reads material properties from the loaded dagmc_geometry
    
    # set of exception nuclides for collapse_elements
    mat_exceptions = Set(nuc_names.keys())
    
    # collapse isotopes into elements
    mats = MaterialLibrary(hdf5,datapath=datapath,nucpath=nucpath)
    mats_collapsed = {}
    for mat_name in mats.keys():
        mats_collapsed[mat_name] = mats[mat_name].collapse_elements(mat_exceptions)
    
    # Check that the materials are valid:
    #   1) non zero and non-negative densities (density = True)
    #   2) set of nuclides is not empty (else it is vacuum) (empty = False)
    #   3) nucids appear in nuc_names
    # might put 2 and 3 later      
    
    # convert mass fraction to atom fraction and then to [at/b-cm]
    Na = 6.022*(10.**23) # Avagadro's number [at/mol]
    barn_conv = 10.**-24 # [cm^2/b]
    mat_lib = {}
    for mat_name, comp in mats_collapsed.iteritems():
        #print(comp)
        comp_atom_frac = comp.to_atom_frac() # atom fractions
        density = comp.mass_density() # [g/cc]
        
        if density < 0.0:
            warn("Material {0} has an invalid negative density.".format(mat_name))
        
        mol_mass = comp.molecular_mass() # [g/mol]
        comp_list = {}
        
        for nucid, frac in comp_atom_frac.iteritems():
            comp_list[nucid] = frac*density*Na*barn_conv/mol_mass # [at/b-cm]
        
        mat_lib[mat_name] = comp_list

    return mat_lib


def _materials_to_cells(hdf5):
    """Takes the material-laden geometry and matches cells to materials
    """
    # Load the geometry
    dag_geom = iMesh.Mesh()
    dag_geom.load(hdf5)
    dag_geom.getEntities()
    mesh_set = dag_geom.getEntSets()

    # Get tag handle
    vol_tag = dag_geom.getTagHandle('GEOM_DIMENSION')
    name_tag = dag_geom.getTagHandle('GLOBAL_ID')
    mat_tag = dag_geom.getTagHandle('NAME')

    # Get list of materials and list of cells
    mat_list = []
    geom_list = []
    for i in mesh_set:
        tags = dag_geom.getAllTags(i)
        for tag in tags:
            if tag == vol_tag:
                geom_list.append(i)
            if tag == mat_tag:
                mat_list.append(i)

    # assign material to cell
    dag_properties = set()
    mat_assigns={}
    for entity in geom_list:
        for meshset in mat_list:
            if meshset.contains(entity):
                mat_name = mat_tag[meshset]
                cell = name_tag[entity]
                dag_properties.add(_tag_to_script(mat_name))
                if 'mat:' in _tag_to_script(mat_name):
                    mat_assigns[cell] = _tag_to_script(mat_name)
    
    #print(mat_assigns)
    return mat_assigns


def _tag_to_script(tag):
    a = []
    # since we have a byte type tag loop over the 32 elements
    for part in tag:
        # if the byte char code is non 0
        if (part != 0):
            # convert to ascii
            a.append(str(unichr(part)))
            # join to end string
            script = ''.join(a)
    return script


def _define_zones(mesh, mat_assigns):
    """This function takes results of discretize_geom and finds unique voxels
    """
    dg = dagmc.discretize_geom(mesh)
    # Create dictionary of each voxel's info    
    voxel = {}
    for i in dg:
        idx = i[0]
        if idx not in voxel.keys():
            voxel[idx] = {}
            voxel[idx]['cell'] = []
            voxel[idx]['vol_frac'] = []
            #voxel[idx]['rel_error'] = []
            
        voxel[idx]['cell'].append(i[1])
        voxel[idx]['vol_frac'].append(i[2])
        #voxel[idx]['rel_error'].append(i[3])
    
    # determine which voxels are identical and remove and then assign zone to
    # voxel
    zone_voxel = {} #
    z = 0
    zones_cells = {} # defined by cell number
    match = False
    first = True    
    for idx, vals in voxel.iteritems():
        #for zone, info in zones_cells.iteritems():
        #    if vals == info:
        #        match = True
        #        break
        #    else:
        #        match = False
        #if first or not match:
        z += 1
        zones_cells[z] = voxel[idx]
        #first = False
        
        zone_voxel[idx] = z
    #zone_voxel = {}
            
    # Replace cell numbers with materials, eliminating duplicate materials
    # within single zone definition
    zones = {}
    for zone in zones_cells.keys():
        zones[zone] = {}
        #zones[zone]['vol_frac'] = zones_cells[zone]['vol_frac']
        zones[zone]['vol_frac'] = []
        zones[zone]['mat'] = []
        for i, cell in enumerate(zones_cells[zone]['cell']):
            if mat_assigns[cell] not in zones[zone]['mat']:
                # create new entry
                zones[zone]['mat'].append(mat_assigns[cell])
                zones[zone]['vol_frac'].append(zones_cells[zone]['vol_frac'][i])
            else:
                # update value that already exists with new volume fraction
                for j, val in enumerate(zones[zone]['mat']):
                    if mat_assigns[cell] == val:
                        vol_frac = zones[zone]['vol_frac'][j] + zones_cells[zone]['vol_frac'][i]
                        zones[zone]['vol_frac'][j] = vol_frac
                        #break
    # Alright Kalin, you are working here. You need to start at this point and
    # eliminate the duplicates in zones and then make sure the volume fractions are
    # adding properly (right now a vol_frac is > 1) !!!!!!!!
    return zones, zone_voxel


#class PartisnWrite(object):

def write_partisn_input(coord_sys, bounds, mat_lib, zones, xs_names, input_file):
    """This function writes out the necessary information to a text partisn 
    input file.
    
    Parameters
    ----------
        coord_sys : list of str, indicator of either 1-D, 2-D, or 3-D Cartesian
            geometry. 
                1-D: [i]
                2-D: [i ,j]
                3-D: [i, j, k]
                where i, j, and k are either "x", "y", or "z".
        bounds : dict of list of floats, coarse mesh bounds for each dimension. 
            Dictionary keys are the dimension "x", "y" or "z" for Cartesian. 
            Must correspond to a 1:1 fine mesh to coarse mesh interval.
        mat_lib : dict of dicts, keys are names of PyNE materials whose keys 
            are bxslib names and their value is atomic density in units 
            [at/b-cm].
        zones : dict of dict of lists, first dict key is PartiSn zone number 
            (int). Inner dict keys are "cell" with a list of cell numbers (int) 
            as values and "vol_frac" with a list of corresponding cell volume 
            fractions (float).
        xs_names : list of str, names of isotope/elements from the bxslib
    
    """
    block01 = _block01(coord_sys, xs_names, mat_lib, zones, bounds)
    #print(block01)
    
    block02 = _block02(bounds)
    #print(block02)

def _title():
    # figure out what to make the title
    pass
        
def _block01(coord_sys, xs_names, mat_lib, zones, bounds):
    block01 = {}
    
    # Determine IGEOM
    if len(coord_sys) == 1:
        block01['IGEOM'] = 'SLAB'
    elif len(coord_sys) == 2:
        block01['IGEOM'] = 'X-Y' # assuming cartesian
    elif len(coord_sys) == 3:
        block01['IGEOM'] = 'X-Y-Z' # assuming cartesian
    
    # NGROUP - have to read from bxslib still
    
    # ISN - have to read from bxslib still
    
    block01['NISO'] = len(xs_names)
    block01['MT'] = len(mat_lib)
    block01['NZONE'] = len(zones)
    
    # Number of Fine and Coarse Meshes
    # one fine mesh per coarse by default
    for key in bounds.keys():
        if key == 'x':
            block01['IM'] = len(bounds[key]) - 1
            block01['IT'] = block01['IM']
        elif key == 'y':
            block01['JM'] = len(bounds[key]) - 1
            block01['JT'] = block01['JM']
        elif key == 'z':
            block01['KM'] = len(bounds[key]) - 1
            block01['KT'] = block01['KM']
    
    # Optional Input IQUAD
    block01['IQUAD'] = 1 # default
    
    return block01

def _block02(bounds):
    block02 = {}
    
    # fine intervals are 1 by default
    for key in bounds.keys():
        if key == 'x':
            block02['XMESH'] = bounds[key]
            block02['XINTS'] = 1
        elif key == 'y':
            block02['YMESH'] = bounds[key]
            block02['YINTS'] = 1
        elif key == 'z':
            block02['XZMESH'] = bounds[key]
            block02['ZINTS'] = 1  
    
    
    #print(bounds)
    return block02
    

def _block03():
    pass

def _block04():
    pass

def _write():
    pass
    
    
