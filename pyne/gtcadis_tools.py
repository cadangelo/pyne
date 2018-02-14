import subprocess
import os
import shutil
import operator
from warnings import warn
from matplotlib import use, rc
use('Agg')
import pylab as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

from pyne.material import Material

thisdir = os.path.dirname(__file__)
run_dir = "run_dir"

icrp74 = [4.85e-14, 1.05178964444e-13, 2.479506e-13, 3.25389261111e-13,
3.61940294444e-13, 3.92039283333e-13, 4.14953202778e-13, 4.66408861111e-13,
6.23770438889e-13, 8.69128547222e-13, 1.23064374722e-12, 1.73501156667e-12,
2.110332e-12, 2.36759852778e-12, 2.51475411667e-12, 2.70672796111e-12,
3.10803125e-12, 3.51841538889e-12, 4.08768680556e-12, 4.98254111111e-12,
5.55262625e-12, 5.80543569444e-12, 6.28434147222e-12, 6.99045691667e-12,
8.08166191667e-12, 9.27983080556e-12, 1.04079044444e-11, 1.14805293611e-11,
1.25242233056e-11, 1.35482440278e-11, 1.45404468333e-11, 1.550478675e-11,
1.64839013889e-11, 1.74841599444e-11, 1.84669667222e-11, 1.94338164444e-11,
2.17549492222e-11, 2.55346733333e-11, 2.91651563889e-11, 3.56874038889e-11,
4.82267172222e-11, 6.926448e-11]

def make_cmap(colors, position=None, bit=False):
    """
    ***************************************************************************
    * FUNCTION TAKE FROM:
    *    http://schubert.atmos.colostate.edu/~cslocum/custom_cmap.html
    *
    * FUNCTION WAS NOT MODIFIED
    *
    * Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License:
    *     http://creativecommons.org/licenses/by-nc-nd/3.0/deed.en_US
    ***************************************************************************
    NAME
        Custom Colormaps for Matplotlib
    PURPOSE
        This program shows how to implement make_cmap which is a function that
        generates a colorbar.  If you want to look at different color schemes,
        check out https://kuler.adobe.com/create.
    PROGRAMMER(S)
        Chris Slocum
    REVISION HISTORY
        20130411 -- Initial version created
        20140313 -- Small changes made and code posted online
        20140320 -- Added the ability to set the position of each color

    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    """
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

def _normalize(neutron_spectrum):
    tol = 1E-8
    total = float(np.sum(neutron_spectrum))
    if abs(total - 1.0) > tol:
        warn("Normalizing neutron spectrum")
        neutron_spectrum = [x/total for x in neutron_spectrum]
    return neutron_spectrum    

def _write_matlib(mats, filename):
    s = ""
    for m, mat in enumerate(mats):
        s += mat.alara()
        s += "\n"
    with open(filename, 'w') as f:
        f.write(s)

def _write_fluxin(fluxes, fluxin_file):
    s = ""
    for flux in fluxes:
        for i, fl in enumerate(reversed(flux)):
           s += "{0:.6E} ".format(fl)
           if (i+1) % 6 ==0:
               s += '\n'
        s += '\n\n'
    with open(fluxin_file, 'w') as f:
        f.write(s)

def _write_inp(mats, num_n_groups, flux_magnitudes, irr_times, decay_times,
               input_file, matlib_file, fluxin_file, phtn_src_file):
        num_zones = len(mats)*(num_n_groups)
        s = "geometry rectangular\n\nvolume\n"
        for z in range(num_zones):
            s += "    1.0 zone_{0}\n".format(z)
        s += 'end\n\nmat_loading\n'
        for z in range(num_zones):
            s += "    zone_{0} mix_{1}\n".format(z, int(np.floor(z/float(num_n_groups))))
        s += 'end\n\n'
        for m, mat in enumerate(mats):
            s += "mixture mix_{0}\n".format(m)
            s += "    material {0} 1 1\nend\n\n".format(mat.metadata["name"])
        s += "material_lib {0}\n".format(matlib_file)
        s += "element_lib {0}/data/nuclib\n".format(thisdir)
        s += "data_library alaralib {}/data/fendl3bin\n".format(thisdir)
        s += "truncation 1e-7\n"
        s += "impurity 5e-6 1e-3\n"
        s += "dump_file {0}\n".format(os.path.join(run_dir, "dump_file"))
        for i, flux_magnitude in enumerate(flux_magnitudes):
            s += "flux flux_{0} {1} {2} 0 default\n".format(i, fluxin_file, flux_magnitude)
        s += "output zone\n"
        s += "integrate_energy\n"
        #s += "    photon_source {0}/data/fendl3bin {1} 24 1.00E4 2.00E4 5.00E4 1.00E5\n".format(thisdir, phtn_src_file)
        #s += "    2.00E5 3.00E5 4.00E5 6.00E5 8.00E5 1.00E6 1.22E6 1.44E6 1.66E6\n"
        #s += "    2.00E6 2.50E6 3.00E6 4.00E6 5.00E6 6.50E6 8.00E6 1.00E7 1.20E7\n"
        #s += "    1.40E7 2.00E7\nend\n"
        s += "    photon_source {0}/data/fendl3bin {1} 42\n".format(thisdir, phtn_src_file)
        s +="     1e4 2e4 3e4 4.5e4 6e4 7e4 7.5e4 1e5 1.5e5 2e5 3e5 4e5\n"
        s +="     4.5e5 5.1e5 5.12e5 6e5 7e5 8e5 1e6 1.33e6 1.34e6 1.5e6 1.66e6 2e6\n"
        s +="     2.5e6 3e6 3.5e6 4e6 4.5e6 5e6 5.5e6 6e6 6.5e6 7e6 7.5e6 8e6 1e7\n"
        s +="     1.2e7 1.4e7 2e7 3e7 5e7\nend\n"
        s += "pulsehistory my_schedule\n"
        s += "    1 0.0 s\nend\n"
        s += "schedule total\n"
        for i, irr_time in enumerate(irr_times):
            s += "    {0} s flux_{1} my_schedule 0 s\n".format(irr_time, i)
        s += "end\n"
        s += "cooling\n"
        for d in decay_times:
            s += "    {0} s\n".format(d)
        s += "end\n"
        with open(input_file, 'w') as f:
            f.write(s)

def calc_T(mats, neutron_spectrum, irr_times, flux_magnitudes, decay_times, remove=True):
    # Need to make a function that calculates T for an arbitrary irradiation/decay scenario
    """This function 

    Parameters:
    -----------
    mats : list of PyNE materials
        The materials to compute eta for. Material metadata should specify
        should specify the "name" of the material.
    neutron_spectrum : list
        Normalized neutron fluxes, from low energy to high energy
    irr_times : list
        The irradiation times to interpolate between
    flux_magnitude : float
        Magnitude of the neutron flux in n/cm2/s
    decay_times : list
        The decay times to interpolated between
    remove : bool
        If true, remove intermediate files 

    Returns:
    --------
    T : 
        T matrix for each material 
    """

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    neutron_spectrum = _normalize(neutron_spectrum)
    num_n_groups = len(neutron_spectrum)
    #num_p_groups = 24
    num_p_groups = 42
    num_mats = len(mats)
    num_decay_times = len(decay_times)
    num_irr_times = len(irr_times)

    # Write matlib file
    matlib_file = os.path.join(run_dir, "matlib")
    _write_matlib(mats, matlib_file)
     
    # Write fluxin file
    fluxin_file = os.path.join(run_dir, "fluxin")
    fluxes = []
    for m in range(num_mats):
        for n in range(num_n_groups):
            fluxes.append([neutron_spectrum[n] if x == n else 0 for x in range(num_n_groups)])
    _write_fluxin(fluxes, fluxin_file)
    
    # write geom file
    input_file = os.path.join(run_dir, "inp")
    phtn_src_file = os.path.join(run_dir, "phtn_src")
    _write_inp(mats, num_n_groups, flux_magnitudes, irr_times, decay_times,
               input_file, matlib_file, fluxin_file, phtn_src_file)
        
    # Run ALARA
    sub = subprocess.Popen(['alara', input_file], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
    T = np.zeros(shape=(num_mats, num_decay_times, num_n_groups, num_p_groups))

    with open(phtn_src_file, 'r') as f:
        i = 0
        for line in f.readlines():
            l = line.split()
            if l[0] == "TOTAL" and l[1] != "shutdown":
                m = int(np.floor(float(i)/(num_n_groups*num_decay_times)))
                dt = i % num_decay_times
                n = int(np.floor(i/float(num_decay_times))) % num_n_groups
                # WHAT FLUX MAGNITUDE SHOULD I ACTUALLY DIVIDE BY???????????
                T[m, dt, n, :] = [float(x)/(neutron_spectrum[n]*flux_magnitudes[0]) for x in l[3:]]
                # print i, m, dt, n
                i += 1
    if remove:
        shutil.rmtree(run_dir)
    return T


def calc_eta(mats, neutron_spectrum, irr_times, flux_magnitudes, decay_times, remove=True):

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    neutron_spectrum = _normalize(neutron_spectrum)
    num_n_groups = len(neutron_spectrum)
    num_mats = len(mats)
    num_decay_times = len(decay_times)
    num_irr_times = len(irr_times)
    eta = np.zeros(shape=(num_mats, num_decay_times))

    # Write matlib file
    matlib_file = os.path.join(run_dir, "matlib")
    _write_matlib(mats, matlib_file)
     
    # Write fluxin file
    fluxin_file = os.path.join(run_dir, "fluxin")
    fluxes = []
    for m in range(num_mats):
        for n in range(num_n_groups):
            fluxes.append([neutron_spectrum[n] if x == n else 0 for x in range(num_n_groups)])
        fluxes.append(neutron_spectrum) # total spectrum
        fluxes.append([0]*175) # blank spectrum
    _write_fluxin(fluxes, fluxin_file)
    
    # write geom file
    input_file = os.path.join(run_dir, "inp")
    phtn_src_file = os.path.join(run_dir, "phtn_src")
    # num_n_groups+2 is needed because there is one extra irradiation for the total spectrum
    # and one extra for the blank spectrum
    _write_inp(mats, num_n_groups+2, flux_magnitudes, irr_times, decay_times,
               input_file, matlib_file, fluxin_file, phtn_src_file)
        
    # Run ALARA
    sub = subprocess.Popen(['alara', input_file], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]

    # Parse ALARA output
    sup = np.zeros(shape=(num_mats, num_decay_times))
    tot = np.zeros(shape=(num_mats, num_decay_times))
    zero = np.zeros(shape=(num_mats, num_decay_times))
    with open(phtn_src_file, 'r') as f:
        i = 0
        for line in f.readlines():
            l = line.split()
            if l[0] == "TOTAL" and l[1] != "shutdown":
                row_sum = np.sum([float(x) for x in l[3:]])
                m = int(np.floor(float(i)/((num_n_groups+2)*num_decay_times)))
                dt = i % num_decay_times
                n = int(np.floor(i/float(num_decay_times))) % (num_n_groups + 2)
                if n == num_n_groups:
                    tot[m, dt] = row_sum
                elif n == num_n_groups + 1:
                    zero[m, dt] = row_sum
                else:
                    sup[m, dt] += row_sum
                i += 1
    for dt, decay_time in enumerate(decay_times):
       for m, mat in enumerate(mats):
           if np.isclose(tot[m, dt] - zero[m, dt], 0.0, rtol=1E-5) and \
              np.isclose(sup[m, dt] - zero[m, dt]*175, 0.0, rtol=1E-5):
               eta[m, dt] = 1.0
           elif tot[m, dt] > 0.0:
               eta[m, dt] = (sup[m, dt] - zero[m, dt]*175)/(tot[m, dt] - zero[m, dt])
           else:
               eta[m, dt] = 1E6

    if remove:
        shutil.rmtree(run_dir)
    
    return eta


def eta_sweep(mats, irr_bounds, decay_bounds, neutron_spectrum, flux_magnitude, remove=True, flux_to_dose=False):
    """This function computes eta values of a collection a materials over a
    range of irradiation/decay times for a given neutron spectrum

    Parameters:
    -----------
    mats : list of PyNE materials
        The materials to compute eta for. Material metadata should specify
        should specify the "name" of the material.
    irr_bounds : list
        The irradiation times to interpolate between
    decay_bounds : list
        The decay times to interpolated between
    neutron_spectrum : list
        Normalized neutron fluxes, from low energy to high energy
    flux_magnitude : float
        Magnitude of the neutron flux in n/cm2/s
    remove : bool
        If true, remove intermediate files used to run ALARA

    Returns:
    --------
    eta : 3D numpy array
        An array of eta values in the form:
        eta[material index, decay index, irradiation index].
        Used as input for plot_eta().
    """

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    neutron_spectrum = _normalize(neutron_spectrum)

    irr_times = [round(10**((np.log10(irr_bounds[x]) + np.log10(irr_bounds[x+1]))/2), 6)
                         for x in range(len(irr_bounds)- 1)]
    decay_times = [round(10**((np.log10(decay_bounds[x]) + np.log10(decay_bounds[x+1]))/2), 6)
                   for x in range(len(decay_bounds)- 1)]
    num_n_groups = len(neutron_spectrum)
    num_mats = len(mats)
    num_decay_times = len(decay_times)
    num_irr_times = len(irr_times)
    eta = np.zeros(shape=(num_mats, num_decay_times, num_irr_times))

    # Write matlib file
    matlib_file = os.path.join(run_dir, "matlib")
    _write_matlib(mats, matlib_file)
     
    # Write fluxin file
    fluxin_file = os.path.join(run_dir, "fluxin")
    fluxes = []
    for m in range(num_mats):
        for n in range(num_n_groups):
            fluxes.append([neutron_spectrum[n] if x == n else 0 for x in range(num_n_groups)])
        fluxes.append(neutron_spectrum)
        fluxes.append([0]*175)
    _write_fluxin(fluxes, fluxin_file)
    
    photon_spectra = np.zeros(shape=(num_mats, num_decay_times, num_irr_times, 42))
    for it, irr_time in enumerate(irr_times):
        # write geom file
        input_file = os.path.join(run_dir, "inp_{0}".format(it))
        phtn_src_file = os.path.join(run_dir, "phtn_src_{0}".format(it))
        tree_file = os.path.join(run_dir, "tree_{0}".format(it))
        irr_ints = [irr_time]
        flux_magnitudes = [flux_magnitude]
        # num_n_groups+2 is needed because there is one extra irradiation for the total spectrum
        # and one for the zero spectrum
        _write_inp(mats, num_n_groups+2, flux_magnitudes, irr_ints, decay_times,
                   input_file, matlib_file, fluxin_file, phtn_src_file)
        
        # Run ALARA
        print("On irradiation time {0} of {1}".format(it+1, len(irr_times)))
        sub = subprocess.Popen(['alara', input_file, '-t', tree_file], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        # Parse ALARA output
        sup = np.zeros(shape=(num_mats, num_decay_times))
        tot = np.zeros(shape=(num_mats, num_decay_times))
        zero = np.zeros(shape=(num_mats, num_decay_times))
        with open(phtn_src_file, 'r') as f:
            i = 0
            for line in f.readlines():
                l = line.split()
                if l[0] == "TOTAL" and l[1] != "shutdown":
                    if flux_to_dose:
                        row_sum = np.sum([float(x)*icrp74[y] for y, x in enumerate(l[3:])])
                    else:
                        row_sum = np.sum([float(x) for x in l[3:]])
                    m = int(np.floor(float(i)/((num_n_groups+2)*num_decay_times)))
                    dt = i % num_decay_times
                    n = int(np.floor(i/float(num_decay_times))) % (num_n_groups +2)
                    if n == num_n_groups:
                        tot[m, dt] = row_sum
                        photon_spectra[m, dt, it, :] = [float(x) for x in l[3:]]
                    elif n == num_n_groups + 1:
                        zero[m, dt] = row_sum
                    else:
                        sup[m, dt] += row_sum
                    i += 1
        for dt, decay_time in enumerate(decay_times):
           for m, mat in enumerate(mats):
               if (np.isclose(tot[m, dt], zero[m, dt], rtol=1E-4) and \
                  np.isclose(sup[m, dt], zero[m, dt]*175, rtol=1E-4)) \
                  or (tot[m, dt] - zero[m, dt] < 0 and sup[m, dt] - 175*zero[m, dt] < 0):
                   eta[m, dt, it] = 1.0
               elif tot[m, dt] - zero[m, dt] > 0.0 and sup[m, dt] - 175*zero[m, dt] < 0:
                   eta[m, dt, it] = 0
               elif tot[m, dt] - zero[m, dt] > 0.0:
                   eta[m, dt, it] = (sup[m, dt] - zero[m, dt]*175)/(tot[m, dt] - zero[m, dt])
               else:
                   print("mat: {} sup: {} tot: {} zero: {}".format(m, sup[m, dt], tot[m, dt], zero[m, dt]))
                   eta[m, dt, it] = 1E6

    if remove:
        shutil.rmtree(run_dir)
    
    maxes = {}
    for i, mat in enumerate(mats):
        m1 = abs(np.max(eta[i]) - 1)
        m2 = abs(np.min(eta[i]) - 1)
        print m1, m2
        maxes[mat.metadata["name"]] = np.max([m2, m1])

    maxes_sorted = ''
    for name, value in sorted(maxes.items(), key=operator.itemgetter(1), reverse=True):
       maxes_sorted += "{0}: {1}\n".format(name, value)

    return eta, maxes_sorted, photon_spectra
    
def plot_eta_sweep(eta, mats, irr_bounds, decay_bounds, flux_magnitude, spectrum_name, title=True, bounds=None, flux_to_dose=False):
    """Plots the output from eta_sweep
    """

    for m, mat in enumerate(mats):
        X,Y = np.meshgrid(irr_bounds, decay_bounds)
        Z = eta[m,:,:]
        if bounds is None:
            extent = np.max([abs(1 - Z.min()), abs(1 - Z.max())])
            if extent == 0.0:
                vmin = 0.9999
                vmax = 1.0001
            elif extent < 1.0:
                extent = np.ceil(extent*10)/10
                vmin = 1 - extent
                vmax = 1 + extent
            else:
                extent = np.ceil(extent)
                vmin = 0
                vmax = 1 + extent
        else:
            vmin = bounds[m][0]
            vmax = bounds[m][1]
    
        #fig=plt.figure(figsize=(5.15,3))
        fig=plt.figure()
        mysize=17
        font = {'family':'serif', 'size':mysize}
        rc('font', **font)

        if title:
            name = mat.metadata["name"].strip()
            name = name.replace("_", " ")
            fig.suptitle("$\eta$ for {0}, spectrum: {1}, total flux: {2:.3E}".format(name, spectrum_name.replace("_", " "), flux_magnitude), fontname="Times New Roman Bold")
        ax=fig.add_subplot(111)
        #im = ax.pcolor(X,Y,Z, cmap='RdBu_r', norm=LogNorm())
        if vmin == 1.0:
            colors = [(1,1,1), (1,0,0)]
            position = [0, 1]
        elif vmax == 1.0:
            colors = [(0,0,1), (1,1,1)]
            position = [0, 1]
        else:
            colors = [(0,0,1), (1,1,1), (1,0,0)]
            position = [0, (1.0-vmin)/(vmax-vmin), 1]

        im = ax.pcolor(X,Y,Z, cmap=make_cmap(colors, position=position), vmin=vmin, vmax=vmax)
        plt.xscale('log')
        plt.yscale('log')
        #plt.colorbar(im, orientation='vertical',format=LogFormatterMathtext())
        cb = plt.colorbar(im)
        cb.solids.set_rasterized(True)
        cb.formatter.set_useOffset(False)
        cb.update_ticks()
        if flux_to_dose:
            cb.set_label('$\eta_{I}$', rotation=0, labelpad=20, fontsize=25)
        else:
            cb.set_label('$\eta$', rotation=0, labelpad=20, fontsize=25)
        ax.set_xlabel("irradiation time (s)")
        ax.set_ylabel("decay time (s)")
        filename="{0}_{1}_{2:.2E}.pdf".format(mat.metadata["name"],spectrum_name, flux_magnitude)
        plt.savefig(filename.replace("/", "_"), bbox_inches='tight')
