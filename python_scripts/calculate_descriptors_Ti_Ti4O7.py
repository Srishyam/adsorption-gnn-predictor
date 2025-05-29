'''
Script to execute functions from classical_descriptors.py in order to generate all descriptor data for the given structures
'''
import pickle
import numpy as np
import pandas
import copy
from classical_descriptors import *
from moments_descriptors import *
from pymatgen.core.structure import Structure

def get_index_from_coords(struct, coords):
    idx = -1
    for site in struct:
        idx += 1
        if np.linalg.norm(site.coords - coords) < 0.01:
            break
    return idx

def get_act_site_coords(struct, ads_sym):        # Assume that adsorbate is first n sites
    #if len(ads_sym) == 1:
    ads_site_coords = struct[0].coords
    act_site_coords = np.array([1000.,1000.,1000.])
    for i in range(len(ads_sym),len(struct)):
        if np.linalg.norm(struct[i].coords - ads_site_coords) < np.linalg.norm(ads_site_coords-act_site_coords):
            act_site_coords = copy.deepcopy(struct[i].coords)
    return act_site_coords
    
## Calculate CN, CN_bar, CN_bar_sd for one example Ti4O7 structure ##

file = '../DFT_calculations/Ti4O7_adsorption/relaxed_structures/201_ter0_site2.vasp'  # Use relaxed structure to get the accurate value
struct = Structure.from_file(file)
file_init = '../DFT_calculations/Ti4O7_adsorption/vasp_poscars/201_ter0_site2.vasp'
struct_init = Structure.from_file(file_init)
cif_struct = Structure.from_file('../DFT_calculations/mp-12205.cif')
# Input the cartesian coordinates for the active site (refer to the table in S.I.) #
act_coords_init = np.array([3.4404,	2.7400,	26.4880])
act_idx = get_site_index_from_coords(struct_init, act_coords_init)
act_coords = struct[act_idx].coords
define_global_variables_classical(set_cutoff=2.35)

bulk_V = get_bulk_V(cif_struct)
mu2_value = get_mu2_sum(struct, act_coords)
cn_max = get_CN_max(cif_struct)

psi = get_psi(struct, act_coords)
cn_bar_sd = mu2_value/(bulk_V)
cn = get_CN(struct, act_coords)
cn_bar = get_CN_bar(struct, act_coords, cn_max)
# delta_m = (12*bulk_V - mu2_value)/12
print('\nTi4O7\nCN = %.3f\nCN_bar = %.3f\nCN_bar_sd = %.3f\nPsi = %.3f' % (cn, cn_bar, cn_bar_sd, psi))

## Calculate CN, CM_bar, CN_bar_sd for one example Ti structure ##

## Test CN_bar, CN values for Ti ##

file = '../DFT_calculations/Ti_adsorption/relaxed_structures/121_site0_slab.vasp' # Use relaxed structure to get the accurate value
struct = Structure.from_file(file)
cif_struct = Structure.from_file('../DFT_calculations/mp-73.cif')

# Input the cartesian coordinates for the active site (refer to the table in S.I.) #
act_coords = np.array([0.000000, 2.815882, 15.265322])
define_global_variables_classical(set_cutoff=3.2)

bulk_V = get_bulk_V(cif_struct)
mu2_value = get_mu2_sum(struct, act_coords)
cn_max = get_CN_max(cif_struct)

psi = get_psi(struct, act_coords)
cn_bar_sd = mu2_value/(bulk_V)
cn = get_CN(struct, act_coords)
cn_bar = get_CN_bar(struct, act_coords, cn_max)
print('\nTi\nCN = %.3f\nCN_bar = %.3f\nCN_bar_sd = %.3f\nPsi = %.3f' % (cn, cn_bar, cn_bar_sd, psi))
