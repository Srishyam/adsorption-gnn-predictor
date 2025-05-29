'''
Python script and functions to find the CN and CN_bar descriptor values based on structure and active site coordinates as inputs
'''
import copy
import json
import math
import numpy as np

from pymatgen.analysis.adsorption import *
from pymatgen.analysis.adsorption import AdsorbateSiteFinder as ASF
from pymatgen.analysis.local_env import MinimumDistanceNN, CrystalNN
from pymatgen.core.periodic_table import Element

from ase.build.supercells import make_supercell

from statistics import geometric_mean

def get_nn_distances(struct, site_coords):
    ''' 
    * Input: pymatgen Structure, np.array of coords
    * Returns: list of near-neighbor distances for the input coord (set global variable: cutoff_distance
    '''
    global cutoff_distance
    site_index = 0
    for site in struct:
        if np.linalg.norm(site_coords - site.coords) < 0.2:
            break
        site_index+=1
    #mnn = MinimumDistanceNN(cutoff=cutoff_distance, get_all_sites=True)
    mnn = CrystalNN()

    nn_list = mnn.get_nn_info(struct,site_index)
    distance_list = list()
    for nn in nn_list:
        dist = np.linalg.norm(site_coords - nn['site'].coords)
        distance_list.append(dist)
    return distance_list


def get_nn_radii(struct, site_coords):
    ''' 
    * Inputs: pymatgen Structure, np.array of coords
    * Returns: dict of lists (info about NN counts for 2 different radii)
    '''
    global cutoff_distance

    site_index = 0
    sv_list = list()
    X_list = list()
    for site in struct:
        if np.linalg.norm(site_coords - site.coords) < 0.2:
            metal_symbol = str(site.specie)
            break
        site_index+=1
#    print(struct[site_index])

    radius_1 = cutoff_distance
    mnn = MinimumDistanceNN(cutoff=radius_1, get_all_sites=True)
    
    radius_2 = radius_1
    list_of_radii = list()
    list_of_radii.append(radius_1)
    list_of_nn_count = list()
#    print(radius_1)
    nn_list = mnn.get_nn_info(struct, site_index)
    list_of_nn_count.append(len(nn_list))
    while radius_2 <= 2*radius_1:
        second_nn_flag = 0
        radius_2 += 0.1
        mnn_mod = MinimumDistanceNN(cutoff=radius_2, get_all_sites=True)
        second_nn_list = mnn_mod.get_nn_info(struct, site_index)
        if len(second_nn_list) > len(nn_list):
            second_nn_flag = 1
        if second_nn_flag == 1:
            list_of_nn_count.append(len(second_nn_list))
            list_of_radii.append(radius_2)
    return dict({'list_of_radii':list_of_radii, 'list_of_nn_count':list_of_nn_count})

def get_psi(struct, site_coords):
    ''' 
    Calculates psi for rutile oxides, i.e., MO2
    Ref: https://www.nature.com/articles/s41467-020-14969-8
    * Inputs: pymatgen Structure, np.array of coords of site
    * Returns: float psi, used in delta_E calculation
    '''
    global cutoff_distance
    with open('./element_properties_v_10.json', 'rt') as jf:
        element_prop_list = json.load(jf)

    site_index = 0
    sv_list = list()
    X_list = list()
    for site in struct:
        if np.linalg.norm(site_coords - site.coords) < 0.2:
            list_index = site.specie.Z - 1
            sv_list.append(math.pow(element_prop_list[list_index]['valence_elec'],2))
            X_list.append(site.specie.X)
            X_list.append(site.specie.X)
            break
        site_index+=1
    mnn = MinimumDistanceNN(cutoff=cutoff_distance, get_all_sites=True)
    nn_list = mnn.get_nn_info(struct, site_index)
    for near_neighbor in nn_list:
        spec = str(near_neighbor['site'].specie)
        element = Element(spec)
        l_idx = element.Z - 1
        sv_list.append(math.pow(element_prop_list[l_idx]['valence_elec'],2))
        X_list.append(element.X)
    print(sv_list)
    print(X_list)
    psi = geometric_mean(sv_list)/geometric_mean(X_list)
    return psi

def get_CN_bar(struct, site_coords, cn_max):
    ''' 
    Calculates CN_bar for input structure
    Ref: DOI - 10.1002/anie.201402958
    * Inputs: pymatgen Structure, np.array of coords of site, maximum CN
    * Returns: float cn_bar, used in delta_E calculation
    '''
    global cutoff_distance
    site_index = 0
    for site in struct:
        if np.linalg.norm(site_coords - site.coords) < 0.02:
            break
        site_index+=1
    mnn = MinimumDistanceNN(cutoff=cutoff_distance, get_all_sites=True)
    nn_list = mnn.get_nn_info(struct,site_index)
    
    cn_bar = 0
    for neighbor in nn_list:
        weighted_nn_count_of_neighbor = len(mnn.get_nn(struct, neighbor['site_index']))/cn_max   
        cn_bar += weighted_nn_count_of_neighbor  
    return cn_bar

def get_CN(struct, site_coords):
    global cutoff_distance
    site_index = 0
    for site in struct:
        if np.linalg.norm(site_coords - site.coords) < 0.02:
            break
        site_index+=1
    mnn = MinimumDistanceNN(cutoff=cutoff_distance, get_all_sites=True)
    nn_list = mnn.get_nn_info(struct,site_index)
    return len(nn_list)

def get_CN_max(cif_struct):
    global cutoff_distance
    mnn = MinimumDistanceNN(cutoff=cutoff_distance, get_all_sites=True)

    site_count = len(cif_struct) - 1
    cn_list = list()
    nn_list = list()
    while site_count >= 0:
        nn_list.append(mnn.get_nn_info(cif_struct,site_count))
        cn_list.append(len(mnn.get_nn_info(cif_struct,site_count)))
        site_count -= 1

    return max(cn_list)

def define_global_variables_classical(set_cutoff):
    '''
    * Global variables for the entire script
    '''
    global supercell_a, supercell_b, a_lattice_tol, b_lattice_tol, vacuum, cutoff_distance
    # Hard code these as global variables #
    supercell_a = 1			# Supercell construction on a axis
    supercell_b = 1			# Supercell construction on b axis
    a_lattice_tol = 5.6			# Lattice tolerance for supercell construction (angstrom) along a axis
    b_lattice_tol = 6.9			# Lattice tolerance for supercell construction (angstrom) along b axis
    vacuum = 18.  			# Minimum slab vacuum (angstrom)
    # cutoff_distance = 3.2  		# Near neighbor sphere radius for metals (angstrom)
    cutoff_distance = set_cutoff  		# Near neighbor sphere radius for metal oxides (angstrom)


def check_coords_unique(lattice_vectors, list_of_coords, coords_to_check, norm_tol=None):
    flag = 0
    global displacement_norm
    norm_tol = displacement_norm
    a_vec = lattice_vectors[0]
    b_vec = lattice_vectors[1]
    for coords in list_of_coords:
        for a_premultiplier in [-1,0,1]:
            for b_premultiplier in [-1,0,1]:
                difference = coords - coords_to_check + a_premultiplier*a_vec + b_premultiplier*b_vec
                diff_norm = np.linalg.norm(difference)
                if diff_norm < norm_tol:
                    flag = 1
                    break
            if flag == 1:
                break
        if flag == 1:
            break

    if flag == 1:
        return True
    else:
        return False

def get_site_index_from_coords(struct, act_site_cart_coords):
	global d_thresh
	site_index = 0
	flag = 0
	prefix_set_a = [-1,0,1]
	prefix_set_b = [-1,0,1]
	for site in struct:
		new_site = copy.deepcopy(site)
		for ia in prefix_set_a:
			for ib in prefix_set_b:
				new_x = site.coords[0] + ia*struct.lattice.matrix[0][0] + ib*struct.lattice.matrix[1][0]
				new_y = site.coords[1] + ia*struct.lattice.matrix[0][1] + ib*struct.lattice.matrix[1][1]
				new_z = site.coords[2] + ia*struct.lattice.matrix[0][2] + ib*struct.lattice.matrix[1][2]
				new_nn_coords = np.array([new_x,new_y,new_z])
				if np.linalg.norm(site.coords - act_site_cart_coords) < d_thresh:
					act_site_index = site_index
					flag = 1
					break
		site_index += 1
	if flag == 0:
		print('Given site not found\n')
		return None
	else:
		return act_site_index
# define_global_variables()
