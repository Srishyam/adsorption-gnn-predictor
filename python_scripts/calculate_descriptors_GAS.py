'''
Script to execute functions from classical_descriptors.py and moments_descriptors.py in order to generate all descriptor data for the given structures in GASdb datasets
'''
import pickle
import numpy as np
import pandas
import copy
from classical_descriptors import *
from moments_descriptors import *
from pymatgen.core.structure import Structure
# Calculate descriptor values for GASdb datasets #
init_or_final_str = 'init'
mat_list_file_loc = './predictions_non_metals_h.pkl'
with open(mat_list_file_loc,'rb') as pf:
    ds_full = pickle.load(pf)
# Get init_struct for psi calcs #
new_list = list()
for d in ds_full:
    try:
        atom_list_init = d['init_struct']['atoms']['atoms']
        atom_list_fin = d['final_struct']['atoms']
        specie_list = list()
        coords_list_init = list()
        coords_list_fin = list()
        lattice = d['init_struct']['atoms']['cell']
        for at_init, at_fin in zip(atom_list_init, atom_list_fin):
            specie_list.append(Element(at_init['symbol']))
            coords_list_init.append(at_init['position'])
            coords_list_fin.append(at_fin['position'])
        py_struct_init = Structure(lattice, specie_list, coords_list_init, coords_are_cartesian=True)
        py_struct_fin = Structure(lattice, specie_list, coords_list_fin, coords_are_cartesian=True)
        active_site_index = get_site_index_from_coords(py_struct_init, get_act_site_coords(py_struct_init, d['adsorbate']))
        active_site_coords = atom_list_init[active_site_index]['position']
        active_site_coords_fin = atom_list_fin[active_site_index]['position']
        
        new_dict = dict()
        new_dict = copy.deepcopy(d)
        new_dict['psi'] = get_psi(py_struct_init, active_site_coords)
        mu2_value = get_mu2_sum(py_struct_fin, active_site_coords_fin)
        # print(d['mpid'])
        cif_struct = d['cif_struct']
        bulk_V = get_bulk_V(d['cif_struct'])
        new_dict['CN_bar_sd']  = mu2_value/(bulk_V)
       
        new_dict['CN_max'] = get_CN_max(cif_struct)
        new_dict['CN'] = get_CN(py_struct_init, active_site_coords)
        new_dict['CN_bar'] = get_CN_bar(py_struct_init, active_site_coords, new_dict['CN_max'])
        # new_dict['energy_predicted_psi'] = get_E_ads_OH(new_dict['CN_bar'], new_dict['psi'])
        new_list.append(new_dict)
    except:
        pass
df = pandas.DataFrame(new_list)
df['ae'] = abs(df['energy'] - df['energy_predicted_psi'])
print(df['ae'].mean())
print(df['ae'].std())
with open('/mnt/hdd1/sragha20/ads_datasets/predictions_non_metals_h_full.pkl','wb') as pf:
    pickle.dump(new_list,pf)