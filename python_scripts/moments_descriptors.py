'''
Python script and functions to find the CN_bar_sd values based on structure and active site coordinates as inputs
'''
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pymatgen.core import Structure
from pymatgen.analysis.local_env import MinimumDistanceNN

from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build.supercells import make_supercell
import copy
import math

import numpy as np

def define_global_variables_moments():
	''' 
	etas_vector:
	eta_ss_sig = etas_vector[0] = -1.32 || -1.40
	eta_sp_sig = etas_vector[1] = 1.42
	eta_ds_sig = etas_vector[2] = -1.83 || -3.16
	eta_dp_sig = etas_vector[3] = -3*math.sqrt(15)/(2*math.pi)
	eta_dp_pi = etas_vector[4] = 3*math.sqrt(5)/(2*math.pi)
	eta_pp_sig = etas_vector[5] = 2.22
	eta_pp_pi = etas_vector[6] = -0.63
	eta_dd_sig = etas_vector[7] = -45/math.pi || -16.2
	eta_dd_pi = etas_vector[8] = 30/math.pi || 8.75
	eta_dd_delta = etas_vector[9] = -15/(2*math.pi) || 0
	'''
	global cutoff_distance_mu, cutoff_distance_cn, d_thresh, h_planck, m_elec, r_p_orbital, r_d_orbital, h_sq_m, etas_vector, mnn
	cutoff_distance_cn=3.2
	d_thresh = 0.1
	cutoff_distance_mu=4.5	# Metals cut-off
	# cutoff_distance_mu=2.35		# Oxides cut-off

	mnn = MinimumDistanceNN(cutoff=cutoff_distance_mu, get_all_sites=False)

#	cutoff_distance_mu=2.05
#	h_planck = 6.62607*math.pow(10,-34)
#	m_elec = 9.1094*math.pow(10,-31)
	h_planck = 1
	m_elec = 1
	#h_sq_m = 7.6205
	h_sq_m = 7.6205
#	r_p_orbital = 44.1*math.pow(10,-12)
#	r_d_orbital = 52.8*math.pow(10,-12)
	r_p_orbital = .441
#	r_d_orbital = .528
	r_d_orbital = 1.081

	etas_vector = [-1.40, 1.42, -3.16, -3*math.sqrt(15)/(2*math.pi), 3*math.sqrt(5)/(2*math.pi), 2.22, -0.63, -16.2, 8.75, 0.]

def define_slab_struct(struct):
	global slab_struct
#	supercell = [[3,0,0],[0,3,0],[0,0,3]]
	supercell = [[1,0,0],[0,1,0],[0,0,1]]
	slab_struct = AseAtomsAdaptor.get_structure(make_supercell(AseAtomsAdaptor.get_atoms(struct), supercell))

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

def get_custom_nn_list(site_cart_coords, struct=None):
	define_global_variables_classical()
	if struct is not None:
		define_slab_struct(struct)
	global d_thresh, slab_struct, cutoff_distance_mu
	cutoff = cutoff_distance_mu
	nn_list = list()
	prefix_set_a = [-3,-2,-1,0,1,2,3]
	prefix_set_b = [-1,0,1]
	#prefix_set_a = [1]
	#prefix_set_b = [1]
	flag = 0
	for site in struct:
		new_site = copy.deepcopy(site)
		for ia in prefix_set_a:
			for ib in prefix_set_b:
				new_x = site.coords[0] + ia*struct.lattice.matrix[0][0] + ib*struct.lattice.matrix[1][0]
				new_y = site.coords[1] + ia*struct.lattice.matrix[0][1] + ib*struct.lattice.matrix[1][1]
				new_z = site.coords[2] + ia*struct.lattice.matrix[0][2] + ib*struct.lattice.matrix[1][2]
				new_nn_coords = np.array([new_x,new_y,new_z])
				if np.linalg.norm(new_nn_coords-site_cart_coords) < cutoff and np.linalg.norm(new_nn_coords-site_cart_coords) > d_thresh:
					new_site.coords = new_nn_coords
					nn_list.append(new_site)
	return nn_list

def calc_sp_sp_hopping_integrals(act_site_coords, nn_coords, scaling_factor=1):
	global h_planck, m_elec, r_p_orbital, etas_vector, h_sq_m
	# Constants from eq. 1-21 of Elementary Electronic Structure by Harrison #
	eta_ss_sig = etas_vector[0]
	eta_sp_sig = etas_vector[1]
	eta_pp_sig = etas_vector[5]
	eta_pp_pi = etas_vector[6]
	dist = np.linalg.norm(act_site_coords - nn_coords)
	# Directional cosines #
	''' 
	l = 0
	m = 0
	n = 1
	'''
	L = (nn_coords[0] - act_site_coords[0])/dist	# cos(alpha)
	M = (nn_coords[1] - act_site_coords[1])/dist	# cos(beta)
	N = (nn_coords[2] - act_site_coords[2])/dist	# cos(gamma)
#	dist = dist*math.pow(10,-10)			# convert angstrom to m
	dist = dist*scaling_factor			# apply scaling factor
	# Hopping integrals #
	v_ss_sigma = eta_ss_sig*h_sq_m*(1./math.pow(dist,2))
	v_sp_sigma = eta_sp_sig*h_sq_m*(1./math.pow(dist,2))
	v_pp_sigma = eta_pp_sig*h_sq_m*(1./math.pow(dist,2))
	v_pp_pi = eta_pp_pi*h_sq_m*(1./math.pow(dist,2))
	return v_ss_sigma, v_sp_sigma, v_pp_sigma, v_pp_pi, L, M, N

def calc_sp_sp_matrix(V_ss_sigma, V_sp_sigma, V_pp_sigma, V_pp_pi, l, m, n):
	# Matrix E_jk equations #
	# Row 1
	E_s_s = V_ss_sigma
	E_s_x = l*V_sp_sigma
	E_s_y = m*V_sp_sigma
	E_s_z = n*V_sp_sigma
	row_1_jk = [E_s_s, E_s_x, E_s_y, E_s_z]
	E_spsp_input = copy.deepcopy(row_1_jk)
	# Row 2
	E_x_s = -l*V_sp_sigma
	E_x_x = math.pow(l,2)*V_pp_sigma + (1 - math.pow(l,2))*V_pp_pi
	E_x_y = l*m*V_pp_sigma - l*m*V_pp_pi
	E_x_z = l*n*V_pp_sigma - l*n*V_pp_pi
	row_2_jk = [E_x_s, E_x_x, E_x_y, E_x_z]
	E_spsp_input.extend(row_2_jk)
	# Row 3
	E_y_s = -m*V_sp_sigma
	E_y_x = l*m*V_pp_sigma - l*m*V_pp_pi
	E_y_y = math.pow(m,2)*V_pp_sigma + (1 - math.pow(m,2))*V_pp_pi
	E_y_z = m*n*V_pp_sigma - m*n*V_pp_pi
	row_3_jk = [E_y_s, E_y_x, E_y_y, E_y_z]
	E_spsp_input.extend(row_3_jk)
	# Row 4
	E_z_s = -n*V_sp_sigma
	E_z_x = l*n*V_pp_sigma - l*n*V_pp_pi
	E_z_y = m*n*V_pp_sigma - m*n*V_pp_pi
	E_z_z = math.pow(n,2)*V_pp_sigma + (1 - math.pow(n,2))*V_pp_pi
	row_4_jk = [E_z_s, E_z_x, E_z_y, E_z_z]
	E_spsp_input.extend(row_4_jk)
	# Obtain the second matrix #
	E_spsp_matrix = np.array(E_spsp_input).reshape(4,4)
	return E_spsp_matrix

def calc_sd_sp_hopping_integrals(act_site_coords, nn_coords, scaling_factor=1):
	global h_planck, m_elec, r_p_orbital, r_d_orbital, etas_vector, h_sq_m
	# Constants #
	eta_ss_sig = etas_vector[0]
	eta_sp_sig = etas_vector[1]
	eta_ds_sig = etas_vector[2]
	eta_dp_sig = etas_vector[3]
	eta_dp_pi = etas_vector[4]
	dist = np.linalg.norm(act_site_coords - nn_coords)
	# Directional cosines #

	L = 0
	M = 0
	N = 1

	'''
	L = (nn_coords[0] - act_site_coords[0])/dist	# cos(alpha)
	M = (nn_coords[1] - act_site_coords[1])/dist	# cos(beta)
	N = (nn_coords[2] - act_site_coords[2])/dist	# cos(gamma)
	'''
	# dist = dist*math.pow(10,-10)			# convert angstrom to m
	dist = dist*scaling_factor			# apply scaling factor
	# Hopping integrals #
	v_ss_sigma = eta_ss_sig*h_sq_m*(1./math.pow(dist,2))
	v_sp_sigma = eta_sp_sig*h_sq_m*(1./math.pow(dist,2))
	v_ds_sigma = eta_ds_sig*h_sq_m*(math.sqrt(math.pow(r_d_orbital,3))/math.pow(dist,3.5))
	v_dp_pi = eta_dp_pi*h_sq_m*(math.sqrt(r_p_orbital*math.pow(r_d_orbital,3))/math.pow(dist,4))
	v_dp_sigma = eta_dp_sig*h_sq_m*(math.sqrt(r_p_orbital*math.pow(r_d_orbital,3))/math.pow(dist,4))
	return v_ss_sigma, v_sp_sigma, v_ds_sigma, v_dp_pi, v_dp_sigma, L, M, N

def calc_sd_sp_matrix(V_ss_sigma, V_sp_sigma, V_ds_sigma, V_dp_pi, V_dp_sigma, l, m, n):
	# Matrix E_ij equations #
	# Row 1
	E_s_s = V_ss_sigma
	E_s_x = l*V_sp_sigma
	E_s_y = m*V_sp_sigma
	E_s_z = n*V_sp_sigma
	row_1 = [E_s_s, E_s_x, E_s_y, E_s_z]
	E_sdsp_input = copy.deepcopy(row_1)
	# Row 2
	E_xy_s = math.sqrt(3)*l*m*V_ds_sigma
	E_xy_x = -1*( math.sqrt(3)*math.pow(l,2)*m*V_dp_sigma + m*(1-2*math.pow(l,2))*V_dp_pi )
	E_xy_y = -1*( math.sqrt(3)*math.pow(m,2)*l*V_dp_sigma + l*(1-2*math.pow(m,2))*V_dp_pi )
	E_xy_z = -1*( math.sqrt(3)*l*m*n*V_dp_sigma - 2*l*m*n*V_dp_pi )
	row_2 = [E_xy_s, E_xy_x, E_xy_y, E_xy_z]
	E_sdsp_input.extend(row_2)
	# Row 3
	E_yz_s = math.sqrt(3)*m*n*V_ds_sigma
	E_yz_x = -1*( math.sqrt(3)*l*m*n*V_dp_sigma - 2*l*m*n*V_dp_pi )
	E_yz_y = -1*( math.sqrt(3)*math.pow(m,2)*n*V_dp_sigma + n*(1-2*math.pow(m,2))*V_dp_pi )
	E_yz_z = -1*( math.sqrt(3)*math.pow(n,2)*m*V_dp_sigma + m*(1-2*math.pow(n,2))*V_dp_pi )
	row_3 = [E_yz_s, E_yz_x, E_yz_y, E_yz_z]
	E_sdsp_input.extend(row_3)
	# Row 4
	E_zx_s = math.sqrt(3)*n*l*V_ds_sigma
	E_zx_x = -1*( math.sqrt(3)*math.pow(l,2)*n*V_dp_sigma + n*(1-2*math.pow(l,2))*V_dp_pi )
	E_zx_y = -1*( math.sqrt(3)*l*m*n*V_dp_sigma - 2*l*m*n*V_dp_pi )
	E_zx_z = -1*( math.sqrt(3)*math.pow(n,2)*l*V_dp_sigma + l*(1-2*math.pow(n,2))*V_dp_pi )
	row_4 = [E_zx_s, E_zx_x, E_zx_y, E_zx_z]
	E_sdsp_input.extend(row_4)
	# Row 5
	E_XsqYsq_s = (1/2)*math.sqrt(3)*(math.pow(l,2)-math.pow(m,2))*V_ds_sigma
	E_XsqYsq_x = -1*( (1/2)*math.sqrt(3)*l*(math.pow(l,2)-math.pow(m,2))*V_dp_sigma + l*(1-math.pow(l,2)+math.pow(m,2))*V_dp_pi )
	E_XsqYsq_y = -1*( (1/2)*math.sqrt(3)*m*(math.pow(l,2)-math.pow(m,2))*V_dp_sigma - m*(1+math.pow(l,2)-math.pow(m,2))*V_dp_pi )
	E_XsqYsq_z = -1*( (1/2)*math.sqrt(3)*n*(math.pow(l,2)-math.pow(m,2))*V_dp_sigma - n*(math.pow(l,2)-math.pow(m,2))*V_dp_pi )
	row_5 = [E_XsqYsq_s, E_XsqYsq_x, E_XsqYsq_y, E_XsqYsq_z]
	E_sdsp_input.extend(row_5)
	# Row 6
	E_ZsqRsq_s = (math.pow(n,2) - (1/2)*(math.pow(l,2)+math.pow(m,2)))*V_ds_sigma
	E_ZsqRsq_x = -1*( l*(math.pow(n,2) - (1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dp_sigma - math.sqrt(3)*l*math.pow(n,2)*V_dp_pi )
	E_ZsqRsq_y = -1*( m*(math.pow(n,2) - (1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dp_sigma - math.sqrt(3)*m*math.pow(n,2)*V_dp_pi )
	E_ZsqRsq_z = -1*( n*(math.pow(n,2) - (1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dp_sigma + math.sqrt(3)*n*(math.pow(l,2)+math.pow(m,2))*V_dp_pi )
	row_6 = [E_ZsqRsq_s, E_ZsqRsq_x, E_ZsqRsq_y, E_ZsqRsq_z]
	E_sdsp_input.extend(row_6)
	# Obtain the first matrix #
	E_sdsp_matrix = np.array(E_sdsp_input).reshape(6,4)
	return E_sdsp_matrix

def calc_sp_sd_matrix(V_ss_sigma, V_sp_sigma, V_ds_sigma, V_dp_pi, V_dp_sigma, l, m, n):
	# Row 1
	E_s_s = V_ss_sigma
	E_s_xy = math.sqrt(3)*l*m*V_ds_sigma
	E_s_yz = math.sqrt(3)*m*n*V_ds_sigma
	E_s_zx = math.sqrt(3)*n*l*V_ds_sigma
	E_s_XsqYsq = (1/2)*math.sqrt(3)*(math.pow(l,2)-math.pow(m,2))*V_ds_sigma
	E_s_ZsqRsq = (math.pow(n,2) - (1/2)*(math.pow(l,2)+math.pow(m,2)))*V_ds_sigma
	row_1 = [E_s_s, E_s_xy, E_s_yz, E_s_zx, E_s_XsqYsq, E_s_ZsqRsq]
	E_spsd_input = copy.deepcopy(row_1)
	# Row 2
	E_x_s = -l*V_sp_sigma
	E_x_xy = math.sqrt(3)*math.pow(l,2)*m*V_dp_sigma + m*(1-2*math.pow(l,2))*V_dp_pi
	E_x_yz = math.sqrt(3)*l*m*n*V_dp_sigma - 2*l*m*n*V_dp_pi
	E_x_zx = math.sqrt(3)*math.pow(l,2)*n*V_dp_sigma + n*(1-2*math.pow(l,2))*V_dp_pi
	E_x_XsqYsq = (1/2)*math.sqrt(3)*l*(math.pow(l,2)-math.pow(m,2))*V_dp_sigma + l*(1-math.pow(l,2)+math.pow(m,2))*V_dp_pi
	E_x_ZsqRsq = l*(math.pow(n,2) - (1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dp_sigma - math.sqrt(3)*l*math.pow(n,2)*V_dp_pi
	row_2 = [E_x_s, E_x_xy, E_x_yz, E_x_zx, E_x_XsqYsq, E_x_ZsqRsq]
	E_spsd_input.extend(row_2)
	# Row 3
	E_y_s = -m*V_sp_sigma
	E_y_xy = math.sqrt(3)*math.pow(m,2)*l*V_dp_sigma + l*(1-2*math.pow(m,2))*V_dp_pi
	E_y_yz = math.sqrt(3)*math.pow(m,2)*n*V_dp_sigma + n*(1-2*math.pow(m,2))*V_dp_pi
	E_y_zx = math.sqrt(3)*l*m*n*V_dp_sigma - 2*l*m*n*V_dp_pi
	E_y_XsqYsq = (1/2)*math.sqrt(3)*m*(math.pow(l,2)-math.pow(m,2))*V_dp_sigma - m*(1+math.pow(l,2)-math.pow(m,2))*V_dp_pi
	E_y_ZsqRsq = m*(math.pow(n,2) - (1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dp_sigma - math.sqrt(3)*m*math.pow(n,2)*V_dp_pi
	row_3 = [E_y_s, E_y_xy, E_y_yz, E_y_zx, E_y_XsqYsq, E_y_ZsqRsq]
	E_spsd_input.extend(row_3)
	# Row 4
	E_z_s = -n*V_sp_sigma
	E_z_xy = math.sqrt(3)*l*m*n*V_dp_sigma - 2*l*m*n*V_dp_pi
	E_z_yz = math.sqrt(3)*math.pow(n,2)*m*V_dp_sigma + m*(1-2*math.pow(n,2))*V_dp_pi
	E_z_zx = math.sqrt(3)*math.pow(n,2)*l*V_dp_sigma + l*(1-2*math.pow(n,2))*V_dp_pi
	E_z_XsqYsq = (1/2)*math.sqrt(3)*n*(math.pow(l,2)-math.pow(m,2))*V_dp_sigma - n*(math.pow(l,2)-math.pow(m,2))*V_dp_pi
	E_z_ZsqRsq = n*(math.pow(n,2) - (1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dp_sigma + math.sqrt(3)*n*(math.pow(l,2)+math.pow(m,2))*V_dp_pi
	row_4 = [E_z_s, E_z_xy, E_z_yz, E_z_zx, E_z_XsqYsq, E_z_ZsqRsq]
	E_spsd_input.extend(row_4)
	# Obtain the matrix #
	E_spsd_matrix = np.array(E_spsd_input).reshape(4,6)
	return E_spsd_matrix

def calc_sd_sd_hopping_integrals(act_site_coords, nn_coords, scaling_factor=1):
	global h_planck, m_elec, r_d_orbital, etas_vector, h_sq_m
	# Constants #
	'''
	eta_ss_sig = etas_vector[0]
	eta_ds_sig = etas_vector[2]
	eta_dd_sig = etas_vector[7]
	eta_dd_pi = etas_vector[8]
	eta_dd_delta = etas_vector[9]
	'''
	eta_ss_sig = -1.4
	eta_ds_sig = -3.16
	eta_dd_sig = -16.2
	eta_dd_pi = 8.75
	eta_dd_delta = 0

	dist = np.linalg.norm(act_site_coords - nn_coords)
	#dist = 2.9
	# Directional cosines #
	
	L = 0
	M = 0
	N = 1
	
	'''
	L = (nn_coords[0] - act_site_coords[0])/dist	# cos(alpha)
	M = (nn_coords[1] - act_site_coords[1])/dist	# cos(beta)
	N = (nn_coords[2] - act_site_coords[2])/dist	# cos(gamma)
	'''
#	dist = dist*math.pow(10,-10)			# convert angstrom to m
	dist = dist*scaling_factor			# apply scaling factor
	# Hopping integrals #
	v_ss_sigma = eta_ss_sig*h_sq_m*(1./math.pow(dist,2))
	v_ds_sigma = eta_ds_sig*h_sq_m*(math.pow(r_d_orbital,1.5))/math.pow(dist,3.5)
	
	v_dd_sig = eta_dd_sig*h_sq_m*(math.pow(r_d_orbital,3)/math.pow(dist,5))
	v_dd_pi = eta_dd_pi*h_sq_m*(math.pow(r_d_orbital,3)/math.pow(dist,5))
	v_dd_delta = eta_dd_delta*h_sq_m*(math.pow(r_d_orbital,3)/math.pow(dist,5))
	return v_ss_sigma, v_ds_sigma, v_dd_sig, v_dd_pi, v_dd_delta, L, M, N

def calc_sd_sd_matrix(V_ss_sigma, V_ds_sigma, V_dd_sig, V_dd_pi, V_dd_delta, l, m, n):
	# Matrix E_ij equations #
	# Row 1
	E_s_s = V_ss_sigma
	E_s_xy = math.sqrt(3)*l*m*V_ds_sigma
	E_s_yz = math.sqrt(3)*m*n*V_ds_sigma
	E_s_zx = math.sqrt(3)*n*l*V_ds_sigma
	E_s_XsqYsq = (1/2)*math.sqrt(3)*(math.pow(l,2)-math.pow(m,2))*V_ds_sigma
	E_s_ZsqRsq = (math.pow(n,2) - (1/2)*(math.pow(l,2)+math.pow(m,2)))*V_ds_sigma
	row_1 = [E_s_s, E_s_xy, E_s_yz, E_s_zx, E_s_XsqYsq, E_s_ZsqRsq]
	E_sdsd_input = copy.deepcopy(row_1)
	# Row 2
	E_xy_s = math.sqrt(3)*l*m*V_ds_sigma
	E_xy_xy = 3*math.pow(l,2)*math.pow(m,2)*V_dd_sig + (math.pow(l,2)+math.pow(m,2)-4*math.pow(l,2)*math.pow(m,2))*V_dd_pi + (math.pow(n,2)+math.pow(l,2)*math.pow(m,2))*V_dd_delta
	E_xy_yz = 3*l*math.pow(m,2)*n*V_dd_sig + l*n*(1-4*math.pow(m,2))*V_dd_pi + l*n*(math.pow(m,2)-1)*V_dd_delta
	E_xy_zx = 3*math.pow(l,2)*m*n*V_dd_sig + n*m*(1-4*math.pow(l,2))*V_dd_pi + m*n*(math.pow(l,2)-1)*V_dd_delta
	E_xy_XsqYsq = (3/2)*l*m*(math.pow(l,2)-math.pow(m,2))*V_dd_sig + 2*l*m*(math.pow(m,2)-math.pow(l,2))*V_dd_pi + (1/2)*l*m*(math.pow(l,2)-math.pow(m,2))*V_dd_delta
	E_xy_ZsqRsq = math.sqrt(3)*l*m*(math.pow(n,2)-(1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dd_sig - math.sqrt(3)*2*l*m*math.pow(n,2)*V_dd_pi + (1/2)*math.sqrt(3)*l*m*(1+math.pow(n,2))*V_dd_delta
	row_2 = [E_xy_s, E_xy_xy, E_xy_yz, E_xy_zx, E_xy_XsqYsq, E_xy_ZsqRsq]
	E_sdsd_input.extend(row_2)
	# Row 3
	E_yz_s = math.sqrt(3)*m*n*V_ds_sigma
	E_yz_xy = 3*l*math.pow(m,2)*n*V_dd_sig + l*n*(1-4*math.pow(m,2))*V_dd_pi + l*n*(math.pow(m,2)-1)*V_dd_delta
	E_yz_yz = 3*math.pow(m,2)*math.pow(n,2)*V_dd_sig + (math.pow(m,2)+math.pow(n,2)-4*math.pow(m,2)*math.pow(n,2))*V_dd_pi + (math.pow(l,2)+math.pow(m,2)*math.pow(n,2))*V_dd_delta
	E_yz_zx = 3*m*math.pow(n,2)*l*V_dd_sig + m*l*(1-4*math.pow(n,2))*V_dd_pi + m*l*(math.pow(n,2)-1)*V_dd_delta
	E_yz_XsqYsq = (3/2)*m*n*(math.pow(l,2)-math.pow(m,2))*V_dd_sig - m*n*(1+2*(math.pow(l,2)-math.pow(m,2)))*V_dd_pi + m*n*(1+(1/2)*(math.pow(l,2)-math.pow(m,2)))*V_dd_delta
	E_yz_ZsqRsq = math.sqrt(3)*m*n*(math.pow(n,2)-(1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dd_sig + math.sqrt(3)*m*n*(math.pow(l,2)+math.pow(m,2)-math.pow(n,2))*V_dd_pi - (1/2)*math.sqrt(3)*m*n*(math.pow(l,2)+math.pow(m,2))*V_dd_delta
	row_3 = [E_yz_s, E_yz_xy, E_yz_yz, E_yz_zx, E_yz_XsqYsq, E_yz_ZsqRsq]
	E_sdsd_input.extend(row_3)
	# Row 4
	E_zx_s = math.sqrt(3)*n*l*V_ds_sigma
	E_zx_xy = 3*math.pow(l,2)*m*n*V_dd_sig + n*m*(1-4*math.pow(l,2))*V_dd_pi + m*n*(math.pow(l,2)-1)*V_dd_delta
	E_zx_yz = 3*m*math.pow(n,2)*l*V_dd_sig + m*l*(1-4*math.pow(n,2))*V_dd_pi + m*l*(math.pow(n,2)-1)*V_dd_delta
	E_zx_zx = 3*math.pow(n,2)*math.pow(l,2)*V_dd_sig + (math.pow(n,2)+math.pow(l,2)-4*math.pow(n,2)*math.pow(l,2))*V_dd_pi + (math.pow(m,2)+math.pow(n,2)*math.pow(l,2))*V_dd_delta
	E_zx_XsqYsq = (3/2)*n*l*(math.pow(l,2)-math.pow(m,2))*V_dd_sig + n*l*(1-2*(math.pow(l,2)-math.pow(m,2)))*V_dd_pi - n*l*(1-(1/2)*(math.pow(l,2)-math.pow(m,2)))*V_dd_delta
	E_zx_ZsqRsq = math.sqrt(3)*l*n*(math.pow(n,2)-(1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dd_sig + math.sqrt(3)*l*n*(math.pow(l,2)+math.pow(m,2)-math.pow(n,2))*V_dd_pi - (1/2)*math.sqrt(3)*l*n*(math.pow(l,2)+math.pow(m,2))*V_dd_delta
	row_4 = [E_zx_s, E_zx_xy, E_zx_yz, E_zx_zx, E_zx_XsqYsq, E_zx_ZsqRsq]
	E_sdsd_input.extend(row_4)
	# Row 5
	E_XsqYsq_s = (1/2)*math.sqrt(3)*(math.pow(l,2)-math.pow(m,2))*V_ds_sigma
	E_XsqYsq_xy = (3/2)*l*m*(math.pow(l,2)-math.pow(m,2))*V_dd_sig + 2*l*m*(math.pow(m,2)-math.pow(l,2))*V_dd_pi + (1/2)*l*m*(math.pow(l,2)-math.pow(m,2))*V_dd_delta
	E_XsqYsq_yz = (3/2)*m*n*(math.pow(l,2)-math.pow(m,2))*V_dd_sig - m*n*(1+2*(math.pow(l,2)-math.pow(m,2)))*V_dd_pi + m*n*(1+(1/2)*(math.pow(l,2)-math.pow(m,2)))*V_dd_delta
	E_XsqYsq_zx = (3/2)*n*l*(math.pow(l,2)-math.pow(m,2))*V_dd_sig + n*l*(1-2*(math.pow(l,2)-math.pow(m,2)))*V_dd_pi - n*l*(1-(1/2)*(math.pow(l,2)-math.pow(m,2)))*V_dd_delta
	E_XsqYsq_XsqYsq = (3/4)*math.pow(math.pow(l,2)-math.pow(m,2),2)*V_dd_sig + (math.pow(l,2)+math.pow(m,2)-math.pow(math.pow(l,2)-math.pow(m,2),2))*V_dd_pi + (math.pow(n,2)+(1/4)*math.pow(math.pow(l,2)-math.pow(m,2),2))*V_dd_delta
	E_XsqYsq_ZsqRsq = (1/2)*math.sqrt(3)*(math.pow(l,2)-math.pow(m,2))*(math.pow(n,2)-(1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dd_sig + math.sqrt(3)*math.pow(n,2)*(math.pow(m,2)-math.pow(l,2))*V_dd_pi + (1/4)*math.sqrt(3)*(1+math.pow(n,2))*(math.pow(l,2)-math.pow(m,2))*V_dd_delta
	row_5 = [E_XsqYsq_s, E_XsqYsq_xy, E_XsqYsq_yz, E_XsqYsq_zx, E_XsqYsq_XsqYsq, E_XsqYsq_ZsqRsq]
	E_sdsd_input.extend(row_5)
	# Row 6
	E_ZsqRsq_s = (math.pow(n,2) - (1/2)*(math.pow(l,2)+math.pow(m,2)))*V_ds_sigma
	E_ZsqRsq_xy = math.sqrt(3)*l*m*(math.pow(n,2)-(1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dd_sig - math.sqrt(3)*2*l*m*math.pow(n,2)*V_dd_pi + (1/2)*math.sqrt(3)*l*m*(1+math.pow(n,2))*V_dd_delta
	E_ZsqRsq_yz = math.sqrt(3)*m*n*(math.pow(n,2)-(1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dd_sig + math.sqrt(3)*m*n*(math.pow(l,2)+math.pow(m,2)-math.pow(n,2))*V_dd_pi - (1/2)*math.sqrt(3)*m*n*(math.pow(l,2)+math.pow(m,2))*V_dd_delta
	E_ZsqRsq_zx = math.sqrt(3)*l*n*(math.pow(n,2)-(1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dd_sig + math.sqrt(3)*l*n*(math.pow(l,2)+math.pow(m,2)-math.pow(n,2))*V_dd_pi - (1/2)*math.sqrt(3)*l*n*(math.pow(l,2)+math.pow(m,2))*V_dd_delta
	E_ZsqRsq_XsqYsq = (1/2)*math.sqrt(3)*(math.pow(l,2)-math.pow(m,2))*(math.pow(n,2)-(1/2)*(math.pow(l,2)+math.pow(m,2)))*V_dd_sig + math.sqrt(3)*math.pow(n,2)*(math.pow(m,2)-math.pow(l,2))*V_dd_pi + (1/4)*math.sqrt(3)*(1+math.pow(n,2))*(math.pow(l,2)-math.pow(m,2))*V_dd_delta
	E_ZsqRsq_ZsqRsq = math.pow(math.pow(n,2)-(1/2)*(math.pow(l,2)+math.pow(m,2)),2)*V_dd_sig + 3*math.pow(n,2)*(math.pow(l,2)+math.pow(m,2))*V_dd_pi + (3/4)*math.pow(math.pow(l,2)+math.pow(m,2),2)*V_dd_delta
	row_6 = [E_ZsqRsq_s, E_ZsqRsq_xy, E_ZsqRsq_yz, E_ZsqRsq_zx, E_ZsqRsq_XsqYsq, E_ZsqRsq_ZsqRsq]
	E_sdsd_input.extend(row_6)
	# Obtain the sd-sd matrix #
	E_sdsd_matrix = np.array(E_sdsd_input).reshape(6,6)
	return E_sdsd_matrix

def calc_mu2(act_site_coords, nn):
# Hop 1
	# Determine if hops are sp-sp or sp-sd or sd-sd #
	if nn['site'].specie == Element('O'):						# sd-sp hop
		V_ss_sigma, V_sp_sigma, V_ds_sigma, V_dp_pi, V_dp_sigma, l, m, n = calc_sd_sp_hopping_integrals(act_site_coords, nn['site'].coords)
		# Obtain the first matrix #
		E_ij_matrix = calc_sd_sp_matrix(V_ss_sigma, V_sp_sigma, V_ds_sigma, V_dp_pi, V_dp_sigma, l, m, n)
	else:										# sd-sd hop
		V_ss_sigma, V_ds_sigma, V_dd_sig, V_dd_pi, V_dd_delta, l, m, n = calc_sd_sd_hopping_integrals(act_site_coords, nn['site'].coords)
		# Obtain the first matrix #
		E_ij_matrix = calc_sd_sd_matrix(V_ss_sigma, V_ds_sigma, V_dd_sig, V_dd_pi, V_dd_delta, l, m, n)
# Hop 2
	# Obtain the second matrix #
	E_ji_matrix = np.transpose(E_ij_matrix)
	
	E_bond = E_ij_matrix @ E_ji_matrix
	mu = np.trace(E_bond)
	return mu

def calc_mu2_mod(act_site_coords, nn):
# Hop 1
	# Determine if hops are sp-sp or sp-sd or sd-sd #
	if nn['site'].specie == Element('O'):						# sd-sp hop
		V_ss_sigma, V_sp_sigma, V_ds_sigma, V_dp_pi, V_dp_sigma, l, m, n = calc_sd_sp_hopping_integrals(act_site_coords, nn['site'].coords)
		# Obtain the first matrix #
		#E_ij_matrix = calc_sd_sp_matrix(V_ss_sigma, V_sp_sigma, V_ds_sigma, V_dp_pi, V_dp_sigma, l, m, n)
		mu = V_ss_sigma**2 + V_sp_sigma**2 + V_ds_sigma**2 + 2*V_dp_pi**2 + V_dp_sigma**2
	else:										# sd-sd hop
		V_ss_sigma, V_ds_sigma, V_dd_sig, V_dd_pi, V_dd_delta, l, m, n = calc_sd_sd_hopping_integrals(act_site_coords, nn['site'].coords)
		# Obtain the first matrix #
		#E_ij_matrix = calc_sd_sd_matrix(V_ss_sigma, V_ds_sigma, V_dd_sig, V_dd_pi, V_dd_delta, l, m, n)
		#mu = V_ss_sigma**2 + 2*V_ds_sigma**2 + V_dd_sig**2  + 2*V_dd_pi**2 + 2*V_dd_delta**2
		mu =  V_ds_sigma**2 + V_dd_sig**2  + 2*V_dd_pi**2 + 2*V_dd_delta**2
# Hop 2
	# Obtain the second matrix #
	#E_ji_matrix = np.transpose(E_ij_matrix)
	#E_bond = E_ij_matrix @ E_ji_matrix
	#mu = np.trace(E_bond)
	return mu


def get_mu2_site(slab_struct, act_site_cart_coords):
	global cutoff_distance_mu
	#nn_list_1 = get_custom_nn_list(act_site_cart_coords, slab_struct)	# List of neighbors of first atom
	nn_list_1 = mnn.get_nn_info(slab_struct, get_site_index_from_coords(slab_struct,act_site_cart_coords))

def get_mu2_sum(slab_struct, act_site_cart_coords):
	global mnn
	nn_mu2_sum = 0
	
	nn_list_1 = mnn.get_nn_info(slab_struct, get_site_index_from_coords(slab_struct,act_site_cart_coords))
	for nn in nn_list_1:
		# Check for repeats
		if np.linalg.norm(act_site_cart_coords-nn['site'].coords) > 0.1:
			nn_mu = calc_mu2(act_site_cart_coords, nn)
			nn_mu2_sum += math.sqrt(nn_mu)
	return nn_mu2_sum

def get_unrelaxed_nn_mu2(struct, act_site_cart_coords):
	define_slab_struct(struct)
	global slab_struct
	
	mu2 = get_mu2_sum(struct,act_site_cart_coords)
	return mu2

def get_bulk_V(cif_struct):
	
	global mnn
	act_site_coords = cif_struct[0].coords
	# Hop 1
	# Determine if hops are sp-sp or sp-sd or sd-sd #
	
	nn_list_1 = mnn.get_nn_info(cif_struct, 0)
	dist=1000
	# Find nearest neighbor #
	for nn in nn_list_1:
		length=np.linalg.norm(nn['site'].coords - act_site_coords)
		if length < dist:
			site=nn
			dist = length
	return math.sqrt(calc_mu2(act_site_coords, site))

def get_unrelaxed_nn_distances(struct, act_site_cart_coords):
	global mnn
	site_index = get_site_index_from_coords(struct, act_site_cart_coords)
	struct = Structure.from_file('%s/POSCAR'%struct)
	nn_list = mnn.get_nn_info(struct,site_index)
	distance_list = list()
	for nn in nn_list:
		dist = np.linalg.norm(act_site_cart_coords - nn['site'].coords)
		distance_list.append(dist)
	return distance_list
define_global_variables_moments()