import pickle
import sys
import os

def get_slab_view(slab_dict, pos_name = 'POSCAR.slab'):
	poscar_file = open('./slabs/%s'%pos_name, 'w')
	poscar_file.write('\n1.0000000\n')

	# Write lattice vectors #
	lattice_vectors = []
	for vector in slab_dict['atoms']['cell']:
		lattice_vectors.append(vector)
		for vec in vector:
			poscar_file.write('%s\t'%vec)
		poscar_file.write('\n')

	# Write symbols #
	symbols = slab_dict['atoms']['chemical_symbols']
	for sym in symbols:
		poscar_file.write('%s  ' % sym)
	poscar_file.write('\n')

	# Write symbol counts #
	symbol_counts = []
	for sym in symbols:
		symbol_count = slab_dict['atoms']['symbol_counts'][sym]
		symbol_counts.append(symbol_count)
		poscar_file.write('%s  ' % symbol_count)
	poscar_file.write('\n')

	# Write Cartesian coordinates #
	poscar_file.write('Cartesian\n')
	positions = []
	for atom in slab_dict['initial_configuration']['atoms']['atoms']:
		cart_coord = atom['position']
		for coord in cart_coord:
			poscar_file.write('%s\t' % coord)
		poscar_file.write('\n')
		positions.append(cart_coord)

	poscar_file.close()

file = open('./mat_16.pkl', 'rb')
docs = pickle.load(file)
pos_name = 'POSCAR.slab'
i = 0
for doc in docs:
	i += 1
	get_slab_view(doc, '%s_%s'%(pos_name,i))
