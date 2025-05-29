import pickle
with open('./mat_1_x6-6.pkl','rb') as f:
	data = pickle.load(f)
print(len(data))

#print(data[0]['atoms']['symbol_counts']['H'])							# Addy of the lattice vectors of the first slab
#print(data[0].keys())
#print(data[0]['atoms'].keys())
#print(data[0]['initial_configuration']['atoms']['atoms'][0]['position'])	# Addy of the coordinates of first atom of first slab

#print(data[i]['atoms']['natoms'] for i in range(10))
#sum = 0
#for i in range(10):
#	sum += data[i]['atoms']['natoms']
#	print(data[i]['atoms']['natoms'])
#print(sum)
