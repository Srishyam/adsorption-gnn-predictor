## Code to move adsorbate position of given slab ##
## Srishyam Raghavan ##
import pickle
def change_id(slab_dict):
	mpid = (slab_dict[0]['mpid'])
	return mpid

#file = open('./mat_lists/mat_1.pkl', 'rb')
# slab = pickle.load(file)
file = open('./mat_16.pkl', 'rb')
slabs = pickle.load(file)
# new_id = change_id(slab)
# print('New ID generated')

## Check ##
for slab in slabs:
	print(slab['initial_configuration']['atoms']['atoms'][0]['position'])

# print(new_pos)

