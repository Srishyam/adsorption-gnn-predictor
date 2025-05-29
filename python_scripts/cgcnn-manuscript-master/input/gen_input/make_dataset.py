import pickle
import copy
import pandas
import json
import glob
def new_x_sites(slab_dict):
        ''' 
        * Returns modified x-coordinates of adsorption sites
        '''
        ads_pos = slab_dict['initial_configuration']['atoms']['atoms'][0]['position']
        a = ads_pos[0]
        b = ads_pos[1]
        c = ads_pos[2]
        print('Old position: %s' % ads_pos)
        x_new = []
        for i in range(1,6):
                x_new.append(a-2 + i*0.8)
        ads_pos_new = []
        for x in x_new:
                ads_pos_new.append([x, b, c])
        return ads_pos_new

def new_y_sites(slab_dict):
        ''' 
        * Returns modified y-coordinates of adsorption sites
        '''
        ads_pos = slab_dict['initial_configuration']['atoms']['atoms'][0]['position']
        a = ads_pos[0]
        b = ads_pos[1]
        c = ads_pos[2]
        print('Old position: %s' % ads_pos)
        y_new = []
        for i in range(1,6):
                y_new.append(b-3 + i*1.2)
        ads_pos_new = []
        for y in y_new:
                ads_pos_new.append([a, y, c])
        return ads_pos_new

def new_z_sites(slab_dict):
        ''' 
        * Returns modified z-coordinates of adsorption sites
        '''
        ads_pos = slab_dict['initial_configuration']['atoms']['atoms'][0]['position']
        a = ads_pos[0]
        b = ads_pos[1]
        c = ads_pos[2]
        print('Old position: %s' % ads_pos)
        z_new = []
        for i in range(1,6):
                z_new.append(c-2 + i*0.6)
        ads_pos_new = []
        for z in z_new:
                ads_pos_new.append([a, b, z])
        return ads_pos_new

def move_adsorbate_init(mat_file):
	''' 
	* Moves the adsorbate along x, y, and z directions for the INITIAL configuration
	* Makes a list of these new positions and dumps that into a pickle file
	'''
	with open(mat_file, 'rb') as pkl_file:
		slab_dict = pickle.load(pkl_file)
	count = 1
	list_dict = []

	#slab_x = slab_dict[0]
	#slab_y = slab_dict[0]
	#slab_z = slab_dict[0]
	list_dict.append(slab_dict[0])
	move_x = new_x_sites(slab_dict[0])
	move_y = new_y_sites(slab_dict[0])
	move_z = new_z_sites(slab_dict[0])

	for site in move_x:
		count += 1
		slab_temp = copy.deepcopy(slab_dict[0])
		slab_temp['initial_configuration']['atoms']['atoms'][0]['position'] = site
		list_dict.append(slab_temp)


	for site in move_y:
		count += 1
		slab_temp = copy.deepcopy(slab_dict[0])
		slab_temp['initial_configuration']['atoms']['atoms'][0]['position'] = site
		list_dict.append(slab_temp)

	for site in move_z:
		count+=1
		slab_temp = copy.deepcopy(slab_dict[0])
		slab_temp['initial_configuration']['atoms']['atoms'][0]['position'] = site
		list_dict.append(slab_temp)
	print(len(move_x))
	print(len(move_y))
	print(len(move_z))

	with open('./mat_%s.pkl'%count, 'wb') as out_pick_file:
		pickle.dump(list_dict, out_pick_file)

def move_adsorbate_fin(mat_file):
	''' 
	* Moves the adsorbate along x, y, and z directions 
	* Makes a list of these new positions and dumps that into a pickle file
	'''
	with open(mat_file, 'rb') as pkl_file:
		slab_dict = pickle.load(pkl_file)
	count = 1
	list_dict = []

	#slab_x = slab_dict[0]
	#slab_y = slab_dict[0]
	#slab_z = slab_dict[0]
	list_dict.append(slab_dict[0])
	move_x = new_x_sites(slab_dict[0])
	move_y = new_y_sites(slab_dict[0])
	move_z = new_z_sites(slab_dict[0])

	for site in move_x:
		count += 1
		slab_temp = copy.deepcopy(slab_dict[0])
		slab_temp['initial_configuration']['atoms']['atoms'][0]['position'] = site
		list_dict.append(slab_temp)


	for site in move_y:
		count += 1
		slab_temp = copy.deepcopy(slab_dict[0])
		slab_temp['initial_configuration']['atoms']['atoms'][0]['position'] = site
		list_dict.append(slab_temp)

	for site in move_z:
		count+=1
		slab_temp = copy.deepcopy(slab_dict[0])
		slab_temp['initial_configuration']['atoms']['atoms'][0]['position'] = site
		list_dict.append(slab_temp)
	print(len(move_x))
	print(len(move_y))
	print(len(move_z))

	with open('./mat_%s.pkl'%count, 'wb') as out_pick_file:
		pickle.dump(list_dict, out_pick_file)

def add_key(docs, key, value):
	''' 
	* Adds a key-value pair to the dictionaries in the docs list
	* Returns changed list
	'''
	slabs = []
	for doc in docs:
		temp = copy.deepcopy(doc)
		temp[key] = value
		slabs.append(temp)
	return slabs

def split_pickle(file):
	''' 
	* Splits the docs.pkl dataset file into mat_*.pkl files
	'''
	with open(file,'rb') as pickle_file:
		data = pickle.load(pickle_file)
		print('Total single-points in dataset: %s'%len(data))
		i = 0
		for dict in data:
			i += 1
			f = open('mat_%s.pkl'%('{0:05}'.format(i)),'wb')
			pickle.dump(dict, f)

def make_pickle(docs, f_name):
	pickle.dump(docs, open(f_name,'wb'))

def merge_pickle(docs_file_name, directory_name, start=1, stop=100):
	''' 
	* Merges given docs.pkl into desired number of slabs
	* Writes corresponding pickle file into the directory
	'''
#	with open('./mat_5-7.pkl', 'rb') as p_fil:
#	with open('./mat_4.pkl', 'rb') as p_fil:
#	with open('./mat_1-10.pkl', 'rb') as p_fil:
	with open(docs_file_name, 'rb') as p_fil:
		data = pickle.load(p_fil)
	list = []
	for i in range(start,stop):
		list.append(data[i])
#	list.append(data[4])
#	list.append(data[0])
#	for i in [5,5]:
#	for i in [5,6]:
#		list.append(data[i-5])

	with open('%s/mat_list.pkl'%directory_name, 'wb') as p1_fil:
		pickle.dump(list, p1_fil)

## <Execution> ##
#directory = './'
#docs_file = '/mnt/hdd1/sragha20/cgcnn-manuscript-master/input/docs.pkl'
#merge_pickle(docs_file,directory)

''' 
mat_file_loc = './mat_1.pkl'
#move_adsorbate(mat_file_loc)
file = open(mat_file_loc, 'rb')
og_slab = pickle.load(file)
# Add new key
new_key = 'work_fn'   # New key name
value = float(5.0)    # New value of key
slab = add_key(og_slab, new_key, value)
list = [og_slab[0], slab[0]]
make_pickle(list, './mat_new_key.pkl')
'''

''' 
dir = './'
start = 1
stop = 6
merge_pickle(dir, start, stop)
'''

