## Python code to convert huge pickle file into parts ##
import pickle
import glob
def split_pickle(file):
	''' 
	Function that reads the pickle file as a dictionary and splits it into separate pickle files
	'''
	with open(file,'rb') as pickle_file:
		data = pickle.load(pickle_file)
		print('Total single-points in dataset: %s'%len(data))
		i = 0
		for dict in data:
			i += 1
			f = open('mat_%s.pkl'%('{0:05}'.format(i)),'wb')
			pickle.dump(dict, f)

def merge_pickle(directory, total=100):
	''' 
	Function to merge the various pickle files into one
	'''
	dict_list = []
	mat_file_list = [file for file in glob.glob('/home/le_dir/databases/data_pickled/mat_*')]
	array = [x for x in range(1, total+1)]
	for i in array:
		with open('mat_%s.pkl'%('{0:05}'.format(i)),'rb') as pickle_file:
			print('In loop')
			data = pickle.load(pickle_file)
			dict_list.append(data)
	f_out = open('%s/mat_%s.pkl'%(directory,total),'wb')
	pickle.dump(dict_list, f_out)

def split_dataset(file, adsorbate=None):
	with open(file,'rb') as pickle_file:
		dataset = pickle.load(pickle_file)
	dataset_co = []
	for slab in dataset:
		if slab['adsorbate'] == adsorbate:
			dataset_co.append(slab)
	with open('./docs_%s.pkl'%adsorbate, 'wb') as p_f:
		pickle.dump(dataset_co, p_f)

def count_dataset(file):
	with open(file,'rb') as pickle_file:
		dict_list = pickle.load(pickle_file)
	print('Dataset length: %s'%len(dict_list))

file = '/mnt/hdd1/sragha20/cgcnn-manuscript-master/input/docs.pkl'
#file = '/mnt/hdd1/sragha20/cgcnn-manuscript-master/input/docs_new_2.pkl'
#file = '/mnt/hdd1/sragha20/cgcnn-manuscript-master/input/docs_new_2_CO.pkl'
#count_dataset(file)
ads = 'OH'
split_dataset(file, ads)
#split_pickle(file)
#dir = '/mnt/hdd1/sragha20/cgcnn_modified/pickled_data/split_data'
#rng = 10
#merge_pickle(dir, rng)
