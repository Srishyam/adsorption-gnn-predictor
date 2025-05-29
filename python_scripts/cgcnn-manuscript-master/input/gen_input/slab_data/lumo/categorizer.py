import pandas as pd
import numpy as np
import json
import pickle
#import copy


### Generate the lists for categorization ###

''' 
directory = '/home/le_dir/databases/generate_data/split_pkl/docs_new_2'
homo_list = []
lumo_list = []
for i in range(1,47):
	data_sublist = pickle.load(open('%s/mat_%s.pkl'%(directory,i), 'rb'))
	for data in data_sublist:
		homo_list.append(data['HOMO'])
		lumo_list.append(data['LUMO'])
	print('Sublist %s done'%i)
print('Dumping HOMO lists...')
with open('./HOMO_list.json','w') as j_fil:
	json.dump(homo_list, j_fil)
print('Dumping LUMO lists...')
with open('./LUMO_list.json','w') as j_file:
	json.dump(lumo_list, j_file)
'''
### Generate the categories ###
lumo_list = json.load(open('./LUMO_list.json', 'r'))
print(len(lumo_list))
lumo_list = [lu for lu in lumo_list if lu != 10000]
#lumo_list.remove(lu for lu in lumo_list if lu == 10000)
#for lu in lumo_list:
#	if lu > 10 :
#		lumo_list.remove(lu)
print(max(lumo_list))
lumo_indices = [i for i in range(1,len(lumo_list)+1)]
dataframe = pd.DataFrame(lumo_list, index=lumo_indices, columns=['LUMO energy'])
no_of_categs = 10
temp, bins = pd.qcut(dataframe['LUMO energy'], no_of_categs, retbins = True)
print(dataframe)
#print(dataframe['Category'].unique())
#print(temp)
print(bins)

''' 
categories = ['[-0.348699, -0.261676)','[-0.261676, -0.220603)', '[-0.220603, -0.202272)', '[-0.202272, -0.195511)', '[-0.195511, -0.172056)', '[-0.172056, -0.161308)', '[-0.161308, -0.153347)', '[-0.153347,  -0.14445)', '[-0.14445, -0.102545)', '[-0.102545, -0.078699)']


categories = ['[-0.348699, -0.261676)','[-0.261676, -0.220603)', '[-0.220603, -0.202272)', '[-0.202272, -0.195511)', '[-0.195511, -0.172056)', '[-0.172056, -0.161308)', '[-0.161308, -0.153347)', '[-0.153347,  -0.14445)', '[-0.14445, -0.102545)', '[-0.102545, -0.078699)']
dataframe['Category'] = pd.cut(dataframe['HOMO energy'], bins, labels = categories)
one_hot = pd.get_dummies(dataframe['Category'])

dataframe = dataframe.join(one_hot)
atom_fea = np.array(homo_indices).reshape(-1,1)
for cat in categories:
	row = np.array(dataframe[cat])
	atom_fea = np.column_stack((atom_fea, row))
fin_fea = np.delete(atom_fea, 0, 1)
print(fin_fea)
'''


''' 
## Slab here ##
slab_dict = pickle.load(open('./mat_1.pkl', 'rb'))
print(slab_dict[0]['atoms']['atoms'][0].keys())
at_nos = copy.deepcopy(slab_dict[0])
pol_vec = [data[str(i)] for i in at_nos]
#print(pol_vec)


temp, bins = pd.cut(np.array([i for i in data.values()]), 10, retbins = True)
bucket_list = pd.arrays.IntervalArray(temp.unique())
print(bucket_list)
#for i in temp:
#	print(i)
#print(temp)
#print(bins[2])
counter = np.zeros(10)
'''
counter = np.zeros(no_of_categs)
for points in lumo_list:
	if bins[0] < points <= bins[1]:
		counter[0] += 1
	if bins[1] < points <= bins[2]:
		counter[1] += 1
	if bins[2] < points <= bins[3]:
		counter[2] += 1
	if bins[3] < points <= bins[4]:
		counter[3] += 1
	if bins[4] < points <= bins[5]:
		counter[4] += 1
	if bins[5] < points <= bins[6]:
		counter[5] += 1
	if bins[6] < points <= bins[7]:
		counter[6] += 1
	if bins[7] < points <= bins[8]:
		counter[7] += 1
	if bins[8] < points <= bins[9]:
		counter[8] += 1
	if bins[9] < points <= bins[10]:
		counter[9] += 1
#	if bins[10] < points <= bins[11]:
#		counter[10] += 1
#	if bins[11] < points <= bins[12]:
#		counter[11] += 1
cnt = -1
for no in counter:
	cnt += 1
	print('%s->%s -- %s'%(bins[cnt],bins[cnt+1],no))

#bins = 
#cat_list = []
#cat_list = np.uniques(temp)
#print(cat_list)
#for cat in temp:
#	print(cat)
#	if cat in cat_list:
#		print('\n')
#	else:
#		cat_list.append(cat)
#	print('\n')
#print(cat_list)
#print(1.25 in temp[1])
