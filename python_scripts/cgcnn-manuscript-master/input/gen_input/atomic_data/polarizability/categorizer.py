import pandas as pd
import numpy as np
import json
import pickle
import copy

data = json.load(open('./atom_polarizability.json', 'r'))
#data = json.load(open('./2nd_ionization.json', 'r'))
#print(len(data))
#data['bins'], bins = pd.cut(np.array([i for i in data.values()]), 10, retbins = True)
#print(data['bins'].unique())
indices = [i for i in range(1,len(data)+1)]
dataframe = pd.DataFrame(data.values(), index=indices, columns=['Atom Polarizability'])
no_of_categs = 10
#temp, bins = pd.cut(dataframe['Atom Polarizability'], no_of_categs, retbins = True)
temp, bins = pd.qcut(dataframe['Atom Polarizability'], no_of_categs, retbins = True)
#print(temp)
#print(bins)
for bin in bins:
	print(bin)
''' 
Polarizability
indices = [i for i in range(1,118)]
dataframe = pd.DataFrame(data.values(), index=indices, columns=['Polarizability'])
temp, bins = pd.cut(dataframe['Polarizability'], 10, retbins = True)
categs = ['(0.98038 - 41.342]', '(41.342 - 81.304]', '(81.304 - 121.266]', '(121.266 - 161.228]', '(161.228 - 201.19]', '(201.19 - 241.152]', '(241.152 - 281.114]', '(281.114 - 321.076]', '(321.076 - 361.038]', '(361.038 - 401.00]']
dataframe['Category'] = pd.cut(dataframe['Polarizability'], bins, labels = categs)
'''
''' 
2nd Ionization Energy
'''
''' 
dataframe['Category'] = pd.cut(dataframe['2nd Ionization Energy'], bins, labels = categs)


one_hot = pd.get_dummies(dataframe['Category'])
#print(dataframe)
#print(dataframe['Category'].unique())
#print(temp)
#print(bins)

dataframe = dataframe.join(one_hot)
atom_fea = np.array(indices).reshape(-1,1)
for cat in categs:
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
'''
counter = np.zeros(10)
for points in data.values():
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

#for no in counter:
#	print(no)
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
