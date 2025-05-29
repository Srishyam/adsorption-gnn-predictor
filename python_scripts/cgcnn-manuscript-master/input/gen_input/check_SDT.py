import pickle
with open('./SDT_lists/SDT_11-511.pkl','rb') as pickle_file:
	SDT_list = pickle.load(pickle_file)
print('Length of SDT list is: %d' % len(SDT_list))
#for key in SDT_list.keys():
#	print(key)
#print(SDT_list[0][3])
#print('Length of tensor: %d'%len(SDT_list[0][3]))
