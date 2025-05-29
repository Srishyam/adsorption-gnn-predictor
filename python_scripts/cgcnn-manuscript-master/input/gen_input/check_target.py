import pickle
with open('./target_list.pkl','rb') as target_file:
	target_list = pickle.load(target_file)
print(target_list)
