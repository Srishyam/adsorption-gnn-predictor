import pickle

pickle_file = './prediction.pkl'
with open(pickle_file, 'rb') as pf:
	data = pickle.load(pf)
pickle_file_1 = './energies.pkl'
with open(pickle_file_1, 'rb') as pf1:
	data_1 = pickle.load(pf1)

abs_error = 0
pred_file = open('./predictions.txt','w')
en_file = open('./dft_energies.txt','w')
err_file = open('./error_percent.txt','w')
count = 0
for (entry, entry_1) in zip(data, data_1):
	count += 1
	abs_error += abs(float(entry)-float(entry_1))
	percentage_error = ((float(entry)-float(entry_1))/float(entry_1))*100
	pred_file.write('%s\n'%float(entry))
	en_file.write('%s\n'%float(entry_1))
	err_file.write('%s\n'%percentage_error)
mean_abs_error = abs_error/count
print(mean_abs_error)

