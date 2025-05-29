import sys
import os
from pathlib import Path
sys.path.insert(1,'/mnt/hdd1/sragha20/manuscript_1_docs/python_scripts/cgcnn-manuscript-master')
import pickle
import run_cgcnn
from input.gen_input import make_lists, make_dataset
from sklearn.model_selection import train_test_split
docs_file_loc = sys.argv[1]
directory = sys.argv[2]

#start_train = 22
#end_train = 3450
#start_test = 1
#end_test = 21

#directory = os.path.dirname(os.path.realpath(__file__))
directory_train = '%s/train_lists'%directory
directory_test = '%s/test_lists'%directory

# Make the directories for SDT lists for training and testing
Path(directory_train).mkdir(parents=True, exist_ok=True)
Path(directory_test).mkdir(parents=True, exist_ok=True)

# Make the training and testing datasets #
# Test-Train Split Here #
ds_full = pickle.load(open(docs_file_loc,'rb'))
ds_train, ds_test = train_test_split(ds_full, test_size=0.1, train_size=0.8)
training_file_loc = '%s/mat_list_train.pkl'%directory_train
test_file_loc = '%s/mat_list_test.pkl'%directory_test
with open(training_file_loc,'wb') as pf:
	pickle.dump(ds_train,pf)
with open(test_file_loc,'wb') as pf:
	pickle.dump(ds_test,pf)


#make_dataset.merge_pickle(training_file_loc, directory_train)

# Make the SDT and target lists for training #
make_lists.make_SDT_list(training_file_loc, directory_train)
make_lists.make_target_list(training_file_loc, directory_train)
# Make the SDT and target lists for testing #
make_lists.make_SDT_list(test_file_loc, directory_test)
make_lists.make_target_list(test_file_loc, directory_test)


target_list = '%s/target_list.pkl'%directory_test
SDT_list='%s/SDT_list.pkl'%directory_train

#num_SDT = end_train - start_train
#num_docs = end_test - start_test
num_docs = len(ds_full)
num_SDT = len(ds_train)
num_epochs = 200

run_cgcnn.figure_of_merit(directory_train, directory_test, training_file_loc, SDT_list, target_list, num_docs, num_SDT, num_epochs)
