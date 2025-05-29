import multiprocess as mp
import sys
import time
import pickle
import json
sys.path.insert(1,'/mnt/hdd1/sragha20/cgcnn-manuscript-master/')
from run_cgcnn import *
from cgcnn.data import StructureDataTransformer_ocp
#from cgcnn.data import Shyam_StructureDataTransformer
import numpy as np
import concurrent.futures
import tqdm

def make_SDT_list(docs_loc, dir_loc, procs=110):
    all_slabs = pickle.load(open(docs_loc, 'rb'))
    docs = all_slabs
    print('Making SDT list of length: %d' % len(docs))

    SDT = StructureDataTransformer_ocp(atom_init_loc='/mnt/hdd1/sragha20/cgcnn-manuscript-master/input/atom_init.json',
                                   max_num_nbr=12, step=0.2, radius=1, use_tag=True, use_fixed_info=False,
                                   use_distance=True, train_geometry='final-adsorbate')
    SDT_out = SDT.transform(docs)
#    procs = mp.cpu_count()

    with mp.Pool(procs) as pool:
#    with concurrent.futures.ProcessPoolExecutor() as pool:
        SDT_list_object = list(tqdm.tqdm(pool.imap_unordered(lambda x: SDT_out[x], range(len(SDT_out)), chunksize=40), total=len(SDT_out)))
#        SDT_list_object = list(tqdm.tqdm(pool.map(lambda x: SDT_out[x], range(len(SDT_out)), chunksize=40), total=len(SDT_out)))
#    SDT_list_object = list(tqdm.tqdm(map(lambda x: SDT_out[x],range(len(SDT_out)))))
    with open('%s/SDT_list.pkl'%dir_loc, 'wb') as SDT_file:
        pickle.dump(SDT_list_object, SDT_file)

def make_target_list(docs_loc, dir_loc):
    all_slabs = pickle.load(open(docs_loc,'rb'))
    data_dict = all_slabs
    print('Making target list of length: %d' % len(data_dict))

    for data in data_dict:
        data['surface'] = get_surface_from_doc_ocp(data)
    target_array = np.array([data['energy'] for data in data_dict]).reshape(-1, 1)

    with open('%s/target_list.pkl'%dir_loc, 'wb') as target_file:
        pickle.dump(target_array, target_file)
