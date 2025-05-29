'''#
#!/usr/bin/env python
'''
# Shyam start/
import sys
sys.path.append('/mnt/hdd1/sragha20/cgcnn-manuscript-master/skorch')
import concurrent.futures
# /end Shyam
# import os
import numpy as np
import pandas as pd
import mongo
import time
import pickle
import math
import torch
from torch.optim import Adam, SGD
# from torch.utils.data import Dataset, DataLoader
import multiprocess as mp
# import mongo
from cgcnn.data import StructureData, ListDataset, StructureDataTransformer, StructureDataTransformer_ocp
# Shyam start/
#from cgcnn.data import Shyam_StructureData, ListDataset, Shyam_StructureDataTransformer
# /end Shyam
import tqdm
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skorch.callbacks import Checkpoint, LoadInitState
#from cgcnn.data import collate_pool, MergeDataset
# Shyam start/
from cgcnn.data import collate_pool, MergeDataset, collate_pool_validation
# /end Shyam
from cgcnn.model import CrystalGraphConvNet
from skorch import NeuralNetRegressor
import skorch.callbacks.base
from skorch.dataset import CVSplit
from skorch.callbacks.lr_scheduler import WarmRestartLR, LRScheduler
from adamwr.adamw import AdamW
from adamwr.cosine_scheduler import CosineLRWithRestarts
#from torch.optim.lr_scheduler import CyclicLR

def round_(n, decimals=0):
    '''
    Python can't round for jack. We use someone else's home-brew to do
    rounding. Credit goes to David Amos
    (<https://realpython.com/python-rounding/#rounding-half-up>).
    '''
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def get_surface_from_doc(doc):
    '''#
    Some of our functions parse by "surface", which we identify by mpid, Miller
    index, shift, and whether it's on the top or bottom of the slab. This
    helper function parses an aggregated/projected Mongo document for you and
    gives you back a tuple that contains these surface identifiers.

    Arg:
        doc     A Mongo document (dictionary) that contains the keys 'mpid',
                'miller', 'shift', and 'top'.
    Returns:
        surface A 4-tuple whose elements are the mpid, Miller index, shift, and
                a Boolean indicating whether the surface is on the top or
                bottom of the slab. Note that the Miller indices will be
                formatted as a string, and the shift will be rounded to 2
                decimal places.
    '''
    surface = (doc['mpid'], str(doc['miller']), round_(doc['shift'], 2), doc['top'])
    return surface
# Shyam start/
def get_surface_from_doc_ocp(doc):
    surface = (doc['bulk_mpid'], str(doc['miller_index']), round_(doc['shift'], 2), doc['top'])
    return surface
# /end Shyam

def get_docs_file(dataset, num_docs):
    start = time.time()
    docs_all = pickle.load(open(dataset, 'rb'))
    total_docs = len(docs_all)
    for doc in docs_all:
        doc['surface'] = get_surface_from_doc(doc)
    docs = docs_all[:num_docs]
    target_list = np.array([doc['energy'] for doc in docs]).reshape(-1,1)
    end = time.time()
    Docs_time = end - start
    print('Documents loaded')
    return docs, Docs_time, total_docs
# Shyam start/
def get_docs_file_ocp(dataset, num_docs):
    start = time.time()
    docs_all = pickle.load(open(dataset, 'rb'))
    total_docs = len(docs_all)
    for doc in docs_all:
        doc['surface'] = get_surface_from_doc_ocp(doc)
    docs = docs_all[:num_docs]
    target_list = np.array([doc['energy'] for doc in docs]).reshape(-1,1)
    end = time.time()
    Docs_time = end - start
    print('Documents loaded')
    return docs, Docs_time, total_docs
# /end Shyam
def get_SDT_list(dataset):
    start = time.time()

    # Shyam start/
#    SDT = Shyam_StructureDataTransformer(atom_init_loc = '/mnt/hdd1/sragha20/cgcnn-manuscript-master/input/atom_init.json',
#                                   max_num_nbr = 12, step = 0.2, radius = 1, use_tag = True,
#                                   use_fixed_info = False, use_distance = True,
#                                   train_geometry = 'final-adsorbate',
#                                   use_Shyam_fea = ['polarizability', 'second-ionization'])
    # /end Shyam
    # Change for OCP #
    SDT = StructureDataTransformer(atom_init_loc='/mnt/hdd1/sragha20/manuscript_1_docs/python_scripts/cgcnn-manuscript-master/input/atom_init.json',
#    SDT = StructureDataTransformer_ocp(atom_init_loc='/mnt/hdd1/sragha20/cgcnn-manuscript-master/input/atom_init.json',
                                  max_num_nbr=12,
                                   step=0.2,
                                  radius=1,
                                  use_tag=False,
                                  use_fixed_info=False,
                                  use_distance=True,
                                  train_geometry = 'final-adsorbate'
                                  )

    SDT_out = SDT.transform(dataset)

    structures = SDT_out[0]

    #Settings necessary to build the model (since they are size of vectors as inputs)
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    SDT_out = SDT.transform(dataset)
    with mp.Pool(110) as pool:
#    with mp.Pool(4) as pool:
#    with concurrent.futures.ProcessPoolExecutor() as pool:
        SDT_list = list(tqdm.tqdm(pool.imap_unordered(lambda x: SDT_out[x],range(len(SDT_out)),chunksize=40),total=len(SDT_out)))
#        SDT_list = list(tqdm.tqdm(pool.map(lambda x: SDT_out[x],range(len(SDT_out)),chunksize=40),total=len(SDT_out)))
#    SDT_list = list(tqdm.tqdm(map(lambda x: SDT_out[x],range(len(SDT_out)))))
#    SDT_list = list(SDT_out)
    end = time.time()
    SDT_time = end-start
    print('SDT list created')
    return SDT_time

def get_device():
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda")
    else:
        device='cpu'

    return device

def shuffle(SDT_list, target_list):
    indices = np.arange(len(SDT_list))
    SDT_training, SDT_test, target_training, target_test, train_idx, test_idx = \
    train_test_split(SDT_list, target_list,indices, test_size=0.2, random_state=42)
    return SDT_training, SDT_test, target_training, target_test

# Shyam start/
def noshuffle(train_dir, test_dir):
    SDT_train_list = pickle.load(open('%s/SDT_list.pkl'%train_dir, 'rb'))
    SDT_test_list = pickle.load(open('%s/SDT_list.pkl'%test_dir, 'rb'))
    target_train_list = pickle.load(open('%s/target_list.pkl'%train_dir, 'rb'))
    target_test_list = pickle.load(open('%s/target_list.pkl'%test_dir, 'rb'))
    return SDT_train_list, SDT_test_list, target_train_list, target_test_list
# /end Shyam
# Shyam start/
def Shyam_training(hyperparams, device, num_epochs, SDT_training: object,target_training ):
    structures = SDT_training[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    train_test_splitter = ShuffleSplit(test_size=0.25, random_state=42)
    # warm restart scheduling from https://arxiv.org/pdf/1711.05101.pdf

    LR_schedule = LRScheduler(CosineLRWithRestarts, batch_size=214, epoch_size=len(SDT_training), \
                              restart_period=10, t_mult=1.2)

    # Make a checkpoint to save parameters every time there is a new best for validation lost
    cp = Checkpoint(monitor='valid_loss_best', fn_prefix='valid_best_')

    # Callback to load the checkpoint with the best validation loss at the end of training
    class train_end_load_best_valid_loss(skorch.callbacks.base.Callback):
        def on_train_end(self, net, X, y):
            net.load_params('valid_best_params.pt')

    load_best_valid_loss = train_end_load_best_valid_loss()

    # To extract intermediate features, set the forward takes only the first return value to calculate loss
    class MyNet(NeuralNetRegressor):
        def get_loss(self, y_pred, y_true, **kwargs):
            y_pred = y_pred[0] if isinstance(y_pred, tuple) else y_pred  # discard the 2nd output
            return super().get_loss(y_pred, y_true, **kwargs)

    net = MyNet(
        CrystalGraphConvNet,
        module__orig_atom_fea_len=orig_atom_fea_len,
        module__nbr_fea_len=nbr_fea_len,
        batch_size=hyperparams[0],	## Batch_Size
        module__classification=False,
        lr=hyperparams[1],              ## LR
        max_epochs=num_epochs,
        module__atom_fea_len=hyperparams[3],	## atom_fea_len
        module__h_fea_len=hyperparams[4],	## h_fea_len
        module__n_conv=hyperparams[5],  ## n_conv
        module__n_h=hyperparams[6],     ## n_h
        optimizer__weight_decay=1e-5,
        optimizer=AdamW,
        iterator_train__pin_memory=True,
        iterator_train__num_workers=0,
        iterator_train__collate_fn=collate_pool,
        iterator_valid__pin_memory=True,
        iterator_valid__num_workers=0,
        iterator_valid__collate_fn=collate_pool,
        device=device,
        criterion=torch.nn.L1Loss,
        dataset=MergeDataset,
        train_split=CVSplit(cv=train_test_splitter),
        callbacks=[cp, load_best_valid_loss, LR_schedule]
    )
    start = time.time()
    net.initialize()
    net.fit(SDT_training, target_training)
    end = time.time()
    training_time = end - start
    return net, train_test_splitter, training_time
# /end Shyam
def training(device, num_epochs, SDT_training: object, target_training):
    structures = SDT_training[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    
    train_test_splitter = ShuffleSplit(test_size=0.25, random_state=42)
    # warm restart scheduling from https://arxiv.org/pdf/1711.05101.pdf

    LR_schedule = LRScheduler(CosineLRWithRestarts, batch_size=214, epoch_size=len(SDT_training), \
                              restart_period=10, t_mult=1.2)

    #Make a checkpoint to save parameters every time there is a new best for validation lost
    cp = Checkpoint(monitor='valid_loss_best',fn_prefix='valid_best_')
    #Callback to load the checkpoint with the best validation loss at the end of training
    class train_end_load_best_valid_loss(skorch.callbacks.base.Callback):
        def on_train_end(self, net, X, y):
            net.load_params('valid_best_params.pt')
    load_best_valid_loss = train_end_load_best_valid_loss()
    # To extract intermediate features, set the forward takes only the first return value to calculate loss
    class MyNet(NeuralNetRegressor):
        def get_loss(self, y_pred, y_true, **kwargs):
            y_pred = y_pred[0] if isinstance(y_pred, tuple) else y_pred  # discard the 2nd output
            return super().get_loss(y_pred, y_true, **kwargs)
    '''    
    net = MyNet(
        CrystalGraphConvNet,
        module__orig_atom_fea_len = orig_atom_fea_len,
        module__nbr_fea_len = nbr_fea_len,
        batch_size=214,
        module__classification=False,
        lr=0.0056,
        max_epochs= num_epochs, 
        module__atom_fea_len=46,
        module__h_fea_len=83,
        module__n_conv=8, #8
        module__n_h=4,
        optimizer__weight_decay=1e-5,
        optimizer=AdamW,
        iterator_train__pin_memory=True,
        iterator_train__num_workers=0,
        iterator_train__collate_fn = collate_pool,
        iterator_valid__pin_memory=True,
        iterator_valid__num_workers=0,
        iterator_valid__collate_fn = collate_pool,
        device=device,
    #   criterion=torch.nn.MSELoss,
        criterion=torch.nn.L1Loss,
        dataset=MergeDataset,
        train_split = CVSplit(cv=train_test_splitter),
        callbacks=[cp, load_best_valid_loss, LR_schedule]
    )
    '''
    # Shyam start/
    net = MyNet(
        CrystalGraphConvNet,
        module__orig_atom_fea_len = orig_atom_fea_len,
        module__nbr_fea_len = nbr_fea_len,
        batch_size=214,
        module__classification=False,
        lr=0.0056,
        # Shyam start/
        # lr=0.1,
        # /end Shyam
        max_epochs= num_epochs,
        module__atom_fea_len=46,
        module__h_fea_len=83,
        module__n_conv=8, #8
        module__n_h=4,
        optimizer__weight_decay=1e-5,
        optimizer=AdamW,
        iterator_train__pin_memory=True,
        iterator_train__num_workers=0,
        iterator_train__collate_fn = collate_pool,
        iterator_valid__pin_memory=True,
        iterator_valid__num_workers=0,
        iterator_valid__collate_fn = collate_pool,
        device=device,
    #   criterion=torch.nn.MSELoss,
        criterion=torch.nn.L1Loss,
        dataset=MergeDataset,
        train_split = CVSplit(cv=train_test_splitter),
        callbacks=[cp, load_best_valid_loss, LR_schedule]
    )
    start = time.time()
    net.initialize()
    net.fit(SDT_training, target_training)
    end = time.time()
    training_time = end-start
    return net, train_test_splitter, training_time

# Shyam start/
def Shyam_predict(hyperparams, train_dir, test_dir, dataset, SDT_list, target_list, num_docs, num_SDT, num_epochs, device):
    SDT_list = pickle.load(open(SDT_list, 'rb'))
    target_list = pickle.load(open(target_list, 'rb'))
    print('SDT_list and target_list are loaded')
    SDT_list = SDT_list[:num_SDT]
    target_list = target_list[:num_SDT]
    # Change for OCP dataset #
    docs, Docs_time, total_docs = get_docs_file(dataset, num_docs)
#    docs, Docs_time, total_docs = get_docs_file_ocp(dataset, num_docs)
    SDT_time = get_SDT_list(docs)

    # SDT_training, SDT_test, target_training, target_test = shuffle(SDT_list, target_list)

    # Shyam start/
    SDT_training, SDT_test, target_training, target_test = noshuffle(train_dir, test_dir)
    # /end Shyam

    net, train_test_splitter, training_time = Shyam_training(hyperparams, device, num_epochs, SDT_training, target_training)
    train_indices, valid_indices = next(train_test_splitter.split(SDT_training))

    output_list = net.predict(SDT_training)
    test_list = net.predict(SDT_test)

    train_error = mean_absolute_error(target_training[train_indices].reshape(-1),
                                      output_list[train_indices].reshape(-1))
    val_error = mean_absolute_error(target_training[valid_indices].reshape(-1),
                                    output_list[valid_indices].reshape(-1))
    test_error = mean_absolute_error(target_test.reshape(-1),
                                     test_list.reshape(-1))
    return val_error
# /end Shyam

'''
def prediction(dataset, SDT_list, target_list, num_docs, num_SDT, num_epochs, device):
'''
# Shyam start/
def prediction(train_dir, test_dir, dataset, SDT_list, target_list, num_docs, num_SDT, num_epochs, device):
# /end Shyam
    SDT_list = pickle.load(open(SDT_list, 'rb'))
    target_list = pickle.load(open(target_list, 'rb'))
    print('SDT_list and target_list are loaded')
    SDT_list = SDT_list[:num_SDT]
    target_list = target_list[:num_SDT]
    # Change for OCP #
    docs, Docs_time, total_docs = get_docs_file(dataset, num_docs)
#    docs, Docs_time, total_docs = get_docs_file_ocp(dataset, num_docs)
    SDT_time = get_SDT_list(docs)

    #SDT_training, SDT_test, target_training, target_test = shuffle(SDT_list, target_list)

    # Shyam start/
    SDT_training, SDT_test, target_training, target_test = noshuffle(train_dir, test_dir)
    # /end Shyam

    net, train_test_splitter, training_time = training(device, num_epochs, SDT_training, target_training)    
    train_indices, valid_indices = next(train_test_splitter.split(SDT_training))
#    val_error = mean_absolute_error(target_training[valid_indices].reshape(-1), output_list[valid_indices].reshape(-1))
#    print('Validation error (eV): %s' %val_error)

    # Shyam start/
    output_list = net.predict(SDT_training)
    test_list = net.predict(SDT_test)

    train_error = mean_absolute_error(target_training[train_indices].reshape(-1),
                                      output_list[train_indices].reshape(-1))
    val_error = mean_absolute_error(target_training[valid_indices].reshape(-1),
                                      output_list[valid_indices].reshape(-1))
#    print(val_error
    test_error = mean_absolute_error(target_test.reshape(-1),
                                      test_list.reshape(-1))
    # /end Shyam

    ''' 
    train_error = mean_absolute_error(target_training[train_indices].reshape(-1), 
                                      net.predict(SDT_training)[train_indices].reshape(-1))
    val_error = mean_absolute_error(target_training[valid_indices].reshape(-1), 
                                      net.predict(SDT_training)[valid_indices].reshape(-1))
    test_error = mean_absolute_error(target_test.reshape(-1), 
                                      net.predict(SDT_test).reshape(-1))
    '''
    start = time.time()
    # Shyam start/
    predict_list = test_list
#    predict_list = net.predict(SDT_list[:num_docs])
    energy_list = target_test
    # /end Shyam
    measure_pred_time = mean_absolute_error(target_list[:num_docs].reshape(-1), 
                                      predict_list.reshape(-1))
    # Shyam start/
    with open('%s/prediction.pkl'%test_dir, 'wb') as pickle_file:
        pickle.dump(predict_list, pickle_file)
    with open('%s/energies.pkl'%test_dir, 'wb') as pick_file:
        pickle.dump(energy_list, pick_file)
    # /end Shyam
    end = time.time()
    pred_time = end - start
    times = (Docs_time, SDT_time, training_time, pred_time)
    errors = (train_error, val_error, test_error)
    return errors, times, SDT_training, total_docs

def Shyam_figure_of_merit(hyperparams, train_dir, test_dir, dataset, SDT_list, target_list, num_docs, num_SDT, num_epochs):
    device = get_device()
    return Shyam_predict(hyperparams, train_dir, test_dir, dataset, SDT_list, target_list, num_docs, num_SDT, num_epochs, device)

# def figure_of_merit(dataset, SDT_list, target_list, num_docs, num_SDT, num_epochs):
# Shyam start/
def figure_of_merit(train_dir, test_dir, dataset, SDT_list, target_list, num_docs, num_SDT, num_epochs):
# /end Shyam
    device = get_device()
    errors, times, SDT_training, total_docs = prediction(train_dir, test_dir, dataset, SDT_list, target_list, num_docs, num_SDT, num_epochs, device)
    Docs_time, SDT_time, training_time, pred_time = times
    train_error, val_error, test_error = errors
    print('\n')   
    print('BENCHMARK RESULTS')
    print('Current device:', device)
    print('Time to load %d documents: %f seconds\n' %(total_docs, Docs_time))
    print('Time to convert %d documents into SDT list: %f seconds' %(num_docs, SDT_time))
    print('Time to train the model using %d training examples for %d epochs: %f seconds' %(len(SDT_training), num_epochs, training_time))
    print('Training error: %f ev'  %train_error)
    print('Validation error: %f ev'  %val_error)
    print('Test error: %f ev'  %test_error)
    print('Time to predict energy for %d documents: %f seconds' %(num_docs, pred_time))
    with open('NESAP.md', 'w') as f:
        f.writelines('# Benchmark Test \n\n')
        f.writelines('## Measurement on Edison\n\n')
        f.writelines('```C\n') 
        f.writelines('sbatch run_benchmark.sh \n')
        f.writelines('```\n\n')
        f.writelines('## Benchmark Results \n')
        if device == 'cpu':
            f.writelines('Current device:' + device + '\n')
        else:
            f.writelines('Current device: ' + device.type + '\n')
        f.writelines('\nTime to load %d documents: %f seconds\n' %(total_docs, Docs_time))
        f.writelines('\nTime to convert %d documents into SDT list: %f seconds\n' %(num_docs, SDT_time))
        f.writelines('\nTime to train the model using %d training examples: %f seconds\n' %(len(SDT_training), training_time))
        f.writelines('\nTraining error: %f ev\n'  %train_error)
        f.writelines('\nValidation error: %f ev\n'  %val_error)
        f.writelines('\nTest error: %f ev\n'  %test_error)
        f.writelines('\nTime to predict energy for %d documents: %f seconds\n' %(num_docs, pred_time))
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./input/docs.pkl',
                       help = 'pickle file contains documents')
    parser.add_argument('--SDT_list', type=str, default='./input/SDT_list.pkl',
                       help = 'pickle file contains documents')
    parser.add_argument('--target_list', type=str, default='./input/target_list.pkl',
                       help = 'pickle file contains documents')
    parser.add_argument('--num_docs', type=int, default=1000,
                       help='number of docs to use')
    parser.add_argument('--num_SDT', type=int, default=20771,
                       help='number of docs to use')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='number of epochs for training')
    
    args = parser.parse_args()
    print(args)
    figure_of_merit(dataset=args.dataset,
                    SDT_list=args.SDT_list,
                    target_list=args.target_list,
                    num_docs=args.num_docs,
                    num_SDT=args.num_SDT,
                    num_epochs=args.num_epochs)
    
