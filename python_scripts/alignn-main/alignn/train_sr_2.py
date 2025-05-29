"""Ignite training script.

from the repository root, run
`PYTHONPATH=$PYTHONPATH:. python alignn/train.py`
then `tensorboard --logdir tb_logs/test` to monitor results...
"""

from functools import partial

# from pathlib import Path
from typing import Any, Dict, Union
import ignite
import torch

from ignite.contrib.handlers import TensorboardLogger
try:
    from ignite.contrib.handlers.stores import EpochOutputStore
    # For different version of pytorch-ignite
except Exception as exp:
    from ignite.handlers.stores import EpochOutputStore

    pass
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from alignn import models
### shafigh
import sys
sys.path.append('/mnt/hdd1/sragha20/alignn-main/alignn')
#from data import get_train_val_loaders
from data_shafigh import get_train_val_loaders
from config import TrainingConfig
import torch.multiprocessing as mp
###
#from alignn.data import get_train_val_loaders
#from alignn.config import TrainingConfig
from alignn.models.alignn import ALIGNN
from alignn.models.alignn_layernorm import ALIGNN as ALIGNN_LN
from alignn.models.modified_cgcnn import CGCNN
from alignn.models.dense_alignn import DenseALIGNN
from alignn.models.densegcn import DenseGCN
from alignn.models.icgcnn import iCGCNN
from alignn.models.alignn_cgcnn import ACGCNN
from jarvis.db.jsonutils import dumpjson
import json
import pprint

import os
import torch.distributed as dist
import ignite
import ignite.distributed as idist

# from sklearn.decomposition import PCA, KernelPCA
# from sklearn.preprocessing import StandardScaler

# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
    #device = torch.device("cuda")
    ### shafigh
#if torch.cuda.is_available():
#    device=idist.device()
#    print('train_sr.py device\n')
#    print(device)
#    print('\n\n\n')
    ### shafigh

def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(sc.transform(y_pred.cpu().numpy()), device=device)
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=device)
    # pc = pk.load(open("pca.pkl", "rb"))
    # y_pred = torch.tensor(pc.transform(y_pred), device=device)
    # y = torch.tensor(pc.transform(y), device=device)

    # y_pred = torch.tensor(pca_sc.inverse_transform(y_pred),device=device)
    # y = torch.tensor(pca_sc.inverse_transform(y),device=device)
    # print (y.shape,y_pred.shape)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        #s.settimeout(30)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def setup(rank, world_size, port):
    # Shyam/
    #os.environ["MASTER_ADDR"] = "localhost"
    print('shyam_setup_1\n\n\n')
    # /Shyam
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    #os.environ["MASTER_PORT"] = "12355"
    os.environ["MASTER_PORT"] = port
    #os.environ["MASTER_PORT"] = "8888"
    # Shyam/
    print('shyam_setup_2')
    print('rank = ' + str(rank) + ' world size = ' + str(world_size) + ' port = ' + port + '\n\n\n')
    # /Shyam
    # Shyam/
    #mp.spawn(train, nprocs=2)
    # /Shyam
    #os.environ["MASTER_PORT"] = "12355"
    # initialize the process group
    #dist.init_process_group("gloo", rank=rank, world_size=world_size)
    #dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # /Shyam
    #print(world_size)
    #print(shyam)
    # /Shyam
    # Shyam/
    print('shyam_setup_3\n\n\n')
    # /Shyam

#def train_dgl(
### shafigh
#def train_dgl(local_rank,


if __name__ == "__main__":
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_dgl(config, progress=True)
