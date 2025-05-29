from __future__ import print_function, division
import os
import csv
import re
import json
import functools
import random
import warnings

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from sklearn.preprocessing import OneHotEncoder
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.base import TransformerMixin
import mongo
import copy
# Shyam start/
import pickle
import pandas as pd
# /end Shyam

def collate_pool(dataset_list):
    """#
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    distances: torch.Tensor shape (N, 1)
      Storing connectivity information of atoms
    connection_atom_idx: torch.Tensor shape (N, 1)
      One hot encoding representation of the connectivity
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_distances = [], [], [], []
    crystal_atom_idx, batch_target = [], []
    connection_idx, connection_atom_idx = [], []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx, distances), target)\
            in enumerate(dataset_list):
            
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        
        #Creating Mask that only considers atoms with distance <= 2
        connection_idx = np.where(distances <= 3)[0]
        connection_base = np.zeros((n_i,1))
        connection_base[connection_idx] = 1
        connection_atom_idx.append(torch.FloatTensor(connection_base))

        batch_distances.append(distances)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        base_idx += n_i
    return {'atom_fea':torch.cat(batch_atom_fea, dim=0), 
            'nbr_fea':torch.cat(batch_nbr_fea, dim=0), 
            'nbr_fea_idx':torch.cat(batch_nbr_fea_idx, dim=0), 
            'crystal_atom_idx':crystal_atom_idx,
    # Shyam start/
    #        'crystal_atom_idx': torch.cat(crystal_atom_idx, dim=0),
    # /end Shyam
            'distances':torch.cat(batch_distances,dim=0),
            'connection_atom_idx':torch.cat(connection_atom_idx, dim=0)}, torch.FloatTensor(batch_target)

# Shyam start/
def collate_pool_validation(dataset_list):
    """ 
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    distances: torch.Tensor shape (N, 1)
      Storing connectivity information of atoms
    connection_atom_idx: torch.Tensor shape (N, 1)
      One hot encoding representation of the connectivity
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_distances = [], [], [], []
    crystal_atom_idx, batch_target = [], []
    connection_idx, connection_atom_idx = [], []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx, distances), target) \
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)

        # Creating Mask that only considers atoms with distance <= 2
        connection_idx = np.where(distances <= 3)[0]
        connection_base = np.zeros((n_i, 1))
        connection_base[connection_idx] = 1
        connection_atom_idx.append(torch.FloatTensor(connection_base))

        batch_distances.append(distances)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        base_idx += n_i
    return {'atom_fea': torch.cat(batch_atom_fea, dim=0),
            'nbr_fea': torch.cat(batch_nbr_fea, dim=0),
            'nbr_fea_idx': torch.cat(batch_nbr_fea_idx, dim=0),
           # 'crystal_atom_idx': crystal_atom_idx,
            # Shyam start/
            'crystal_atom_idx': torch.cat(crystal_atom_idx),
            # /end Shyam
            'distances': torch.cat(batch_distances, dim=0),
            'connection_atom_idx': torch.cat(connection_atom_idx, dim=0)}, torch.FloatTensor(batch_target)
# / end Shyam

class GaussianDistance(object):
    """#
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """#
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """#
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """#
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)

        # Shyam start/
        elem_embedding_1 = {}
        for key,value in elem_embedding.items():
            elem_embedding_1[int(key)] = value
        elem_embedding = elem_embedding_1
        # /end Shyam
        '''elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        '''
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class Shyam_StructureData():
    """#

    RE-COMMENT THIS


    Parameters
    ----------

    atoms_list: list of ASE atoms objects
        List of ASE atoms objects (final relaxed geometry)
    atoms_list_initial_config: list of ASE atoms objects
        List of ASE atoms objects (initial unrelaxed geometry)
        This is very important if the model will be used to predict
        the properties of unrelaxed structures
    atom_init_loc: str
        The location of the atom_init.json file that contains atomic properties
    * homo_list: list of numbers
        Contains the list of HOMO energies of the slabs
    * lumo_list: list of numbers
        Contains the list of LUMO energies of the slabs
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset
    use_voronoi: bool
        Controls whether the original (pair distance) or voronoi
        method from pymatgen is used to determine neighbor lists
        and distances.
    use_fixed_info: bool
        If True, add whether each atom is fixed by ASE constraints as an atomic feature.
        Hypothesized to improve the fit because there is information in the fixed
        atoms being in the bulk
    use_tag:
        If true, add the ASE tag as an atomic feature
    use_distance:
        If true, for each atom add a graph distance from the atom to the nearest atom
        on the graph that has a tag of 1 (indicated it is an adsorbate atom in our scheme).
        This allows atoms near the adsorbate to have a higher influence if the model
        deems it helpful.
    train_geometry: str
        If 'final', use the final relaxed structure for input to the graph
        If 'initial' use the initial unrelaxed structure
        If 'final-adsorbate', 'use the initial relax structure for everything with tag=0,
            but add a fixed-edge feature to adsorbate atoms in the final configuration.
            We did this so that the information from adsorbate movement (ex. on-top to bridge)
            is included in the input space, but the final relaxed bond distance is not included.
            This makes the method transferable to the predictions for unrelaxed structures with
            various adsorbate locations

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    """

    #    def __init__(self, atoms_list, atoms_list_initial_config, atom_init_loc, max_num_nbr=12, radius=8, dmin=0, step=0.2, random_seed=123, use_voronoi=True, use_fixed_info=False, use_tag=False, use_distance=False, train_geometry='final-adsorbate'):
    # Shyam start/
    def __init__(self, atoms_list, atoms_list_initial_config, atom_init_loc, homo_list, lumo_list, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123, use_voronoi=True, use_fixed_info=False, use_tag=False, use_distance=False,
                 train_geometry='final-adsorbate', use_Shyam_fea=None, use_Shyam_slab_fea=None):
        # /end Shyam
        # this copy is very important; otherwise things ran, but there was some sort
        # of shuffle that was affecting the real list, resulting in weird loss
        # loss functions and poor training
        # if use_Shyam_fea is None:
        #   use_Shyam_fea = ['polarizability', 'second-ionization']
        self.atoms_list = copy.deepcopy(atoms_list)
        self.atoms_list_initial_config = copy.deepcopy(atoms_list_initial_config)

        self.atom_init_loc = atom_init_loc
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.use_voronoi = use_voronoi

        # Load the atom features and gaussian distribution functions
        assert os.path.exists(self.atom_init_loc), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_loc)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        # Shyam start/
        self.use_Shyam_fea = use_Shyam_fea
        self.use_Shyam_slab_fea = use_Shyam_slab_fea
        self.homo_list = homo_list
        self.lumo_list = lumo_list
        # /end Shyam
        # Store some tags inside the object for later use
        self.use_fixed_info = use_fixed_info
        self.use_tag = use_tag
        self.use_distance = use_distance
        self.train_geometry = train_geometry  # could be initial, final, or final-adsorbate?

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, idx):
        atoms = copy.deepcopy(self.atoms_list[idx])
        crystal = AseAtomsAdaptor.get_structure(atoms)
        atoms_initial_config = copy.deepcopy(self.atoms_list_initial_config[idx])
        crystal_initial_config = AseAtomsAdaptor.get_structure(atoms_initial_config)

        # Stack the features from atom_init
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])

        # Shyam start/
        homo = self.homo_list[idx]
        lumo = self.lumo_list[idx]
        # /end Shyam

        # Shyam start/
        #    atom_fea = []
        #    for i in range(len(crystal)):
        #       at = crystal[i].specie.number
        #        temp = self.ari.get_atom_fea(at)
        #       atom_fea = np.vstack([temp])
        # /end Shyam
        # If use_tag=True, then add the tag as an atom feature
        if self.use_tag:
            atom_fea = np.hstack([atom_fea, atoms.get_tags().reshape((-1, 1))])
        # Shyam start/
        junk = 0
        # /end Shyam
        # If use_fixed_info=True, then add whether the atom is fixed by ASE constraint to the features
        if self.use_fixed_info:
            fix_loc, = np.where([type(constraint) == FixAtoms for constraint in atoms.constraints])
            fix_atoms_indices = set(atoms.constraints[fix_loc[0]].get_indices())
            fixed_atoms = np.array([i in fix_atoms_indices for i in range(len(atoms))]).reshape((-1, 1))
            atom_fea = np.hstack([atom_fea, fixed_atoms])

        # If use_voronoi, then use the voronoi connectivity from pymatgen to determine neighbors and distances
        if self.use_voronoi:

            # Get the connectivity array for the initial and final structure
            VC = VoronoiConnectivity(crystal)
            VC_initial_config = VoronoiConnectivity(crystal_initial_config)
            conn = copy.deepcopy(VC.connectivity_array)
            conn_initial_config = copy.deepcopy(VC_initial_config.connectivity_array)

            # Iterate through each atom, find it's neighbors, and add their distances
            all_nbrs = []

            # Loop over central atom
            for ii in range(0, conn.shape[0]):
                curnbr = []

                # Loop over neighbor atoms
                for jj in range(0, conn.shape[1]):

                    # Loop over each possible PBC image for the chosen image
                    for kk in range(0, conn.shape[2]):
                        # Only add as a neighbor if the atom is not the currently selected center one and there is connectivity
                        # to that image
                        if jj is not kk and conn[ii][jj][kk] != 0:

                            # Add the neighbor strength depending on train_geometry base
                            if self.train_geometry == 'initial':
                                curnbr.append(
                                    [ii, conn_initial_config[ii][jj][kk] / np.max(conn_initial_config[ii]), jj])
                            elif self.train_geometry == 'final':
                                curnbr.append([ii, conn[ii][jj][kk] / np.max(conn[ii]), jj])
                            elif self.train_geometry == 'final-adsorbate':
                                # In order for this to work, each adsorbate atom should be set to tag==1 in the atoms object
                                if (atoms.get_tags()[ii] == 1 or atoms.get_tags()[jj] == 1):
                                    if conn[ii][jj][kk] / np.max(conn[ii]) > 0.3:
                                        curnbr.append([ii, 1.0, jj])
                                    else:
                                        curnbr.append([ii, 0.0, jj])
                                else:
                                    curnbr.append(
                                        [ii, conn_initial_config[ii][jj][kk] / np.max(conn_initial_config[ii]), jj])
                            else:
                                curnbr.append([ii, conn[ii][jj][kk] / np.max(conn[ii]), jj])
                        else:
                            curnbr.append([ii, 0.0, jj])
                all_nbrs.append(np.array(curnbr))

            # If use_distance=True, then add the distance to an adsorbate (tag=1) as a feature
            if self.use_distance:
                distances, distances_OHE = distance_to_adsorbate_feature(atoms, VC)
                atom_fea = np.hstack([atom_fea, distances_OHE])

            else:
                distances = [0] * len(atoms)

            # Shyam start/
            if self.use_Shyam_fea:
                for fea in self.use_Shyam_fea:
                    Shyam_fea_vec = Shyam_add_feature(atoms, fea)
                    atom_fea = np.hstack([atom_fea, Shyam_fea_vec])

            else:
                Shyam_fea_vec = [0] * len(atoms)
            if self.use_Shyam_slab_fea:
                for feat in self.use_Shyam_slab_fea:
                    Shyam_slab_fea_vec = Shyam_add_slab_feature(atoms, feat, homo=homo, lumo=lumo)
                    atom_fea = np.hstack([atom_fea, Shyam_slab_fea_vec])

            else:
                Shyam_slab_fea_vec = [0] * len(atoms)

            # /end Shyam

            # Find the strongest neighbors for each atom
            all_nbrs = np.array(all_nbrs)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1], reverse=True) for nbrs in all_nbrs]
            nbr_fea_idx = np.array([list(map(lambda x: x[2],
                                             nbr[:self.max_num_nbr])) for nbr in all_nbrs])
            nbr_fea = np.array([list(map(lambda x: x[1], nbr[:self.max_num_nbr]))
                                for nbr in all_nbrs])

            # expand distance one-hot encoding with GDF
            nbr_fea = self.gdf.expand(nbr_fea)
        else:
            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.max_num_nbr:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                       [0] * (self.max_num_nbr - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                   [self.radius + 1.] * (self.max_num_nbr -
                                                         len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2],
                                                nbr[:self.max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1],
                                            nbr[:self.max_num_nbr])))
            nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
            nbr_fea = self.gdf.expand(nbr_fea)
            distances = [0] * len(atoms)

        try:
            nbr_fea = torch.Tensor(nbr_fea)
        except RuntimeError:
            print(nbr_fea)

        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        distances = torch.LongTensor(distances)
        atom_fea = torch.Tensor(atom_fea)
        #        Shyam_fea = torch.LongTensor(Shyam_fea_vec)
        #        return (atom_fea, nbr_fea, nbr_fea_idx, distances, work_fn)
        # /end Shyam

        # nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        # distances=torch.LongTensor(distances)
        # atom_fea = torch.Tensor(atom_fea)

        return (atom_fea, nbr_fea, nbr_fea_idx, distances)


class StructureData():
    """#
    
    RE-COMMENT THIS 
    
    
    Parameters
    ----------

    atoms_list: list of ASE atoms objects
        List of ASE atoms objects (final relaxed geometry)
    atoms_list_initial_config: list of ASE atoms objects
        List of ASE atoms objects (initial unrelaxed geometry)
        This is very important if the model will be used to predict
        the properties of unrelaxed structures
    atom_init_loc: str
        The location of the atom_init.json file that contains atomic properties
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset
    use_voronoi: bool
        Controls whether the original (pair distance) or voronoi
        method from pymatgen is used to determine neighbor lists 
        and distances.
    use_fixed_info: bool
        If True, add whether each atom is fixed by ASE constraints as an atomic feature.
        Hypothesized to improve the fit because there is information in the fixed
        atoms being in the bulk
    use_tag: 
        If true, add the ASE tag as an atomic feature
    use_distance:
        If true, for each atom add a graph distance from the atom to the nearest atom
        on the graph that has a tag of 1 (indicated it is an adsorbate atom in our scheme). 
        This allows atoms near the adsorbate to have a higher influence if the model
        deems it helpful.
    train_geometry: str
        If 'final', use the final relaxed structure for input to the graph
        If 'initial' use the initial unrelaxed structure
        If 'final-adsorbate', 'use the initial relax structure for everything with tag=0,
            but add a fixed-edge feature to adsorbate atoms in the final configuration.
            We did this so that the information from adsorbate movement (ex. on-top to bridge)
            is included in the input space, but the final relaxed bond distance is not included.
            This makes the method transferable to the predictions for unrelaxed structures with
            various adsorbate locations

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    """
    def __init__(self, atoms_list, atoms_list_initial_config, atom_init_loc, max_num_nbr=12, radius=8, dmin=0, step=0.2, random_seed=123, use_voronoi=True, use_fixed_info=False, use_tag=False, use_distance=False, train_geometry='final-adsorbate'):
        # this copy is very important; otherwise things ran, but there was some sort
        # of shuffle that was affecting the real list, resulting in weird loss
        # loss functions and poor training
        
        self.atoms_list = copy.deepcopy(atoms_list)
        self.atoms_list_initial_config = copy.deepcopy(atoms_list_initial_config)
        
        self.atom_init_loc = atom_init_loc
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.use_voronoi = use_voronoi
        
        #Load the atom features and gaussian distribution functions
        assert os.path.exists(self.atom_init_loc), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_loc)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
       
        #Store some tags inside the object for later use
        self.use_fixed_info = use_fixed_info
        self.use_tag = use_tag
        self.use_distance = use_distance
        self.train_geometry = train_geometry  # could be initial, final, or final-adsorbate?

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, idx):
        atoms = copy.deepcopy(self.atoms_list[idx])
        crystal = AseAtomsAdaptor.get_structure(atoms)
        atoms_initial_config = copy.deepcopy(self.atoms_list_initial_config[idx])
        crystal_initial_config = AseAtomsAdaptor.get_structure(atoms_initial_config)
        
        # Stack the features from atom_init
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])

        
        # If use_tag=True, then add the tag as an atom feature
        if self.use_tag:
            atom_fea = np.hstack([atom_fea,atoms.get_tags().reshape((-1,1))])
   
        # If use_fixed_info=True, then add whether the atom is fixed by ASE constraint to the features
        if self.use_fixed_info:
            fix_loc, = np.where([type(constraint)==FixAtoms for constraint in atoms.constraints])
            fix_atoms_indices = set(atoms.constraints[fix_loc[0]].get_indices())
            fixed_atoms = np.array([i in fix_atoms_indices for i in range(len(atoms))]).reshape((-1,1))
            atom_fea = np.hstack([atom_fea,fixed_atoms])

        # If use_voronoi, then use the voronoi connectivity from pymatgen to determine neighbors and distances
        if self.use_voronoi:
        
            #Get the connectivity array for the initial and final structure
            VC = VoronoiConnectivity(crystal)
            VC_initial_config = VoronoiConnectivity(crystal_initial_config)
            conn = copy.deepcopy(VC.connectivity_array)
            conn_initial_config = copy.deepcopy(VC_initial_config.connectivity_array)
            
            #Iterate through each atom, find it's neighbors, and add their distances
            all_nbrs = []
            
            # Loop over central atom
            for ii in range(0, conn.shape[0]):
                curnbr = []
                
                #Loop over neighbor atoms
                for jj in range(0, conn.shape[1]):
                
                    #Loop over each possible PBC image for the chosen image
                    for kk in range(0,conn.shape[2]):
                        # Only add as a neighbor if the atom is not the currently selected center one and there is connectivity
                        # to that image
                        if jj is not kk and conn[ii][jj][kk] != 0:
                        
                            #Add the neighbor strength depending on train_geometry base
                            if self.train_geometry =='initial':
                                curnbr.append([ii, conn_initial_config[ii][jj][kk]/np.max(conn_initial_config[ii]), jj])
                            elif self.train_geometry =='final':
                                curnbr.append([ii, conn[ii][jj][kk]/np.max(conn[ii]), jj])
                            elif self.train_geometry == 'final-adsorbate':
                                #In order for this to work, each adsorbate atom should be set to tag==1 in the atoms object
                                if (atoms.get_tags()[ii]==1 or atoms.get_tags()[jj]==1):
                                    if conn[ii][jj][kk]/np.max(conn[ii])>0.3:
                                        curnbr.append([ii, 1.0, jj])
                                    else:
                                        curnbr.append([ii, 0.0, jj])
                                else:
                                    curnbr.append([ii, conn_initial_config[ii][jj][kk]/np.max(conn_initial_config[ii]), jj])
                            else:
                                curnbr.append([ii, conn[ii][jj][kk]/np.max(conn[ii]), jj])
                        else:
                            curnbr.append([ii, 0.0, jj])
                all_nbrs.append(np.array(curnbr))
                
            # If use_distance=True, then add the distance to an adsorbate (tag=1) as a feature
            if self.use_distance:
                distances, distances_OHE = distance_to_adsorbate_feature(atoms, VC)
                atom_fea = np.hstack([atom_fea,distances_OHE])
                
            else:
                distances = [0]*len(atoms)

            # Find the strongest neighbors for each atom
            all_nbrs = np.array(all_nbrs)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1],reverse=True) for nbrs in all_nbrs]
            nbr_fea_idx = np.array([list(map(lambda x: x[2],
                                    nbr[:self.max_num_nbr])) for nbr in all_nbrs])
            nbr_fea = np.array([list(map(lambda x: x[1], nbr[:self.max_num_nbr]))
                                for nbr in all_nbrs])
                                
            # expand distance one-hot encoding with GDF
            nbr_fea = self.gdf.expand(nbr_fea)
        else:
            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.max_num_nbr:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                       [0] * (self.max_num_nbr - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                   [self.radius + 1.] * (self.max_num_nbr -
                                                         len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2],
                                                nbr[:self.max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1],
                                            nbr[:self.max_num_nbr])))
            nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
            nbr_fea = self.gdf.expand(nbr_fea)
            distances = [0]*len(atoms)

        try:
            nbr_fea = torch.Tensor(nbr_fea)
        except RuntimeError:
            print(nbr_fea)

        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        distances=torch.LongTensor(distances)
        atom_fea = torch.Tensor(atom_fea)

        return (atom_fea, nbr_fea, nbr_fea_idx, distances)


class ListDataset():
    def __init__(self, list_in):
        self.list = list_in
        
    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        return self.list[idx]

# Shyam start/
class Shyam_StructureDataTransformer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return

    def transform(self,X):
        structure_list = []
        homo_list = []
        lumo_list = []
        for doc in X:
            mon = mongo.make_atoms_from_doc(doc)
            #print(mon)
            structure_list.append(mon)
            homo_list.append(doc['HOMO'])
            lumo_list.append(doc['LUMO'])
        #structure_list = [mongo.make_atoms_from_doc(doc) for doc in X]
        structure_list_orig = [mongo.make_atoms_from_doc(doc['initial_configuration']) for doc in X]

        SD = Shyam_StructureData(structure_list, structure_list_orig, homo_list=homo_list, lumo_list=lumo_list,
                                 **self.kwargs)
        return SD

    def fit(self,*_):
        return self
# /end Shyam

class StructureDataTransformer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return

    def transform(self,X):
#        structure_list = [mongo.make_atoms_from_doc(doc) for doc in X]
        structure_list = [mongo.make_atoms_from_doc(doc['final_struct']) for doc in X]
#        structure_list_orig = [mongo.make_atoms_from_doc(doc['initial_configuration']) for doc in X]
        structure_list_orig = [mongo.make_atoms_from_doc(doc['final_struct']) for doc in X]

        SD = StructureData(structure_list, structure_list_orig, *self.args, **self.kwargs)
        return SD
 
    def fit(self,*_):
        return self

# Shyam start/
class StructureDataTransformer_ocp(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return

    def transform(self,X):
        structure_list = [mongo.make_atoms_from_doc_ocp(doc['final_struct']) for doc in X]
        structure_list_orig = [mongo.make_atoms_from_doc_ocp(doc['final_struct']) for doc in X]

        SD = StructureData(structure_list, structure_list_orig, *self.args, **self.kwargs)
        return SD
 
    def fit(self,*_):
        return self
# /end Shyam

class MergeDataset(torch.utils.data.Dataset):
    #Simple custom dataset to combine two datasets 
    # (one for input X, one for label y)
    def __init__(
            self,
            X,
            y,
            length=None,
    ):

        self.X = X
        self.y = copy.deepcopy(y)

        len_X = len(X)
        if y is not None:
            len_y = len(y)
            if len_y != len_X:
                raise ValueError("X and y have inconsistent lengths.")
        self._len = len_X

    def __len__(self):
        return self._len

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, i):
        X, y = self.X, self.y
        
        if y is not None:
            yi = copy.deepcopy(y[i])
        else:
            yi = np.nan

        return X[i], yi
# Shyam start/
def Shyam_add_feature(atoms, feature):
    '''

    Parameters
    ----------
    atoms - ase.Atoms object

    Returns
    -------
    The additional feature to be inserted to atom_fea in SDT

    '''
    total_atoms = len(atoms)
    atomic_nos = atoms.get_atomic_numbers()  # ASE object datamember cannot be accessed from outside;  use member functions
    if feature == 'polarizability':
        data = json.load(open('/mnt/hdd1/sragha20/cgcnn-manuscript-master/input/atom_polarizability.json', 'r'))
        pol_vec = [data[str(no)] for no in atomic_nos]      # Vector with polarizabilities
        indices = [i for i in range(1,119)]
        dataframe = pd.DataFrame(data.values(), index=indices, columns=['Polarizability'])
#        temp, bins = pd.cut(dataframe['Polarizability'], 10, retbins=True)
        temp, bins = pd.qcut(dataframe['Polarizability'], 10, retbins=True)
#        df = pd.DataFrame(pol_vec, index=range(1,total_atoms+1), columns=['Polarizability'])
        categs = ['(1.38 - 20.85]', '(20.85 - 33.3]', '(33.3 - 40.20000000000002]',
                  '(40.20000000000002 - 50.0]', '(50.0 - 67.0]', '(67.0 - 97.20000000000002]',
                  '(97.20000000000002 - 128.60000000000002]', '(128.60000000000002 - 159.8]',
                  '(159.8 - 205.89999999999998]', '(205.89999999999998 - 401.00]']
        '''
        Equal interval length categories
        categs = ['(0.98038 - 41.342]', '(41.342 - 81.304]', '(81.304 - 121.266]', '(121.266 - 161.228]',
                  '(161.228 - 201.19]', '(201.19 - 241.152]', '(241.152 - 281.114]', '(281.114 - 321.076]',
                  '(321.076 - 361.038]', '(361.038 - 401.00]']
        '''
        fea_vec = pd.cut(pol_vec, bins=bins, labels=categs)
        one_hot = pd.get_dummies(fea_vec)
        return one_hot

    if feature == 'second-ionization':
        data = json.load(open('/mnt/hdd1/sragha20/cgcnn-manuscript-master/input/gen_input/atomic_data/2nd_ionization.json', 'r'))
#        second_ionization_vec = [data[str(no)] for no in atomic_nos]
        second_ionization_vec = []
        for no in atomic_nos:
            if no == 1:
                second_ionization_vec.append(0)
            else:
                second_ionization_vec.append(data[str(no)])
        indices = [i for i in range(1,108)]
        dataframe = pd.DataFrame(data.values(), index=indices, columns=['2nd Ionization Energy'])
#        temp, bins = pd.cut(dataframe['2nd Ionization Energy'], 10, retbins = True)
        temp, bins = pd.qcut(dataframe['2nd Ionization Energy'], 10, retbins = True)
        categs = ['(10.003826 - 11.5]', '(11.5 - 11.9328]', '(11.9328 - 12.719816000000002]',
                  '(12.719816000000002 - 14.633442]', '(14.633442 - 16.37]', '(16.37 - 17.340000000000003]',
                  '(17.340000000000003 - 18.771214]', '(18.771214 - 20.883028000000007]',
                  '(20.883028000000007 - 24.691824400000005]', '(24.691824400000005 - 75.640097]']
        '''
        Equal interval length categories
        categs = ['(9.93818973 - 16.5674531]', '(16.5674531 - 23.1310802]', '(23.1310802 - 29.6947073]',
                  '(29.6947073 - 36.2583344]', '(36.2583344 - 42.8219615]', '(42.8219615 - 49.3855886]',
                  '(49.3855886 - 55.9492157]', '(55.9492157 - 62.5128428]', '(62.5128428 - 69.0764699]',
                  '(69.0764699 - 75.640097]']
        '''
        dataframe['Category'] = pd.cut(dataframe['2nd Ionization Energy'], bins, labels = categs)
        fea_vec = pd.cut(second_ionization_vec, bins=bins, labels=categs)
        one_hot = pd.get_dummies(fea_vec)
        return one_hot

# /end Shyam

def Shyam_add_slab_feature(atoms, feature_name, homo=10000, lumo=10000):
    '''

    Parameters
    ----------
    atoms - ase.Atoms object
    feature_name - str
    homo - energy of highest occupied molecular orbital
    lumo - energy of lowest unoccupied molecular orbital

    Returns
    -------
    The one-hot vector corresponding to the feature, to be added to EACH atom
    '''
    atomic_nos = atoms.get_atomic_numbers()
    if feature_name == 'HOMO':
        homo_list = []
        for a_no in atomic_nos:
            homo_list.append(homo)
        bins = [-0.348699, -0.261676, -0.220603, -0.202272, -0.195511, -0.172056, -0.161308, -0.153347,
                -0.14445,  -0.102545, -0.078699]
        categories = ['[-0.348699, -0.261676)','[-0.261676, -0.220603)', '[-0.220603, -0.202272)',
                      '[-0.202272, -0.195511)', '[-0.195511, -0.172056)', '[-0.172056, -0.161308)',
                      '[-0.161308, -0.153347)', '[-0.153347,  -0.14445)', '[-0.14445, -0.102545)',
                      '[-0.102545, -0.078699)']
        fea_vec = pd.cut(homo_list, bins=bins, labels=categories)
        one_hot = pd.get_dummies(fea_vec)
        ret_vec = np.array(one_hot)
        return ret_vec

    if feature_name == 'LUMO':
        lumo_list = []
        for a_no in atomic_nos:
            lumo_list.append(lumo)
        bins = [-0.335189, -0.245806, -0.210375, -0.197497, -0.181413, -0.162334, -0.160771, -0.153293,
               -0.141411, -0.102545, -0.078699]
        categories = ['[-0.335189, -0.245806)', '[-0.245806, -0.210375)', '[-0.210375, -0.197497)',
                      '[-0.197497, -0.181413)', '[-0.181413, -0.162334)', '[-0.162334, -0.160771)',
                      '[-0.160771, -0.153293)', '[-0.153293,  -0.141411)', '[-0.141411, -0.102545)',
                      '[-0.102545, -0.078699)']
        fea_vec = pd.cut(lumo_list, bins=bins, labels=categories)
        one_hot = pd.get_dummies(fea_vec)
        ret_vec = np.array(one_hot)
        return ret_vec

def distance_to_adsorbate_feature(atoms, VC, max_dist = 6):    
    # This function looks at an atoms object and attempts to find
    # the minimum distance from each atom to one of the adsorbate 
    # atoms (marked with tag==1)
    conn = copy.deepcopy(VC.connectivity_array)
    conn = np.max(conn,2)

    for i in range(len(conn)):
        conn[i]=conn[i]/np.max(conn[i])

    #get a binary connectivity matrix
    conn=(conn>0.3)*1
    
    #Everything is connected to itself, so add a matrix with zero on the diagonal 
    # and a large number on the off-diagonal
    ident_connection = np.eye(len(conn))
    ident_connection[ident_connection==0]=max_dist+1
    ident_connection[ident_connection==1]=0

    #For each distance, add an array of atoms that can be connected at that distance
    arrays = [ident_connection]
    for i in range(1,max_dist):
        arrays.append((np.linalg.matrix_power(conn,i)>=1)*i+(np.linalg.matrix_power(conn,i)==0)*(max_dist+1))

    #Find the minimum distance from each atom to every other atom (over possible distances)
    arrays=np.min(arrays,0)

    # Find the minimum distance from one of the adsorbate atoms to the other atoms
    min_distance_to_adsorbate = np.min(arrays[atoms.get_tags()==1],0).reshape((-1,1))
    
    # Make sure all of the one hot distance vectors are encoded to the same length.
    # Encode, return
    min_distance_to_adsorbate[min_distance_to_adsorbate>=max_dist]=max_dist-1
    OHE = OneHotEncoder(categories=[range(max_dist)]).fit(min_distance_to_adsorbate)
    return min_distance_to_adsorbate, OHE.transform(min_distance_to_adsorbate).toarray()

 

