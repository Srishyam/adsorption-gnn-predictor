"""Module to generate dataset from cgcnn."""
import sys
sys.path.append('/mnt/hdd1/sragah20/.local/lib/python3.8/site-packages/jarvis')
from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms
# from jarvis.core.atoms import Atoms_shyam

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.inputs import Poscar
import pickle
from tqdm import tqdm
#import random

#ds = pickle.load(open('../run_02/cgcnn_h_data.pkl','rb'))
data_loc = sys.argv[1]
ds = pickle.load(open(data_loc,'rb'))
#data = ds
#max_samples = 3000
#max_samples = 300
max_samples = len(ds)
data = ds[:max_samples]
#dataset_dir = "/mnt/hdd1/sragha20/alignn-main/alignn/run_02_1"
dataset_dir = sys.argv[2]
init_or_final_str = sys.argv[3]
count = 0
f = open("%s/id_prop.csv" % dataset_dir, "w")
for d in tqdm(data):
	mpid = d['mpid']
	miller_string = "".join(str(i) for i in d['miller'])
	posc_name = '%s/poscar_%s.vasp' % (dataset_dir,count)
	count += 1
	target = d['energy']
	f.write('%s,%6f\n' % (posc_name,target))
	atom_list = d['final_struct']['atoms'] if init_or_final_str == 'final' else d['init_struct']['atoms']['atoms']
#	atom_list = d['atoms']['atoms']

	specie_list = list()
	coords_list = list()
#	lattice = d['init_struct'].cell
	lattice = d['final_struct']['cell'] if init_or_final_str == 'final' else d['init_struct']['atoms']['cell']
	for at in atom_list:
		specie_list.append(Element(at['symbol']))
		coords_list.append(at['position'])
	py_struct = Structure(lattice, specie_list, coords_list, coords_are_cartesian=True)
	py_struct_pos_obj = Poscar(structure=py_struct, selective_dynamics=None, true_names=True, velocities=None, predictor_corrector=None, predictor_corrector_preamble=None, sort_structure=False)
	with open(posc_name, 'wt') as pf:
		pf.write(py_struct_pos_obj.get_string(direct=False))
f.close()


''' 
dft_3d = jdata("dft_3d")
prop = "optb88vdw_bandgap"
#max_samples = 50
for i in dft_3d:
    atoms = Atoms.from_dict(i["atoms"])
    jid = i["jid"]
    poscar_name = dataset_dir + "/POSCAR-" + jid + ".vasp"
    target = i[prop]
    if target != "na":
        atoms.write_poscar(poscar_name)
        f.write("%s,%6f\n" % (poscar_name, target))
        count += 1
        if count == max_samples:
            break
'''
