from htmd.ui import *
from htmd.molecule.voxeldescriptors import *
from pathlib import Path
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
from pdb_parser import pdb_to_features
import sys

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

'''
This script creates 24x24x24x2 descriptors from ligands,
and the spatial coordinates the 24x24x24 features are build from
The coordinates are required to create the 24x24x24x2 descriptors when pairing with proteins
'''

ligands = []
ligand_root = './processed_data/ligands_2'
Path(ligand_root).mkdir(exist_ok = True, parents=True)
csv_root = './data/csv/'
Path(csv_root).mkdir(exist_ok = True, parents=True)

raw_ligand_paths = sorted(list(Path('./data/training_data').glob('*lig_cg.pdb')))

dests = []
idxs = []

idx = 0
for lig_path in tqdm(raw_ligand_paths, file=sys.stdout):
    features, centers = pdb_to_features(lig_path)
    # 0 is the hydrophobic, 7 is excluded vol feature
    features = features[:,:,:,[0,7]]
    
    dest = ligand_root + '/' + str(idx+1).zfill(4) + '.pk'
    with open(dest, 'wb') as f:
        # reuse the voxel coordinates for the ligand
        pk.dump((features, centers), f)
        dests.append(dest)
        idxs.append(idx+1)
        idx+=1

df = pd.DataFrame({'id':idxs, 'path': dests})
df.to_csv('./data/csv/ligand_2_data.csv', index=None)
