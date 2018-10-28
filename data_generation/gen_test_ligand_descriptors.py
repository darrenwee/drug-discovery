from htmd.ui import *
from htmd.molecule.voxeldescriptors import *
from pathlib import Path
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
from pdb_parser import test_pdb_to_features
import sys

#import logging
#logger = logging.getLogger()
#logger.setLevel(logging.CRITICAL)

ligands = []
ligand_root = './processed_data/eval/ligands_2'
Path(ligand_root).mkdir(exist_ok = True, parents=True)


### Generate the ligands

raw_ligand_paths = sorted(list(Path('./data/testing_data/').glob('*lig_cg.pdb')))

dests = []
idxs = []
idx = 0
for lig_path in tqdm(raw_ligand_paths, file=sys.stdout):
    features, centers = test_pdb_to_features(lig_path)
    features = features[:,:,:,[0,7]]
    
    dest = ligand_root + '/' + str(idx+1).zfill(4) + '.pk'
    with open(dest, 'wb') as f:
        pk.dump((features, centers), f)
        dests.append(dest)
        idxs.append(idx+1)
        idx+=1

df = pd.DataFrame({'id':idxs, 'path': dests})
df.to_csv('./data/csv/ligand_2_data_eval.csv', index=None)
