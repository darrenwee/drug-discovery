from htmd.ui import *
from htmd.molecule.voxeldescriptors import *
from pathlib import Path
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

ligands = []
ligand_root = './processed_data/ligands_2'
Path(ligand_root).mkdir(exist_ok = True, parents=True)

raw_ligand_paths = sorted(list(Path('./data/filtered_data').glob('*lig_cg.pdb')))

dests = []
idx = 0
for idx, lig_path in tqdm(raw_ligand_paths):
    mol = Molecule(str(lig_path), keepaltloc='all')
    bb = htmd.molecule.util.boundingBox(mol)

    xx = (bb[1][0] + bb[0][0])/2 - 12
    yy = (bb[1][1] + bb[0][1])/2 - 12
    zz = (bb[1][2] + bb[0][2])/2 - 12

    centers = []
    for ix in range(24):
        for iy in range(24):
            for iz in range(24):
                centers.append([xx + ix, yy + iy, zz + iz])

    features, centers = getVoxelDescriptors(mol, usercenters=np.array(centers), voxelsize=1, method='CUDA')
    features = features.reshape(24, 24, 24, features.shape[1])
    features = features[:,:,:,[0,7]]
    
    dest = ligand_root + '/' + str(idx+1).zfill(4) + '.pk'
    with open(dest, 'wb') as f:
        pk.dump((features, centers), f)
        dests.append(dest)

        #print('done with '+str(idx+1))
    idx += 1

idxs = list(range(1,3001))
df = pd.DataFrame({'id':idxs, 'path': dests})
df.to_csv('ligand_2_data.csv', index=None)