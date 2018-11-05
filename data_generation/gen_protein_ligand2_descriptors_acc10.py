from pathlib import Path
import numpy as np
import pickle as pk
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
from htmd.ui import *
from htmd.molecule.voxeldescriptors import *
from multiprocessing import Pool 

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

test_size = 300
base_dest = './processed_data/test_acc10_2'
Path(base_dest).mkdir(exist_ok=True, parents=True)
df = pd.read_csv('./data/csv/test_lig_2_pro_pairs.csv')
idxs = sorted(list(df.lig_id.unique()))
idxs = idxs[:test_size]

'''
for lig_idx in tqdm(idxs):
    lig_src = './processed_data/ligands_2/' + str(lig_idx).zfill(4) + '.pk'
    with open(lig_src, 'rb') as f:
        lig_features, centers = pk.load(f)
    
    for pro_idx in idxs:
        dest = base_dest + '/' + str(pro_idx).zfill(4) + '_pro_' + str(lig_idx).zfill(4) + '_lig.npy'
        protein_path = './data/filtered_data/'+str(pro_idx).zfill(4)+'_pro_cg.pdb'

        mol = Molecule(protein_path, keepaltloc='all')

        pro_features, _ = getVoxelDescriptors(mol, usercenters=np.array(centers), voxelsize=1, method='CUDA')
        pro_features = pro_features.reshape(24, 24, 24, pro_features.shape[1])
        pro_features = pro_features[:,:,:,[0,7]]
        combined = np.concatenate((pro_features, lig_features), axis=3)
        combined = combined.astype(np.float32)

        np.save(dest, combined)

'''

# generate the csv
dests = []
ligs = []
proteins = []
score = []
for lig_idx in tqdm(idxs):
    lig_src = './processed_data/ligands/' + str(lig_idx).zfill(4) + '.pk'
    
    for pro_idx in idxs:
        dest = base_dest + '/' + str(pro_idx).zfill(4) + '_pro_' + str(lig_idx).zfill(4) + '_lig.npy'
        
        dests.append(dest)
        ligs.append(lig_idx)
        proteins.append(pro_idx)
        score.append(1 if lig_idx == pro_idx else 0)

csv_dest = f'./data/csv/test_acc10_2_{test_size}.csv'
pro_lig_record = pd.DataFrame({'lig_id': ligs, 'pro_id': proteins, 
                               'dest': dests, 'score': score}, index=None)
pro_lig_record.to_csv(csv_dest, index=None)
