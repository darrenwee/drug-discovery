from htmd.ui import *
from htmd.molecule.voxeldescriptors import *
from pathlib import Path
import numpy as np
import pickle as pk
from multiprocessing import Pool
import pandas as pd

import multiprocessing
from functools import partial

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

'''
This script creates 24x24x24x4 descriptors from protein-ligand pairs, used for regression training
It uses the coordinates from the ligand descriptors.
Train & test splitting of the dataset also occurs here.
'''

np.random.seed = 1
num_random_proteins = 4

ligand_df = pd.read_csv('./data/csv/ligand_2_data.csv', index_col=False)

# split ligands here
train_size = int(len(ligand_df) * 0.90)
train_df = ligand_df.iloc[:train_size]
test_df = ligand_df.iloc[train_size:]

def gen_data(df, base_dest):
    Path(base_dest).mkdir(exist_ok=True, parents=True)
    
    all_lig_idxs = []
    all_protein_idxs = []
    dests = []
    score = []

    # generates the protein-ligand pairs
    def gen_pro_lig_voxel_pairs(lig_idx, protein_idxs, path):
        with open(path, 'rb') as f:
            lig_features, centers = pk.load(f)

        for pidx in protein_idxs:
            mol = Molecule('./data/training_data/'+str(pidx).zfill(4)+'_pro_cg.pdb', keepaltloc='all')
            
            # same process as creating ligand features (see gen_liga_descriptors_2 for more info)
            pro_features, _ = getVoxelDescriptors(mol, usercenters=np.array(centers), voxelsize=1, method='CUDA')
            pro_features = pro_features.reshape(24, 24, 24, pro_features.shape[1])
            pro_features = pro_features[:,:,:,[0,7]]
            combined = np.concatenate((pro_features, lig_features), axis=3)
            combined = combined.astype(np.float32)

            dest = base_dest + '/' + str(pidx).zfill(4) + '_pro_' + str(lig_idx).zfill(4) + '_lig.npy'
            np.save(dest, combined)

            all_lig_idxs.append(lig_idx)
            all_protein_idxs.append(pidx)
            dests.append(dest)
            # score is 1 if the ligand id == protein id
            score.append(1 if pidx == lig_idx else 0)
            
    for row in df.itertuples():
        df_idx = row[0]
        lig_idx = row[1]
        path = row[2]
        
        # generate random proteins to pair
        r = range(0,len(df)-1)
        selection = np.random.choice(r, num_random_proteins, replace=False)
        selection = [len(df)-1 if s == df_idx else s for s in selection]
        # convert them into protein idxs
        protein_idxs = list(df.iloc[selection].id.values)
        protein_idxs.append(lig_idx)
        
        gen_pro_lig_voxel_pairs(lig_idx, protein_idxs, path)
        
    pro_lig_record = pd.DataFrame({'lig_id': all_lig_idxs, 'pro_id': all_protein_idxs, 'dests': dests, 'score': score}, index=None)
    return pro_lig_record

train_dest = './processed_data/pro_lig_2_voxels/train'
test_dest = './processed_data/pro_lig_2_voxels/test'

def parallelize_dataframe(df, func):
    num_cores = multiprocessing.cpu_count()-1  #leave one free to not freeze machine
    num_partitions = num_cores #number of partitions to split dataframe
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

test_lig_pro_pairs = parallelize_dataframe(test_df, partial(gen_data, base_dest=test_dest))
test_lig_pro_pairs.to_csv('./data/csv/val_lig_2_pro_pairs.csv')

train_lig_pro_pairs = parallelize_dataframe(train_df, partial(gen_data, base_dest=train_dest))
train_lig_pro_pairs.to_csv('./data/csv/train_lig_2_pro_pairs.csv')