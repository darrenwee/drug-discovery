from htmd.ui import *
from htmd.molecule.voxeldescriptors import *
from pathlib import Path
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
from pdb_parser import test_pdb_to_features
import sys

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

from multiprocessing import Pool
import time
from tqdm import *

def imap_unordered_bar(func, args):
    p = Pool()
    res_list = []
    with tqdm(total = len(args), file=sys.stdout) as pbar:
        for i, res in enumerate(p.imap_unordered(func, args)):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list

### Generate the protein-ligand complexes

base_dest = './processed_data/eval/test_acc10_2'
Path(base_dest).mkdir(exist_ok=True, parents=True)
df = pd.read_csv('./data/csv/ligand_2_data_eval.csv')
idxs = sorted(list(df.id.unique()))

# generate the data & csv

def gen(lig_idx):
    dests = []
    ligs = []
    proteins = []

    lig_src = './processed_data/eval/ligands_2/' + str(lig_idx).zfill(4) + '.pk'
    with open(lig_src, 'rb') as f:
        lig_features, centers = pk.load(f)
    
    for pro_idx in idxs:
        dest = base_dest + '/' + str(pro_idx).zfill(4) + '_pro_' + str(lig_idx).zfill(4) + '_lig.npy'
        
        protein_path = './data/testing_data/'+str(pro_idx).zfill(4)+'_pro_cg.pdb'
        
        pro_features, _ = test_pdb_to_features(protein_path, centers)
        pro_features = pro_features[:,:,:,[0,7]]

        combined = np.concatenate((pro_features, lig_features), axis=3)
        combined = combined.astype(np.float32)

        np.save(dest, combined)
        
        dests.append(dest)
        ligs.append(lig_idx)
        proteins.append(pro_idx)
    
    return dests, ligs, proteins

mess_of_outputs = imap_unordered_bar(gen, idxs)

all_dests = []
all_ligs = []
all_proteins = []

for mess in mess_of_outputs:
    all_dests.extend(mess[0])
    all_ligs.extend(mess[1])
    all_proteins.extend(mess[2])

csv_dest = f'./data/csv/eval_acc10_2.csv'
pro_lig_record = pd.DataFrame({'lig_id': all_ligs, 'pro_id': all_proteins, 
                               'dest': all_dests,}, index=None)
pro_lig_record.to_csv(csv_dest, index=None)
