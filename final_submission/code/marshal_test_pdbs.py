from pathlib import Path
from pdb_parser import _marshall_test_pdb
from tqdm import tqdm

# this file exists to fix a race condition with our parallelized code
# when generating protein-lig pairs for the test set.

test_pdbs = list(Path('./data/testing_data/').glob('*.pdb'))
for pdb in tqdm(test_pdbs):
    _marshall_test_pdb(pdb)
