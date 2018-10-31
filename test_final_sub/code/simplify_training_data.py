from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

# script to simplify complex polar atoms to N
dest_root = './data/simplified_train_data'

Path(dest_root).mkdir(exist_ok=True)
src_root = Path('./data/training_data/')
pdbs = list(src_root.glob('*.pdb'))

def replace_non_C_with_N(pdb):
    text = pdb.read_text()
    lines = text.split('\n')
    dup_lines = deepcopy(lines)
    
    for idx, line in enumerate(dup_lines[:-1]):
        atom = line[66:]
        if atom.strip() != 'C':
            line = line[:66] + ' '*11 + 'N'
            lines[idx] = line
    fixed_text = '\n'.join(lines)
    dest = Path(dest_root+'/'+pdb.name)
    dest.write_text(fixed_text)


for pdb in tqdm(pdbs):
    replace_non_C_with_N(pdb)
