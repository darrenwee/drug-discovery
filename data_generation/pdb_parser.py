from htmd.ui import Molecule
import htmd.molecule.util
import htmd.molecule.voxeldescriptors
import numpy as np

def pdb_to_features(filepath):
    mol = _pdb_to_molecule(filepath)
    return _molecule_to_features(mol)

def test_pdb_to_features(filepath, centers=[]):
    mol = _test_pdb_to_molecule(filepath)
    return _molecule_to_features(mol, centers)

def _pdb_to_molecule(filepath):
    return Molecule(str(filepath), keepaltloc='all')


def _test_pdb_to_molecule(filepath):
    out_file = _marshall_test_pdb(filepath)
    return _pdb_to_molecule(out_file)


def _marshall_test_pdb(filepath):
    with open(filepath, 'r') as file:
        strline_L = file.readlines()
    out_contents = []
    for i, strline in enumerate(strline_L):
        stripped_line = strline.strip()
        x, y, z, atom = stripped_line.split('\t')
        pdb_line = 'ATOM  %5s%19s%8s%8s%8s%22s%2s' % (i, '', x, y, z, '', _obscured_atom_type_to_element(atom))
        out_contents.append(pdb_line)

    out_file = str(filepath).replace('.pdb', 't.pdb')
    with open(out_file, 'w') as f:
        for item in out_contents:
            f.write("%s\n" % item)

    return out_file


def _obscured_atom_type_to_element(atomtype):
    if str(atomtype).lower() == 'h':
        return 'C'
    return 'N'


def _molecule_to_features(mol, centers=[]):
    bb = htmd.molecule.util.boundingBox(mol)

    xx = (bb[1][0] + bb[0][0])/2 - 12
    yy = (bb[1][1] + bb[0][1])/2 - 12
    zz = (bb[1][2] + bb[0][2])/2 - 12
    
    if len(centers) == 0:
        centers = []
        for ix in range(24):
            for iy in range(24):
                for iz in range(24):
                    centers.append([xx + ix, yy + iy, zz + iz])
    features, centers = htmd.molecule.voxeldescriptors.getVoxelDescriptors(mol, usercenters=np.array(centers),
                                                                           voxelsize=1)
    features = features.reshape(24, 24, 24, features.shape[1])
    return features, centers

