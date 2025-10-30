import sys, os
from glob import glob
from argparse import ArgumentParser
import rdkit
from rdkit import Chem
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.lmdb import new_lmdb
from src.data.lmdb import data_len_to_blen
from src.utils.logger import get_logger, add_file_handler

parser = ArgumentParser()
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--size', type=int, default=0)
args = parser.parse_args()

logger = get_logger(stream=True)
add_file_handler(logger, "raw/filter_get_scaf.log")
logger.info(f"{rdkit.__version__=}")

# From MolecularSets
def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def mol_passes_filters(mol,
                       allowed=None,
                       isomericSmiles=False):
    """
    Checks if mol
    * passes MCF and PAINS filters,
    * has only allowed atoms
    * is not charged
    """
    mol = get_mol(mol)
    if mol is None:
        return False
    """
    allowed = allowed or {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(
            len(x) >= 8 for x in ring_info.AtomRings()
    ):
        return False
    h_mol = Chem.AddHs(mol)
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    if smiles is None or len(smiles) == 0:
        return False
    if Chem.MolFromSmiles(smiles) is None:
        return False
    return True

def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()

def compute_scaffold(mol, min_rings=2):
    mol = get_mol(mol)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles

def process_molecule(mol_row, isomeric):
    mol_row = mol_row.decode('utf-8')
    smiles, _id = mol_row.split()
    if not mol_passes_filters(smiles):
        return None
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles),
                              isomericSmiles=isomeric)
    return _id, smiles


def filter_lines(lines, n_jobs, isomeric):
    logger.info('Filtering SMILES')
    with Pool(n_jobs) as pool:


        process_molecule_p = partial(process_molecule, isomeric=isomeric)
        dataset = [
            x for x in tqdm(
                pool.imap_unordered(process_molecule_p, lines),
                total=len(lines),
                miniters=1000
            )
            if x is not None
        ]
        dataset = pd.DataFrame(dataset, columns=['ID', 'SMILES'])
        dataset = dataset.sort_values(by=['ID', 'SMILES'])
        dataset = dataset.drop_duplicates('ID')
        dataset = dataset.sort_values(by='ID')
        dataset = dataset.drop_duplicates('SMILES')
        dataset['scaffold'] = pool.map(
            compute_scaffold, dataset['SMILES'].values
        )
    return dataset