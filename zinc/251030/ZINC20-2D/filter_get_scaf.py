import sys, os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
import rdkit
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.lmdb import load_lmdb
from src.data.lmdb import data_len_to_blen
from src.utils.logger import get_logger, add_file_handler


# cf. MolecularSets
def get_mol(smiles):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles, str):
        if len(smiles) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles

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

parser = ArgumentParser()
parser.add_argument('--rank', type=int, required=True)
parser.add_argument('--size', type=int, required=True)
args = parser.parse_args()

logger = get_logger(stream=True)
os.makedirs("raw/filter_get_scaf", exist_ok=True)
add_file_handler(logger, f"raw/filter_get_scaf/{args.rank}.log")
logger.info(f"{rdkit.__version__=}")
logger.info(f"{args=}")

env, txn = load_lmdb("smi.lmdb")

# get start & end
size = env.stat()['entries']
blen = data_len_to_blen(size)
idx_start = int((size*args.rank)/args.size)
idx_max = int((size*(args.rank+1))/args.size)-1
key_start = idx_start.to_bytes(blen)
key_max = idx_max.to_bytes(blen)
logger.info(f"{size=}, {key_start=}, {key_max=}")

cursor = txn.cursor()
cursor.set_key(key_start)
def get_mol(smiles):
    if len(smiles) == 0:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return mol

scaf2idxs = defaultdict(list)
for i, (key, value) in enumerate(cursor.iternext(), 1):
    idx = int.from_bytes(key)
    smi = value.decode('ascii')
    
    # process_molecule

    ## mol_pass_filters
    mol = get_mol(smi)
    if mol is None: continue
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    if smiles is None or len(smiles) == 0: continue
    if Chem.MolFromSmiles(smiles) is None: continue
    ## - mol_pass_filters
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    # - process_molecule

    scaf = compute_scaffold(smi)

    scaf2idxs[scaf].append(idx)

    if i % 100000 == 0:
        logger.info(f"Finished {i}")
    if key == key_max: break
with open(f"raw/filter_get_scaf/{args.rank}.pkl", 'wb') as f:
    pickle.dump(dict(scaf2idxs), f)
