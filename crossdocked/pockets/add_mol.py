import sys, os, gzip
import pickle
from argparse import ArgumentParser
import pandas as pd
import rdkit
from rdkit import Chem
WORKDIR = os.environ.get('WORKDIR', '/workspace')
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.lmdb import load_lmdb, new_lmdb
from src.utils.logger import get_logger, add_file_handler

parser = ArgumentParser()
parser.add_argument('--rank', type=int)
args = parser.parse_args()

out_dir = f".tmp/{args.rank}"
os.makedirs(out_dir, exist_ok=True)
env, txn = load_lmdb(f"{out_dir}/remove_mol.lmdb", readahead=True)
env_w, txn_w = new_lmdb(f"{out_dir}/add_mol.lmdb")
df = pd.read_csv(f'{WORKDIR}/cplm/preprocess/results/finetune/r4_all/filenames.csv.gz')
dnames = df['dname'].values
lnames = df['lig_name'].values
pnames = df['protein_name'].values
sdf_idxs = df['sdf_idx'].values

logger = get_logger(stream=True)
add_file_handler(logger, f"{out_dir}/add_mol.log")
logger.info(f"{rdkit.__version__=}")
assert rdkit.__version__.split('.')[0] == '2022'

# sdf cache
cur_dname = None
cur_lname = None
mol_supplier: Chem.SDMolSupplier = None

for i, (key, value) in enumerate(txn.cursor().iternext(), 1):
    data = pickle.loads(value)
    idx = int(key.decode('ascii'))
    dname = dnames[idx]
    lname = lnames[idx]
    
    # mol
    if (dname, lname) != (cur_dname, cur_lname):
        lig_path = f"{WORKDIR}/cheminfodata/crossdocked/CrossDocked2020/{dname}/{lname}"
        if os.path.exists(lig_path):
            mol_supplier = Chem.SDMolSupplier(lig_path)
        else:
            with gzip.open(lig_path+'.gz') as f:
                mol_supplier = list(Chem.ForwardSDMolSupplier(f))
    data['lig_mol'] = mol_supplier[sdf_idxs[idx]]
    data['dname'] = dname
    data['lig_name'] = lname
    data['protein_name'] = pnames[idx]
    data['sdf_idx'] = int(sdf_idxs[idx])
    txn_w.put(key, pickle.dumps(data))
    if i % 10000 == 0:
        logger.info(f"{i} finished.")

txn_w.commit()
env_w.close()
