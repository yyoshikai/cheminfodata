import sys, os
import pickle
from argparse import ArgumentParser
import torch
import rdkit
from rdkit.Chem.Scaffolds import MurckoScaffold
WORKDIR = os.environ.get('WORKDIR', '/workspace')
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.lmdb import load_lmdb, new_lmdb
from src.utils.logger import get_logger, add_file_handler
assert rdkit.__version__.split('.')[0] == '2022'

parser = ArgumentParser()
parser.add_argument('--rank', type=int)
args = parser.parse_args()

out_dir = f".tmp/{args.rank}"
os.makedirs(out_dir, exist_ok=True)

logger = get_logger(stream=True)
add_file_handler(logger, f"{out_dir}/get_scaf.log")
logger.info(f"{rdkit.__version__=}")

env, txn = load_lmdb(f"{out_dir}/add_mol.lmdb", readahead=True)
env_w, txn_w = new_lmdb(f"{out_dir}/scaf.lmdb")
test_idxs = []

data_names = torch.load("../targetdiff/split_by_name.pt")
test_dnames = {pname.split('/')[0] for pname, _ in data_names['test']}

# sdf cache
for i, (key, value) in enumerate(txn.cursor().iternext(), 1):
    data = pickle.loads(value)
    dname = data['dname']
    if data['dname'] in test_dnames:
        test_idxs.append(int(key.decode('ascii')))
    else:
        mol = data['lig_mol']
        if mol is not None:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=data['lig_mol'], includeChirality=True)
            txn_w.put(key, scaf.encode('ascii'))
            if i % 10000 == 0:
                logger.info(f"{i} finished.")
                logger.info(f"{len(test_idxs)=}")

txn_w.commit()
env_w.close()
with open(f"{out_dir}/test_idxs.pkl", 'wb') as f:
    pickle.dump(test_idxs, f)
