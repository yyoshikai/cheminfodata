import sys, os
from argparse import ArgumentParser
import rdkit
from rdkit import Chem
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/cplm", f"{WORKDIR}/github/moses"]
from src.utils.lmdb import load_lmdb, new_lmdb
from src.data.lmdb import data_len_to_blen
from src.utils.logger import get_logger, add_file_handler, log_git_hash
from moses.metrics.utils import get_mol

parser = ArgumentParser()
parser.add_argument('--rank', type=int, required=True)
parser.add_argument('--size', type=int, required=True)
args = parser.parse_args()

logger = get_logger(stream=True)
os.makedirs("raw/canonicalize", exist_ok=True)
add_file_handler(logger, f"raw/canonicalize/{args.rank}.log")
logger.info(f"{rdkit.__version__=}")
log_git_hash(logger)

env, txn = load_lmdb("smi.lmdb", readahead=True)

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

env_w, txn_w = new_lmdb(f"raw/canonicalize/{args.rank}.lmdb")

for i, (key, value) in enumerate(cursor.iternext(), 1):
    idx = int.from_bytes(key)
    smi = value.decode('ascii')
    mol = get_mol(smi)
    can = Chem.MolToSmiles(mol) if mol is not None else ''
    value = can.encode('ascii')
    txn_w.put(key, value)
    if i % 10000 == 0:
        logger.info(f"{i} finished.")
txn_w.commit()
env_w.close()
