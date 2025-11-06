import sys, os
import concurrent.futures as cf
from argparse import ArgumentParser
import rdkit
from rdkit import Chem
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/cplm", f"{WORKDIR}/github/moses"]
from src.utils.lmdb import load_lmdb, new_lmdb
from src.utils.logger import get_logger, add_file_handler, log_git_hash
from moses.metrics.utils import get_mol

parser = ArgumentParser()
parser.add_argument('--num-workers', type=int, default=28)
args = parser.parse_args()

logger = get_logger(stream=True)
add_file_handler(logger, f"raw/canonicalize.log")
logger.info(f"{rdkit.__version__=}")
log_git_hash(logger)

env, txn = load_lmdb("smi.lmdb", readahead=True)
def process(item: tuple[bytes, bytes]):
    key, value = item
    smi = value.decode('ascii')
    mol = get_mol(smi)
    return key, (Chem.MolToSmiles(mol) if mol is not None else '')

env_w, txn_w = new_lmdb("can.lmdb", map_size=int(100e10))
with cf.ProcessPoolExecutor(args.num_workers) as e:
    for i, (key, can) in enumerate(e.map(process, txn.cursor().iternext()), 1):
        value = can.encode('ascii')
        txn_w.put(key, value)
        if i % 1_000_000 == 0:
            logger.info(f"{i} finished.")
txn_w.commit()
env_w.close()
