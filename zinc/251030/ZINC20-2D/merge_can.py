import sys, os
import concurrent.futures as cf
from argparse import ArgumentParser
import rdkit
from rdkit import Chem
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/cplm", f"{WORKDIR}/github/moses"]
from src.utils.lmdb import load_lmdb, new_lmdb
from src.utils.logger import get_logger, add_file_handler, log_git_hash

parser = ArgumentParser()
parser.add_argument('--size', type=int, required=True)
args = parser.parse_args()
logger = get_logger(stream=True)
add_file_handler(logger, "raw/canonicalize.log")
log_git_hash(logger)

env_w, txn_w = new_lmdb("can.lmdb", map_size=int(100e10))
for rank in range(args.size):
    env, txn = load_lmdb(f"raw/canonicalize/{rank}.lmdb", readahead=True)
    for key, value in txn.cursor().iternext():
        txn_w.put(key, value)
    logger.info(f"{rank} loaded.")
txn_w.commit()
env_w.close()

