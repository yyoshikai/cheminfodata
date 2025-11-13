import sys, os, math
import pickle
from argparse import ArgumentParser
import rdkit
WORKDIR = os.environ.get('WORKDIR', '/workspace')
sys.path.append(f"{WORKDIR}/cplm")
from src.data.lmdb import data_len_to_blen
from src.utils.lmdb import load_lmdb, new_lmdb
from src.utils.logger import get_logger, add_file_handler

parser = ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--size', type=int)
args = parser.parse_args()

logger = get_logger(stream=True)

raw_path ="./raw_test.lmdb" if args.test else \
    f"{WORKDIR}/cplm/preprocess/results/finetune/r4_all/main.lmdb"
env_raw, txn_raw = load_lmdb(raw_path, readahead=True)

blen = data_len_to_blen(env_raw.stat()['entries'])
add_file_handler(logger, ".tmp/merge.log")

env_w, txn_w = new_lmdb(f"./main.lmdb", map_size=int(100e10))

for rank in range(args.size):
    env, txn = load_lmdb(f".tmp/{rank}/add_mol.lmdb", readahead=True)
    for key, value in txn.cursor().iternext():
        idx = int(key.decode('ascii'))
        key = idx.to_bytes(blen)
        txn_w.put(key, value)
    logger.info(f"{rank=} size={env.stat()['entries']}")
txn_w.commit()
env_w.close()
