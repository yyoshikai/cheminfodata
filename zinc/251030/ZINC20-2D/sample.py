import sys, os
import pickle
from argparse import ArgumentParser
import numpy as np

WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.lmdb import new_lmdb
from src.data.lmdb import data_len_to_blen
from src.utils.logger import get_logger, add_file_handler, log_git_hash

parser = ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--n', type=int, required=True)
args = parser.parse_args()
logger = get_logger(stream=True)
add_file_handler(logger, "raw/sample.log", mode='a')
log_git_hash(logger)

for seed, split in enumerate(['test', 'test_scaffolds'], 1):
    with open(f"raw/split/{split}_idxs.pkl", 'rb') as f:
        idxs = pickle.load(f)
    idxs = np.array(sorted(idxs))

    rng = np.random.default_rng(seed)
    idxs_sample = rng.choice(idxs, args.n, replace=False)
    idxs_sample.sort()
    idxs_sample = idxs_sample.tolist()

    env, txn = new_lmdb(f"idxs/{split}_{args.name}.lmdb")
    blen = data_len_to_blen(idxs_sample)
    idx_blen = data_len_to_blen(1_940_000_000)
    for i, idx in enumerate(idxs_sample):
        key = i.to_bytes(blen)
        value = idx.to_bytes(idx_blen)
        txn.put(key, value)
    txn.commit()
    env.close()
