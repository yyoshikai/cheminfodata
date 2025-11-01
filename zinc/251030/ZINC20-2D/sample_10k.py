import sys, os
import pickle
from argparse import ArgumentParser
import numpy as np

WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.lmdb import load_lmdb, new_lmdb
from src.data.lmdb import data_len_to_blen

parser = ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

for seed, split in enumerate(['test', 'test_scaffolds'], 1):
    with open(f"raw/split/{split}_idxs.pkl", 'rb') as f:
        idxs = pickle.load(f)
    idxs = np.array(sorted(idxs))

    rng = np.random.default_rng(seed)
    idxs_random10k = rng.choice(idxs, size=1000 if args.test else 10000, replace=False)
    idxs_random10k.sort()
    idxs_random10k = idxs_random10k.tolist()

    env, txn = new_lmdb(f"idxs/{split}_random10k.lmdb")
    blen = data_len_to_blen(idxs_random10k)
    idx_blen = data_len_to_blen(1_940_000_000)
    for i, idx in enumerate(idxs_random10k):
        key = i.to_bytes(blen)
        value = idx.to_bytes(idx_blen)
        txn.put(key, value)
    txn.commit()
    env.close()
