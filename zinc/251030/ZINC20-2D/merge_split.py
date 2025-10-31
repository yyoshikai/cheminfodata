import sys, os
import pickle
import itertools as itr
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
import rdkit
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.lmdb import new_lmdb
from src.data.lmdb import data_len_to_blen
from src.utils.logger import get_logger, add_file_handler

parser = ArgumentParser()
parser.add_argument('--size', type=int, required=True)
args = parser.parse_args()

logger = get_logger(stream=True)
add_file_handler(logger, f"raw/merge_split.log")
logger.info(f"{rdkit.__version__=}")
logger.info(f"{args=}")

scaf2idxs = defaultdict(list)
for rank in range(args.size):
    with open(f"raw/filter_get_scaf/{rank}.pkl", 'rb') as f:
        scaf2idxs0 = pickle.load(f)
    logger.info(f"Loaded {rank=}")
    for scaf, idxs in scaf2idxs0.items():
        scaf2idxs[scaf] += idxs    
    logger.info(f"Merged {rank=}")

# see moses/scripts/prepare_dataset.py
logger.info("Sorting...")
scaf_idxs = list(scaf2idxs.items())
scaf_idxs = sorted(scaf_idxs, key=lambda scaf_idx: (-len(scaf_idx[1]), scaf_idx[0])) 

test_scaf_idxs = []
train_test_idxs = []
for i, (scaf, idx) in enumerate(scaf_idxs):
    if i % 10 == 9:
        test_scaf_idxs += idx
    else:
        train_test_idxs += idx
rng = np.random.default_rng(seed=0)
logger.info("Shuffling...")
rng.shuffle(train_test_idxs)
logger.info(f"{len(train_test_idxs)=}, {len(set(train_test_idxs))=}")
test_size = round(len(train_test_idxs)*0.1)
test_idxs, train_idxs = train_test_idxs[:test_size], train_test_idxs[test_size:]
os.makedirs("idxs", exist_ok=True)
for split, idxs in zip(['train', 'test', 'test_scaffolds'], [train_idxs, test_idxs, test_scaf_idxs]):
    logger.info(f"Saving {split}...")
    idxs = sorted(idxs)
    env, txn = new_lmdb(f"idxs/{split}.lmdb")
    blen = data_len_to_blen(idxs)
    idx_blen = data_len_to_blen(1_940_000_000)
    for i, idx in enumerate(idxs):
        key = i.to_bytes(blen)
        value = idx.to_bytes(idx_blen)
        txn.put(key, value)
    txn.commit()
    env.close()
logger.info("Finished!")
