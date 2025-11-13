import sys, os
import pickle
import random
from argparse import ArgumentParser
from collections import defaultdict
import rdkit
WORKDIR = os.environ.get('WORKDIR', '/workspace')
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.lmdb import load_lmdb, new_lmdb
from src.utils.logger import get_logger, add_file_handler
from src.data.lmdb import data_len_to_blen
assert rdkit.__version__.split('.')[0] == '2022'

parser = ArgumentParser()
parser.add_argument('--size', type=int, default=100)
args = parser.parse_args()
size = args.size

logger = get_logger(stream=True)
add_file_handler(logger, f".tmp/merge_split.log")
logger.info(f"{rdkit.__version__=}")
os.makedirs("mask", exist_ok=True)

# merge test idx
logger.info("merging test idx...")
test_idxs = []
for rank in range(size):
    with open(f".tmp/{rank}/test_idxs.pkl", 'rb') as f:
        test_idxs += pickle.load(f)
    logger.info(f"{rank} loaded.")
logger.info("Saving...")
test_idxs = sorted(test_idxs)
if len(test_idxs) > 0:
    blen = data_len_to_blen(len(test_idxs))
    idx_blen = data_len_to_blen(max(test_idxs)+1)
    env, txn = new_lmdb("mask/test_idxs.lmdb")
    for i, idx in enumerate(test_idxs):
        key = i.to_bytes(blen)
        value = idx.to_bytes(idx_blen)
        txn.put(key, value)
    txn.commit()
    env.close()

# split train/valid
logger.info("Loading train scaf...")
scaf2idxs = defaultdict(list)
for rank in range(size):
    env, txn = load_lmdb(f".tmp/{rank}/scaf.lmdb", readahead=True)
    for key, value in txn.cursor().iternext():
        idx = int(key.decode('ascii'))
        scaf = value.decode('ascii')
        scaf2idxs[scaf].append(idx)
    logger.info(f"{rank} loadded.")
    
logger.info("Splitting...")
scaf_idxs = list(scaf2idxs.items())
random.seed(0)
random.shuffle(scaf_idxs)
scaf_idxs = sorted(scaf_idxs, key=lambda x: len(x[1]))

train_idxs = []
valid_idxs = []
for scaf, idxs in scaf_idxs:
    if len(valid_idxs) < 300:
        valid_idxs += idxs
    else:
        train_idxs += idxs
logger.info(f"{len(scaf_idxs)=}")
train_idxs = sorted(train_idxs)
valid_idxs = sorted(valid_idxs)
max_idx = max(max(train_idxs), max(valid_idxs))
idx_blen = data_len_to_blen(max_idx+1)
logger.info(f"{len(train_idxs)=}, {len(valid_idxs)=}")

for split, idxs in zip(['valid', 'train'], [valid_idxs, train_idxs]):
    logger.info(f"Saving {split}...")
    env, txn = new_lmdb(f"mask/{split}_idxs.lmdb")
    blen = data_len_to_blen(len(idxs))
    for i, idx in enumerate(idxs):
        key = i.to_bytes(blen)
        value = idx.to_bytes(idx_blen)
        txn.put(key, value)
    txn.commit()
    env.close()


