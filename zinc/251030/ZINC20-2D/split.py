import sys, os
import pickle
from psutil import virtual_memory
import numpy as np
from argparse import ArgumentParser
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.logger import get_logger, add_file_handler

parser = ArgumentParser()
parser.add_argument('--size', type=int, required=True)
args = parser.parse_args()

logger = get_logger(stream=True)
add_file_handler(logger, f"raw/merge_split.log")
logger.info(f"{args=}")

def log_mem():
    mem = virtual_memory()
    logger.info(f"memory use={mem.used/2**30:.03f}GB/{mem.total/2**30:.03f}GB")

logger.info("Loading test_scaf_scafs")
with open("raw/test_scaf_scafs.pkl", 'rb') as f:
    test_scaf_scafs = set(pickle.load(f))
log_mem()

test_scaf_idxs = []
train_test_idxs = []
for rank in range(args.size):
    logger.info(f"Porcessing {rank=}")
    with open(f"raw/filter_get_scaf/{rank}.pkl", 'rb') as f:
        scaf2idxs0 = pickle.load(f)
    for scaf, idxs in scaf2idxs0.items():
        if scaf in test_scaf_scafs:
            test_scaf_idxs += idxs
        else:
            train_test_idxs += idxs
log_mem()

test_scaf_idxs_set = set(test_scaf_idxs)
test_scaf_idxs = sorted(test_scaf_idxs_set)
os.makedirs("raw/split", exist_ok=True)
with open("raw/split/test_scaffolds_idxs.pkl", 'wb') as f:
    pickle.dump(test_scaf_idxs, f)
log_mem()

train_test_idxs = np.array(train_test_idxs, dtype=int)
rng = np.random.default_rng(0)
rng.shuffle(train_test_idxs)
test_size = round(len(train_test_idxs)*0.1)
train_idxs, test_idxs = train_test_idxs[test_size:], train_test_idxs[:test_size] 
log_mem()
del train_test_idxs
log_mem()
train_idxs.sort()
test_idxs.sort()

with open("raw/split/train_idxs.pkl", 'wb') as f:
    pickle.dump(train_idxs.tolist(), f)
with open("raw/split/test_idxs.pkl", 'wb') as f:
    pickle.dump(test_idxs.tolist(), f)

