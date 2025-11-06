import sys, os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
import rdkit
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.logger import get_logger, add_file_handler, log_git_hash

parser = ArgumentParser()
parser.add_argument('--size', type=int, required=True)
args = parser.parse_args()

logger = get_logger(stream=True)
add_file_handler(logger, f"raw/split_scaf.log")
logger.info(f"{rdkit.__version__=}")
logger.info(f"{args=}")
log_git_hash(logger)

scaf2sizes = defaultdict(int)
for rank in range(args.size):
    with open(f"raw/filter_get_scaf/{rank}.pkl", 'rb') as f:
        scaf2idxs0 = pickle.load(f)
    logger.info(f"Loaded {rank=}")
    for scaf, idxs in scaf2idxs0.items():
        scaf2sizes[scaf] += len(idxs)    
    logger.info(f"Merged {rank=}")

# see moses/scripts/prepare_dataset.py
logger.info("Sorting...")
scaf_sizes = list(scaf2sizes.items())
scaf_sizes = sorted(scaf_sizes, key=lambda scaf_size: (-scaf_size[1], scaf_size[0])) 

test_scaf_scafs = {scaf_size[0] for scaf_size in scaf_sizes[9::10]}
with open("raw/test_scaf_scafs.pkl", 'wb') as f:
    pickle.dump(test_scaf_scafs, f)
