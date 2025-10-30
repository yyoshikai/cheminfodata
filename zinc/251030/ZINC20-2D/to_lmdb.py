import sys, os
from glob import glob
from argparse import ArgumentParser
import rdkit
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.lmdb import new_lmdb
from src.data.lmdb import data_len_to_blen
from src.utils.logger import get_logger, add_file_handler

parser = ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

logger = get_logger(stream=True)
add_file_handler(logger, "raw/to_lmdb.log")
logger.info(f"{rdkit.__version__=}")
logger.info(f"{args.test=}")

env_smi, txn_smi = new_lmdb("smi.lmdb", map_size=int(100e10))
env_id, txn_id = new_lmdb("id.lmdb", map_size=int(100e10))
size = 100000 if args.test else 1_940_000_000 # from website
blen = data_len_to_blen(size)
idx = 0

for path in sorted(glob("raw/ZINC20-2D/*/*.smi")):
    with open(path) as f:
        f.readline()
        if args.test:
            for line in f:
                smi, id = line[:-1].split(' ')
                key = idx.to_bytes(blen)
                txn_smi.put(key, smi.encode('ascii'))
                txn_id.put(key, id.encode('ascii'))
                idx += 1
                if idx == 100000: break
        else:
            for line in f:
                smi, id = line[:-1].split(' ')
                key = idx.to_bytes(blen)
                txn_smi.put(key, smi.encode('ascii'))
                txn_id.put(key, id.encode('ascii'))
                idx += 1  
    logger.info(f"Finished {path}, {idx=:>10}")
    assert idx <= size
    if idx == size: break
txn_smi.commit()
env_smi.close()
txn_id.commit()
env_id.close()