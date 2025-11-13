import sys, os, math
import pickle
from argparse import ArgumentParser
import rdkit
WORKDIR = os.environ.get('WORKDIR', '/workspace')
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.lmdb import load_lmdb, new_lmdb
from src.utils.logger import get_logger, add_file_handler

parser = ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--rank', type=int)
parser.add_argument('--size', type=int)
args = parser.parse_args()
raw_path ="./raw_test.lmdb" if args.test else \
    f"{WORKDIR}/cplm/preprocess/results/finetune/r4_all/main.lmdb"
env, txn = load_lmdb(raw_path, readahead=True)

out_dir = f".tmp/{args.rank}"
os.makedirs(out_dir, exist_ok=True)
env_w, txn_w = new_lmdb(f"{out_dir}/remove_mol.lmdb")

logger = get_logger(stream=True)
add_file_handler(logger, f"{out_dir}/remove_mol.log")
logger.info(f"{rdkit.__version__=}")

# workerの開始位置を推定
data_size = env.stat()['entries']
keys = sorted(map(str, range(data_size)))
i_start = int((data_size*args.rank)/args.size)
i_sup = int((data_size*(args.rank+1))/args.size)
logger.info(f"{data_size=}, {i_start=}, {i_sup=}")
key_start = keys[i_start].encode('ascii')
key_end = keys[i_sup-1].encode('ascii')
logger.info(f"{key_start=}, {key_end=}")

cursor = txn.cursor()
cursor.set_key(key_start)
for i, (key, value) in enumerate(cursor.iternext()):
    data = pickle.loads(value)
    del data['lig_mol']
    txn_w.put(key, pickle.dumps(data))
    if (i+1) % 10000 == 0:
        logger.info(f"{i+1} finished.")
    if key == key_end: break
txn_w.commit()
env_w.close()
