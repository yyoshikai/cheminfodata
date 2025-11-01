import sys, os
import pickle
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.logger import get_logger, add_file_handler
from src.utils.lmdb import new_lmdb
from src.data.lmdb import data_len_to_blen


logger = get_logger(stream=True)
add_file_handler(logger, f"raw/make_idx.log")

os.makedirs("idxs", exist_ok=True)
for split in ['train', 'test', 'test_scaffolds']:
    with open(f"raw/split/{split}_idxs.pkl", 'rb') as f:
        idxs = pickle.load(f)
    idxs = sorted(idxs)
    blen = data_len_to_blen(idxs)
    idx_blen = data_len_to_blen(1_940_000_000)
    env, txn = new_lmdb(f"idxs/{split}.lmdb")
    for i, idx in enumerate(idxs):
        key = i.to_bytes(blen)
        value = idx.to_bytes(idx_blen)
        txn.put(key, value)
    txn.commit()
    env.close()
    del idxs
