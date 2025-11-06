import sys, os, pickle
from argparse import ArgumentParser
from collections import Counter
import numpy as np
import scipy
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/cplm", f"{WORKDIR}/github/moses"]
from src.utils.lmdb import load_lmdb, new_lmdb
from src.data.lmdb import data_len_to_blen
from src.utils.logger import get_logger, add_file_handler, log_git_hash
from moses.utils import get_mol
from moses.metrics.utils import fragmenter, compute_scaffold, fingerprint

parser = ArgumentParser()
parser.add_argument('--size', type=int)
parser.add_argument('--rank', type=int)
args = parser.parse_args()

for split in ['test', 'test_scaffolds']:
    out_dir = f"raw/compute_stat/{split}/{args.rank}"
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(stream=True)
    add_file_handler(logger, f"{out_dir}/.log")
    log_git_hash(logger)

    env_idx, txn_idx = load_lmdb(f"idxs/{split}.lmdb", readahead=True)
    env_smi, txn_smi = load_lmdb(f"smi.lmdb", readahead=True)
    blen_idx = data_len_to_blen(env_smi.stat()['entries'])

    size = env_idx.stat()['entries']
    i_start = int((size*args.rank)/args.size)
    i_stop = int((size*(args.rank+1))/args.size)
    if args.rank == args.size-1: i_stop = size
    blen_i = data_len_to_blen(size)
    key_i_start = i_start.to_bytes(blen_i)
    cursor_idx = txn_idx.cursor()
    cursor_idx.set_key(key_i_start)
    logger.info(f"{args.rank=}, {args.size=} {i_start=}, {i_stop=}")

    frag_counter = Counter()
    scaf_counter = Counter()
    fps = []
    for di, value_idx in enumerate(cursor_idx.iternext(keys=False)):
        if di == i_stop-i_start: break
        value_idx = int.from_bytes(value_idx).to_bytes(blen_idx) # Temp
        value_smi = txn_smi.get(value_idx)
        smi = value_smi.decode('ascii')
        mol = get_mol(smi)
        if mol is None: continue
        frag_counter.update(fragmenter(mol))
        scaf_counter[compute_scaffold(mol, 2)] += 1
        fps.append(fingerprint(mol, 'morgan').astype(np.uint8))
        if (di+1) % 10000 == 0:
            logger.info(f"{di+1} finished.")
    # fps = scipy.sparse.vstack(fps).tocsr()
    fps = np.vstack(fps)

    with open(f"{out_dir}/frag_counter.pkl", 'wb') as f:
        pickle.dump(frag_counter, f)
    with open(f"{out_dir}/scaf_counter.pkl", 'wb') as f:
        pickle.dump(scaf_counter, f)
    np.save(f"{out_dir}/fps.npy", fps)
