import sys, os, pickle
from argparse import ArgumentParser
from collections import Counter
import psutil
import numpy as np
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/cplm", f"{WORKDIR}/github/moses"]
from src.utils.logger import get_logger, add_file_handler, log_git_hash

parser = ArgumentParser()
parser.add_argument('--size', type=int, required=True)
parser.add_argument('--sample-size', type=int, required=True)
args = parser.parse_args()
os.makedirs("stat", exist_ok=True)

logger = get_logger(stream=True)
add_file_handler(logger, "raw/merge_stat.log")
log_git_hash(logger)

def get_mem():
    mem = psutil.virtual_memory()
    return f"{mem.used/2**30:.03f}GB/{mem.total/2**30:.03f}"

for split in ['test', 'test_scaffolds']:
    stat = {}

    # FCD
    logger.info(f'[FCD] started. {get_mem()}')
    fcd_predictions = []
    for i, rank in enumerate(range(args.sample_size), 1):
        fcd_predictions.append(np.load(f"raw/compute_stat_sample/{split}/{rank}/fcd_predictions.npy"))
        if i % 50 == 0:
            logger.info(f'[FCD] loaded {i} {get_mem()}')
    fcd_predictions = np.concatenate(fcd_predictions, axis=0)
    stat['FCD'] = {
        'mu': fcd_predictions.mean(0), 
        'sigma': np.cov(fcd_predictions.T)
    }

    # SNN
    os.makedirs(f"stat/{split}_fps", exist_ok=True)
    logger.info(f'[SNN] started. {get_mem()}')
    for i, rank in enumerate(range(args.size), 1):
        fps = np.load(f"raw/compute_stat/{split}/{rank}/fps.npy")
        fps = np.packbits(fps, axis=1)
        np.save(f"stat/{split}_fps/{rank}.npy", fps)
        if i % 50 == 0:
            logger.info(f'[SNN] loaded {i} {get_mem()}')
    fps = np.concatenate(fps, axis=0)
    stat['SNN'] = {'fps': fps}

    # Frag
    logger.info(f'[Frag] started. {get_mem()}')
    counter = Counter()
    for i, rank in enumerate(range(args.size), 1):
        with open(f"raw/compute_stat/{split}/{rank}/frag_counter.pkl", 'rb') as f:
            counter0 = pickle.load(f)
        counter.update(counter0)
        if i % 50 == 0:
            logger.info(f'[Frag] loaded {i} {get_mem()}')
    stat['Frag'] = {'frag': counter}

    # Scaf
    logger.info(f"[Scaf] started.")
    counter = Counter()
    for i, rank in enumerate(range(args.size), 1):
        with open(f"raw/compute_stat/{split}/{rank}/scaf_counter.pkl", 'rb') as f:
            counter0 = pickle.load(f)
        counter.update(counter0)
        if i % 50 == 0:
            logger.info(f'[Scaf] loaded {i} {get_mem()}')
    stat['Scaf'] = {'scaf': counter}

    # wassertain
    for name in ['logP', 'SA', 'QED', 'weight']:
        logger.info(f"[{name}] started. {get_mem()}")
        value = []
        for i, rank in enumerate(range(args.sample_size)):
            with open(f"raw/compute_stat_sample/{split}/{rank}/{name}.pkl", 'rb') as f:
                value += pickle.load(f)
            if i % 50 == 0:
                logger.info(f'[{name}] loaded {i} {get_mem()}')
        stat[name] = {'values': value}

    with open(f"stat/{split}.pkl", 'wb') as f:
        pickle.dump(stat, f)

