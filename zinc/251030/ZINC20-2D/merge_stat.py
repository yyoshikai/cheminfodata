import sys, os, pickle
from argparse import ArgumentParser
from collections import Counter
import numpy as np
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/cplm", f"{WORKDIR}/github/moses"]
from src.utils.logger import get_logger, add_file_handler, log_git_hash


parser = ArgumentParser()
parser.add_argument('--size', type=int, required=True)
args = parser.parse_args()
os.makedirs("stat", exist_ok=True)

logger = get_logger(stream=True)
add_file_handler(logger, "raw/merge_stat.log")
log_git_hash(logger)

for split in ['test', 'test_scaffolds']:
    stat = {}

    # FCD
    fcd_predictions = []
    for rank in range(args.size):
        fcd_predictions.append(np.load(f"raw/compute_stat_sample/{split}/{rank}/fcd_predictions.npy"))
    fcd_predictions = np.concatenate(fcd_predictions, axis=0)
    stat['FCD'] = {
        'mu': fcd_predictions.mean(0), 
        'sigma': np.cov(fcd_predictions.T)
    }

    # SNN
    fps = []
    for rank in range(args.size):
        fps.append(np.load(f"raw/compute_stat/{split}/{rank}/fps.npy"))
    fps = np.concatenate(fps, axis=0)
    stat['SNN'] = {'fps': fps}

    # Frag
    counter = Counter()
    for rank in range(args.size):
        with open(f"raw/compute_stat/{split}/{rank}/frag_counter.pkl", 'rb') as f:
            counter0 = pickle.load(f)
        counter.update(counter0)
    stat['Frag'] = {'frag': counter}

    # Scaf
    counter = Counter()
    for rank in range(args.size):
        with open(f"raw/compute_stat/{split}/{rank}/scaf_counter.pkl", 'rb') as f:
            counter0 = pickle.load(f)
        counter.update(counter0)
    stat['Scaf'] = {'scaf': counter}

    # wassertain
    for name in ['logP', 'SA', 'QED', 'weight']:
        value = []
        for rank in range(args.size):
            with open(f"raw/compute_stat_sample/{split}/{rank}/{name}.pkl", 'rb') as f:
                value += pickle.load(f)
        stat[name] = {'values': value}

    with open(f"stat/{split}.pkl", 'wb') as f:
        pickle.dump(stat, f)

