from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from scipy.special import softmax
import numpy as np
import pandas as pd

def score_losses(row):
    scores = softmax([row['loss1'], row['loss2']])
    row['score1'] = scores[1]
    row['score2'] = scores[0]
    return row

def write_winner(p):
    df = pd.read_csv(p).apply(score_losses,axis=1)
    gid_prob = len(df[df['score1'] > df['score2']]) / len(df)
    priors = [np.log(gid_prob), np.log(1 - gid_prob)]
    gid_likelihood = np.sum(np.log(df['score1'])) + priors[0]
    hid_likelihood = np.sum(np.log(df['score2'])) + priors[1]
    if gid_likelihood >= hid_likelihood:
        with open(p.parent / 'winner.txt', 'w') as f:
            f.write('gutenberg\n{} {}\n'.format(gid_likelihood, hid_likelihood))
    else:
        with open(p.parent / 'winner.txt', 'w') as f:
            f.write('hathi\n{} {}\n'.format(gid_likelihood, hid_likelihood))

paths = list(Path('/home/allekim/stonybook-data/guten_hathi_alignment').glob('*/model_scored_toks.csv'))
with Pool(30) as pool:
    list(tqdm(pool.imap(write_winner, paths), total=len(paths)))

