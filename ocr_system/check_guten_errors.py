import pandas as pd
from pathlib import Path
from scipy.special import softmax
from tqdm import tqdm

paths = Path('/home/allekim/stonybook-data/guten_hathi_alignment').glob('*/model_scored_toks.csv')

def score_losses(row):
    scores = softmax([row['loss1'], row['loss2']])
    row['score1'] = scores[1]
    row['score2'] = scores[0]
    return row

def filter_df(df):
    return df[df.apply(lambda row: (abs(len(row['ctx1'])-len(row['ctx2'])) < 1)
                & (row['loss1']> row['loss2']), axis=1)].apply(score_losses,axis=1)


all_df = []
for p in tqdm(paths):
    all_df.append(filter_df(pd.read_csv(p, converters={'ctx1':eval, 'ctx2':eval})))

df = pd.concat(all_df)
sorted_df = df.sort_values(by=['score1'])
sorted_df.to_csv('guten_errors_v3.csv', index=False)

