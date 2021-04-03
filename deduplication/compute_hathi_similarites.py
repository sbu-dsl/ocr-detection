import pickle
import pandas as pd
import numpy as np
from ast import literal_eval
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

def get_same_author_ids(i):
    book_ids = set()
    author = id_to_author[i]
    for other_id in author_books[author]:
        if i != other_id:
            book_ids.add(other_id)
    return book_ids

def compute_similarities(path):
    hlab, hid = str(path).split('/')[-2:]
    full_hid = "{}.{}".format(hlab, hid)
    ngram_path = 'fivegram.pkl'
    with open(path / 'overlap_scores.txt', 'w') as f:
        with open(path / ngram_path, 'rb') as pkl:
            fivegrams = pickle.load(pkl)
        author_books = get_same_author_ids(full_hid)
        for other_hid in author_books:
            other_lab, other_id = other_hid.split('.',1)
            other_path = book_path / other_lab / other_id
            with open(other_path / 'fivegram.pkl', 'rb') as opkl:
                other_fivegrams = pickle.load(opkl)
            fivegram_intersection = fivegrams & other_fivegrams
            f.write("{} {} {} {}\n".format(
                other_hid, 
                len(fivegrams), 
                len(other_fivegrams), 
                len(fivegram_intersection)
            ))

book_path = Path('/home/allekim/stonybook-data/hathi/ocr_detection')
book_paths = list(book_path.glob('*/*'))
hathi_df = pd.read_csv('/home/allekim/stonybook-data/normalized_hathi_metadata.csv')

ids = list(hathi_df['htid'])
authors = list(hathi_df['author'])
id_to_author = {}
author_books = defaultdict(list)
for idx in range(len(ids)):
    hid = ids[idx]
    author = authors[idx]
    id_to_author[hid] = author
    author_books[author].append(hid)

with Pool(48) as pool:
    list(tqdm(pool.imap(compute_similarities, book_paths)))

