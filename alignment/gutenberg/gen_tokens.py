import csv
import pandas as pd
import pickle
from ast import literal_eval
from lcs import lcs_with_anchors
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

def parse_guten_body(guten_id):
    guten_path = Path('/home/cpethe/stonybook-data/gutenberg/processed/{}/base_clean.xml'.format(guten_id))
    if not guten_path.exists():
        guten_path = Path('/home/cpethe/stonybook-data/gutenberg/processed/{}/base.xml'.format(guten_id))
    book = etree.parse(str(guten_path))
    body = book.find('body')
    text = body.text
    return text

def parse_hathi_tokens(hathi_id):
    hathi_path = Path('/home/allekim/stonybook-data/hathi/ocr_detection/{}/{}/parsed_content.pkl'.format(*hathi_id.split('.')))
    with open(hathi_path, 'rb') as f:
        _, tokens, _ = pickle.load(f)
    return tokens

def count_all_diffs(full_matches):
    count = 0
    prev = (0,0)
    for i, p in enumerate(full_matches):
        if p[0] - prev[0] > 1 or p[1] - prev[1] > 1:
            count += 1
        prev = p
    return count

def write_lcs_guten_results(guten_hathi_ids):
    guten_id, hathi_id_set = guten_hathi_ids
    hathi_id_set = hathi_id_set.split()

    guten_text = parse_guten_body(guten_id).split()
    base_guten_path = Path('/home/allekim/stonybook-data/lcs_results/gutenberg')
    guten_path = base_guten_path / guten_id
    guten_path.mkdir(parents=True, exist_ok=True)
    for hathi_id in hathi_id_set:
        hathi_text = parse_hathi_body(hathi_id).split()
        full_matches = lcs_with_anchors(guten_text, hathi_text)
        num_diffs = count_all_diffs(full_matches)
        hathi_path = guten_path / '{}.txt'.format(hathi_id)
        with open(hathi_path, 'w') as f:
            f.write("{} {} {} {}\n".format(len(guten_text), len(hathi_text), len(full_matches), num_diffs))
            for idx1, idx2 in full_matches:
                f.write("{} {}\n".format(idx1, idx2))
    print(guten_id)


if __name__ == "__main__":
    meta_df = pd.read_csv('/home/allekim/stonybook-data/final_combined_metadata.csv')
    filtered_df = df[df[['guten_id', 'hathi_id']].notnull().all(1)]
    final_ids = meta_df[meta_df['guten_id'] 

    """
    with Pool(50) as p:
        list(tqdm(p.imap_unordered(write_lcs_hathi_results, multi_hathi_books), total=len(multi_hathi_books)))
    """

