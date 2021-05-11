import csv
import pickle
from ast import literal_eval
import pandas as pd
from lcs import lcs_with_anchors
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

def parse_hathi_tokens(hathi_id):
    hathi_path = Path('/home/allekim/stonybook-data/hathi/processed/{}/{}/parsed_content.pkl'.format(*hathi_id.split('.')))
    if not hathi_path.exists():
        return None
    with open(hathi_path, 'rb') as f:
        _, tokens, _ = pickle.load(f)
    return tokens

def parse_guten_tokens(guten_id):
    guten_path = Path('/home/cpethe/stonybook-data/gutenberg/processed/{}/parsed_content.pkl'.format(guten_id))
    if not guten_path.exists():
        return None
    with open(guten_path, 'rb') as f:
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

def write_lcs_guten_results(guten_id):
    hathi_id = gid_to_hid[guten_id]
    base_guten_path = Path('/home/allekim/stonybook-data/guten_hathi_alignment')
    guten_path = base_guten_path / str(guten_id)
    guten_path.mkdir(parents=True, exist_ok=True)
    hathi_path = guten_path / '{}_new.txt'.format(hathi_id)
    if hathi_path.exists():
        return
    
    guten_text = parse_guten_tokens(guten_id)
    if guten_text == None:
        return
    hathi_text = parse_hathi_tokens(hathi_id)
    if hathi_text == None:
        return
    full_matches = lcs_with_anchors(guten_text, hathi_text)
    num_diffs = count_all_diffs(full_matches)
    with open(hathi_path, 'w') as f:
        f.write("{} {} {} {}\n".format(len(guten_text), len(hathi_text), len(full_matches), num_diffs))
        for idx1, idx2 in full_matches:
            f.write("{} {}\n".format(idx1, idx2))



if __name__ == "__main__":
    df = pd.read_csv('/home/allekim/stonybook-data/final_combined_metadata.csv')
    filtered_df = df[df[['guten_id', 'hathi_id']].notnull().all(1)]
    filtered_df = filtered_df.astype({'guten_id': int})
    final_ids = list(filtered_df[['guten_id','hathi_id']].to_records(index=False))
    gid_to_hid = {}
    for gid, hid in final_ids:
        gid_to_hid[gid] = hid
    gids = [x[0] for x in final_ids]
    with Pool(30) as p:
        list(tqdm(p.imap_unordered(write_lcs_guten_results, gids), total=len(gids)))

