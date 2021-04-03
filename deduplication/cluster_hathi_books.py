import pickle
import csv
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm

def is_similar_len(l1, l2, ratio=0.1):
    return abs(l1-l2) < max(l1,l2)*ratio

book_path = Path('/home/allekim/stonybook-data/hathi/ocr_detection')
parsed_ids = [tuple(str(p).split('/')[-2:]) for p in book_path.glob('*/*')]

hid_groups = defaultdict(list)
seen = set()
for lab, i in tqdm(parsed_ids):
    overlap_path = book_path / lab / i / 'overlap_scores.txt'
    hid = "{}.{}".format(lab, i)
    if hid in seen:
        continue
    seen.add(hid)
    with open(overlap_path, 'r') as f:
        for line in f:
            stats = line.strip().split()
            other_id, l1, l2, overlap = stats
            if other_id in seen:
                continue
            l1 = int(l1)
            l2 = int(l2)
            overlap = int(overlap)
            if is_similar_len(l1,l2) and overlap / min(l1,l2) > 0.5:
                hid_groups[hid].append(other_id)
                seen.add(other_id)


with open('dedup_hathi.csv', 'w') as csvfile:
    headers = ['hathi_id', 'hathi_id_set']
    writer = csv.DictWriter(csvfile, headers)
    writer.writeheader()
    for hid in hid_groups:
        row = {
            'hathi_id': hid,
            'hathi_id_set': [hid] + hid_groups[hid]
        }
        writer.writerow(row)
