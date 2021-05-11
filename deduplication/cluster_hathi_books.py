import pickle
import csv
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm

def is_similar_len(l1, l2, ratio=0.1):
    return abs(l1-l2) < max(l1,l2)*ratio

def connected_components(neighbors):
    seen = set()
    def component(node):
        nodes = set([node])
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes |= neighbors[node] - seen
            yield node
    for node in neighbors:
        if node not in seen:
            yield component(node)


anthologies = set(pd.read_csv('anthologies.csv')['hathi_id'])
book_path = Path('/home/allekim/stonybook-data/hathi/ocr_detection')
parsed_ids = [tuple(str(p).split('/')[-2:]) for p in book_path.glob('*/*')]

adj= defaultdict(set)
for lab, i in tqdm(parsed_ids):
    overlap_path = book_path / lab / i / 'overlap_scores.txt'
    hid = "{}.{}".format(lab, i)
    if hid in anthologies:
        continue
    with open(overlap_path, 'r') as f:
        for line in f:
            stats = line.strip().split()
            other_id, l1, l2, overlap = stats
            if other_id in anthologies:
                continue
            l1 = int(l1)
            l2 = int(l2)
            overlap = int(overlap)
            if l1 > 0 and l2 > 0 and overlap / min(l1,l2) > 0.25:
                adj[(l1, hid)].add((l2, other_id))
                adj[(l2, other_id)].add((l1, hid))


with open('dedup_hathi.csv', 'w') as csvfile:
    headers = ['hathi_id', 'hathi_id_set']
    writer = csv.DictWriter(csvfile, headers)
    writer.writeheader()
    for component in connected_components(adj):
        c = sorted(component, reverse=True)
        row = {
            'hathi_id': c[0][1],
            'hathi_id_set': [x[1] for x in c]
        }
        writer.writerow(row)

