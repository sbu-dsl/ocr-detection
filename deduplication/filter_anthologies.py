import pickle
import csv
from collections import Counter, defaultdict
from nltk import word_tokenize
from pathlib import Path
from tqdm import tqdm

def is_similar_len(l1, l2, ratio=0.1):
    return abs(l1-l2) < max(l1,l2)*ratio

def tokenize_title(title):
    return set([x for x in word_tokenize(title) if x.isalpha()])

def is_overlapping_titles(titles):
    overlap = tokenize_title(titles.pop())
    for title in titles:
        overlap &= tokenize_title(title)
    return len(overlap) > 0

def parse_hathi_meta(hathi_meta_path):
    htid_to_meta = {}
    with open(hathi_meta_path) as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            htid_to_meta[row['htid']] = row
    return htid_to_meta

hathi_meta_path = Path('normalized_hathi_metadata.csv')
book_path = Path('/home/allekim/stonybook-data/hathi/ocr_detection')
parsed_ids = [tuple(str(p).split('/')[-2:]) for p in book_path.glob('*/*')][:1000]

hid_groups = defaultdict(list)
for lab, i in tqdm(parsed_ids):
    overlap_path = book_path / lab / i / 'overlap_scores.txt'
    hid = "{}.{}".format(lab, i)
    with open(overlap_path, 'r') as f:
        for line in f:
            stats = line.strip().split()
            other_id, l1, l2, overlap = stats
            l1 = int(l1)
            l2 = int(l2)
            overlap = int(overlap)
            if l1 and l2 and l1 > l2 and not is_similar_len(l1,l2) and overlap / min(l1,l2) > 0.5:
                hid_groups[hid].append(other_id)


hid_to_meta = parse_hathi_meta(hathi_meta_path)
with open('anthologies.csv', 'w') as csvfile:
    headers = ['hathi_id', 'title', 'author', 'titles', 'hathi_id_set']
    writer = csv.DictWriter(csvfile, headers)
    writer.writeheader()
    for hid in tqdm(hid_groups):
        titles = [hid_to_meta[x]['title'] for x in hid_groups[hid]]
        if not is_overlapping_titles(titles):
            row = {
                'hathi_id': hid,
                'title': hid_to_meta[hid]['title'],
                'author': hid_to_meta[hid]['author'],
                'titles': titles,
                'hathi_id_set': [hid] + hid_groups[hid]
            }
            writer.writerow(row)
