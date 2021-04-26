import pickle
import csv
from collections import Counter, defaultdict
from nltk import word_tokenize
from pathlib import Path
from tqdm import tqdm
from names_dataset import NameDataset

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
    works = set()
    with open(hathi_meta_path) as csvf:
        reader = csv.DictReader(csvf)
        for row in reader:
            if is_works(row['title']):
                works.add(row['htid'])
            htid_to_meta[row['htid']] = row
    return htid_to_meta, works

def is_works(title):
    title = title.strip('[]').lower()
    work_exact = {
        'works',
        'novels',
        'the works',
        'selected works',
        'his works',
        'works, illus',
        'historical works',
        'choice works',
        'poetical works',
        'works, in an english translation',
        'novels and stories',
    }
    work_in = {
        'works of',
        'collected works',
        'complete works',
        'miscellaneous works',
        'comprising works by',
        'posthumous works',
        'and other works',
        'select works',
        'novels and tales of',
        'novels and stories of',
        'complete writings of'
    }
    work_starting = {
        'works. ',
        'the works and life',
        'the works : revised',
        'works]',
        'works, ed',
        'the works, ',
        'the novels of',
        'the writings of'
    }
    if title in work_exact:
        return True
    for term in work_in:
        if term in title:
            return True
    for term in work_starting:
        if title.startswith(term):
            return True
    words = [x.strip('[]') for x in title.split()]
    if 'works' in words:
        idx = words.index('works')
        if idx > 0:
            author = words[idx-1]
            p_idx = author.find("'")
            if p_idx != -1:
                author = author[:p_idx]
            first_score = m.search_first_name(author)
            last_score = m.search_last_name(author)
            if first_score > 0 or last_score > 0:
                return True
    return False


print("Loading names")
m = NameDataset()

print("Parsing meta")
hathi_meta_path = Path('normalized_hathi_metadata.csv')
hid_to_meta, works = parse_hathi_meta(hathi_meta_path)
print(len(works))

book_path = Path('/home/allekim/stonybook-data/hathi/ocr_detection')
parsed_ids = [tuple(str(p).split('/')[-2:]) for p in book_path.glob('*/*')]

print("Parsing books")
hid_groups = defaultdict(list)
for lab, i in tqdm(parsed_ids):
    overlap_path = book_path / lab / i / 'overlap_scores.txt'
    hid = "{}.{}".format(lab, i)
    if hid in works:
        hid_groups[hid] = []
        continue
    with open(overlap_path, 'r') as f:
        for line in f:
            stats = line.strip().split()
            other_id, l1, l2, overlap = stats
            if other_id in works:
                continue
            l1 = int(l1)
            l2 = int(l2)
            overlap = int(overlap)
            if l1 and l2 and l1 > l2 and not is_similar_len(l1,l2) and overlap / min(l1,l2) > 0.5:
                hid_groups[hid].append(other_id)

print("Writing to file")
with open('anthologies.csv', 'w') as csvfile:
    headers = ['hathi_id', 'author', 'hathi_id_set']
    writer = csv.DictWriter(csvfile, headers)
    writer.writeheader()
    for hid in tqdm(hid_groups):
        group_meta = [(x, hid_to_meta[x]['title']) for x in hid_groups[hid]]
        titles = [x[1] for x in group_meta]
        if len(titles) == 0 or not is_overlapping_titles(titles):
            row = {
                'hathi_id': hid,
                'author': hid_to_meta[hid]['author'],
                'hathi_id_set': [(hid, hid_to_meta[hid]['title'])] + group_meta
            }
            writer.writerow(row)

