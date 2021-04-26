import csv
import pandas as pd
from pathlib import Path
from ast import literal_eval
from tqdm import tqdm
from copy import deepcopy

hathi_path = Path('/home/allekim/stonybook-data/hathi/ocr_detection')
all_ids = ["{}.{}".format(*(str(p).split('/')[-2:])) for p in hathi_path.glob('*/*')]

hathi_meta_path = Path('normalized_hathi_metadata.csv')
dedup_hathi_meta_path = Path('dedup_hathi.csv')
final_hathi_meta = Path('dedup_hathi_metadata.csv')


duped_books = set()
clusters = {}
with open(dedup_hathi_meta_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        gid = row['hathi_id']
        gid_set = literal_eval(row['hathi_id_set'])
        for i in gid_set:
            duped_books.add(i)
        clusters[gid] = gid_set

anthologies = set(pd.read_csv('anthologies.csv')['hathi_id'])
seen = set()
with open(hathi_meta_path, 'r') as rf:
    reader = csv.DictReader(rf)
    with open(final_hathi_meta, 'w') as wf:
        headers = ['htid', 'title', 'version', 'pub_date', 'author']
        writer = csv.DictWriter(wf, headers)
        writer.writeheader()
        for row in tqdm(reader):
            gid = row['htid']
            if gid in anthologies:
                continue
            if gid in seen:
                continue
            seen.add(gid)
            if gid not in all_ids:
                continue
            if gid in duped_books:
                if gid in clusters:
                    cluster = clusters[gid]
                    for i in cluster:
                        seen.add(i)
                    writer.writerow(row)
                else:
                    continue
            else:
                writer.writerow(row)


