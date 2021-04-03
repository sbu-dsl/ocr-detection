import csv
import pickle
from ast import literal_eval
from lcs import lcs_with_anchors
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

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

def write_lcs_hathi_results(hathi_id_set):
    orig_hathi_id = hathi_id_set[0]
    orig_hathi_text = parse_hathi_tokens(orig_hathi_id)
    base_hathi_path = Path('/home/allekim/stonybook-data/hathi/lcs_results/double_books')
    base_hathi_path.mkdir(parents=True, exist_ok=True)
    for hathi_id in hathi_id_set:
        if hathi_id == orig_hathi_id:
            continue
        hathi_text = parse_hathi_tokens(hathi_id)
        full_matches = lcs_with_anchors(orig_hathi_text, hathi_text)
        num_diffs = count_all_diffs(full_matches)
        hathi_path = base_hathi_path / '{}-{}.txt'.format(orig_hathi_id, hathi_id)
        with open(hathi_path, 'w') as f:
            f.write("{} {} {} {}\n".format(len(orig_hathi_text), len(hathi_text), len(full_matches), num_diffs))
            for idx1, idx2 in full_matches:
                f.write("{} {}\n".format(idx1, idx2))

if __name__ == "__main__":
    multi_hathi_books = set()
    dedup_hathi_path = Path('/home/allekim/ocr-detection/deduplication/dedup_hathi.csv')
    with open(dedup_hathi_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            books = tuple(literal_eval(row['hathi_id_set']))
            if len(books) == 2:
                multi_hathi_books.add(books)
    print(len(multi_hathi_books))

    with Pool(50) as p:
        list(tqdm(p.imap_unordered(write_lcs_hathi_results, multi_hathi_books), total=len(multi_hathi_books)))

