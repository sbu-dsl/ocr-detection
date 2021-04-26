import re
import pickle
import csv
from lxml import etree
from nltk import tokenize
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

def parse_hathi_content(hathi_id):
    hathi_path = Path('/home/allekim/stonybook-data/hathi/ocr_detection/{}/{}/parsed_content.pkl'.format(*hathi_id.split('.')))
    with open(hathi_path, 'rb') as f:
        sents, tokens, tok_to_sent_idx = pickle.load(f)
    return sents, tokens, tok_to_sent_idx

def parse_sentence_mask(tokens, tok_to_sent_idx, idx1, idx2):
    sent_idx1 = tok_to_sent_idx[idx1]
    sent_idx2 = tok_to_sent_idx[idx2]
    sidx = idx1
    while sidx >= 0 and tok_to_sent_idx[sidx] == sent_idx1:
        sidx -= 1
    sidx += 1

    eidx = idx2
    while eidx < len(tokens) and tok_to_sent_idx[eidx] == sent_idx2:
        eidx += 1

    return tokens[sidx:eidx], (idx1-sidx+1, idx2-sidx)

def parse_sent_window(ctx1, ctx2, full_matches, match_idx):
    _, tokens1, tok_to_sent_idx1 = ctx1
    _, tokens2, tok_to_sent_idx2 = ctx2

    p1, p2 = -1, -1
    if match_idx > 0:
        p1, p2 = full_matches[match_idx-1]
    q1, q2 = full_matches[match_idx]

    sents1 = parse_sentence_mask(tokens1, tok_to_sent_idx1, p1, q1)
    sents2 = parse_sentence_mask(tokens2, tok_to_sent_idx2, p2, q2)
    return (sents1, sents2)

def parse_match_diffs(ctx1, ctx2, full_matches):
    diffs = []
    prev = (0,0)
    for i, p in enumerate(full_matches):
        if p[0] - prev[0] > 1 or p[1] - prev[1] > 1:
            sent_window = parse_sent_window(ctx1, ctx2, full_matches, i)
            diffs.append(sent_window)
        prev = p
    return diffs

def parse_hathi_diffs(hid1, hid2):
    hathi_path = Path('/home/allekim/stonybook-data/hathi/lcs_results/double_books') / "{}-{}.txt".format(hid1, hid2)
    with open(hathi_path, 'r') as f:
        full_matches = [[int(i) for i in x.strip().split()] for x in f.readlines()[1:]]
    ctx1 = parse_hathi_content(hid1)
    ctx2 = parse_hathi_content(hid2)
    return parse_match_diffs(ctx1, ctx2, full_matches)

def filter_nonalphnum(toks):
    cleaned_toks = [re.sub(r'\W+', '', tok) for tok in toks]
    return ''.join(cleaned_toks)

def write_ocr(base_path):
    ocr_path = Path('/home/allekim/stonybook-data/hathi/ocr_results/double_books')
    ocr_path.mkdir(parents=True, exist_ok=True)
    with open(ocr_path / '{}.csv'.format(base_path.stem), 'w') as writefile:
        headers = [
            'hid1', 'hid2', 'ctx1', 'ctx2', 'diff1', 'diff2'
            ]
        writer = csv.DictWriter(writefile, headers)
        writer.writeheader()
        hid1, hid2 = str(base_path.stem).split('-')
        diffs = parse_hathi_diffs(hid1, hid2)
        for s1, s2 in diffs:
            tok1, idx1 = s1
            tok2, idx2 = s2
            diff1 = filter_nonalphnum(tok1[idx1[0]:idx1[1]])
            diff2 = filter_nonalphnum(tok2[idx2[0]:idx2[1]])
            if diff1 == diff2:
                continue
            writer.writerow({
                'hid1': hid1,
                'hid2': hid2,
                'ctx1': tok1,
                'ctx2': tok2,
                'diff1': idx1,
                'diff2': idx2
            })


if __name__=="__main__":
    lcs_path = Path('/home/allekim/stonybook-data/hathi/lcs_results/double_books')
    lcs_paths = list(lcs_path.glob('*'))
    with Pool(50) as p:
        list(tqdm(p.imap_unordered(write_ocr, lcs_paths), total=len(lcs_paths)))
