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

    return tokens[sidx:eidx], (sent_idx1-sidx, sent_idx2-sidx2)


def parse_sent_window(ctx1, ctx2, full_matches, match_idx):
    _, tokens1, tok_to_sent_idx1 = ctx1
    _, tokens2, tok_to_sent_idx2 = ctx2

    p1, p2 = 0, 0
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
            diffs.append((window, sent_window))
        prev = p
    return diffs

def parse_hathi_diffs(hid1, hid2):
    hathi_path = Path('/home/allekim/stonybook-data/hathi/lcs_results/double_books') / "{}-{}.txt".format(hid1, hid2)
    with open(hathi_path, 'r') as f:
        full_matches = [[int(i) for i in x.strip().split()] for x in f.readlines()[1:]]
    ctx1 = parse_hathi_content(hid1)
    print(ctx1[2])
    ctx2 = parse_hathi_content(hid2)
    return parse_match_diffs(ctx1, ctx2, full_matches)

def count_diff(seq1, seq2):
    return sum(a != b for a, b in zip(seq1, seq2))+ abs(len(seq1) - len(seq2))


def is_punctuation(ctx1, ctx2):
    ctxx1 = ''.join([x for x in re.split(r'[^0-9a-zA-Z]+', ctx1) if x!='']).lower()
    ctxx2 = ''.join([x for x in re.split(r'[^0-9a-zA-Z]+', ctx2) if x!='']).lower()
    return (ctxx1 == ctxx2, len(ctx1)-len(ctxx1), len(ctx2)-len(ctxx2))

def strip_common_ffixes(ctx1, ctx2):
    i = 0
    while i < len(ctx1) and i < len(ctx2) and ctx1[i] == ctx2[i]:
        i += 1
    ctx1 = ctx1[i:]
    ctx2 = ctx2[i:]
    i = -1
    while -i <= len(ctx1) and -i <= len(ctx2) and ctx1[i] == ctx2[i]:
        i -= 1
    if i != -1:
        ctx1 = ctx1[:i+1]
        ctx2 = ctx2[:i+1]
    return (ctx1, ctx2)

def is_title(s):
    if "CHAPTER" in s:
        return True
    toks = s.split()
    if len(toks) < 2:
        return False
    return s.isupper()

def is_title_diff(ctx1, ctx2):
    ctx1, ctx2 = [x.strip() for x in strip_common_ffixes(ctx1, ctx2)]
    if ctx1 == '' and is_title(ctx2):
        return (True, 0, 1)
    elif ctx2 == '' and is_title(ctx1):
        return (True, 1, 0)
    else:
        return (False, 0, 0)

def is_diff_ctx(ctx1, ctx2):
    ctx1, ctx2 = [x.strip() for x in strip_common_ffixes(ctx1, ctx2)]
    ctx1 = ''.join([x for x in re.split(r'[^ 0-9a-zA-Z]+', ctx1) if x!=''])
    ctx2 = ''.join([x for x in re.split(r'[^ 0-9a-zA-Z]+', ctx2) if x!=''])
    ctx1_toks = ctx1.split()
    ctx2_toks = ctx2.split()
    if len(ctx1_toks) - len(ctx2_toks) >= 5:
        return (True, 1, 0)
    elif len(ctx2_toks) - len(ctx1_toks) >= 5:
        return (True, 0, 1)
    else:
        return (False, 0, 0)

def write_ocr(base_path):
    ocr_path = Path('/home/allekim/stonybook-data/hathi/ocr_results/double_books')
    ocr_path.mkdir(parents=True, exist_ok=True)
    with open(ocr_path / '{}.csv'.format(base_path.name), 'w') as writefile:
        headers = [
            'hid1', 'hid2', 'ctx1', 'ctx2', 'diff1', 'diff2'
            ]
        writer = csv.DictWriter(writefile, headers)
        writer.writeheader()
        hid1, hid2 = str(base_path.stem).split('-')
        diffs = parse_hathi_diffs(hid1, hid2)
        """
        for other_path in base_path.glob('*'):
            diffs = parse_hathi_diffs(base_path.name, other_path.stem)
            for windows, sent_windows in diffs:
                ctx1, ctx2 = windows
                fctx1, fctx2 = sent_windows
                base_hathi_context = unidecode(ctx1)
                hathi_context = unidecode(ctx2)
                row = {
                    'base_hathi_id': base_path.name, 
                    'hathi_id': other_path.stem, 
                    'base_hathi_context': base_hathi_context,
                    'hathi_context': hathi_context, 
                }
                if base_hathi_context == hathi_context:
                    row['err_type'] = 'UNICODE'
                    row['base_hathi_score'] = count_diff(base_hathi_context, row['base_hathi_context'])
                    row['hathi_score'] = count_diff(hathi_context, row['hathi_context'])
                    writer.writerow(row)
                    continue
                result, num1, num2 = is_punctuation(base_hathi_context, hathi_context)
                if result:
                    row['err_type'] = 'PUNCTUATION'
                    row['base_hathi_score'] = num1
                    row['hathi_score'] = num2
                    writer.writerow(row)
                    continue
                result, num1, num2 = is_title_diff(base_hathi_context, hathi_context)
                if result:
                    row['err_type'] = 'TITLE'
                    row['base_hathi_score'] = num1
                    row['hathi_score'] = num2
                    writer.writerow(row)
                    continue
                result, num1, num2 = is_diff_ctx(base_hathi_context, hathi_context)
                if result:
                    row['err_type'] = 'MISALIGN'
                    row['base_hathi_score'] = num1
                    row['hathi_score'] = num2
                    writer.writerow(row)
                    continue
                row['err_type'] = 'MISSPELL'
                row['base_hathi_score'] = -1
                row['hathi_score'] = -1
                writer.writerow(row)
        """

if __name__=="__main__":
    lcs_path = Path('/home/allekim/stonybook-data/hathi/lcs_results/double_books')
    lcs_paths = list(lcs_path.glob('*'))
    for p in lcs_paths:
        print(p)
        write_ocr(p)
        break
    """
    with Pool(50) as p:
        list(tqdm(p.imap_unordered(write_ocr, lcs_paths), total=len(lcs_paths)))
    """
