from pathlib import Path
import torch
import pickle
import numpy as np
from pathlib import Path
from scipy.special import softmax
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool

def map_subtok_to_tok_idx(offsets):
    idx_mapping = []
    tok_num = -1
    for idx, pair in enumerate(offsets):
        s, e = pair
        if e == 0:
            continue
        if s == 1:
            tok_num += 1
        idx_mapping.append(tok_num)
    return idx_mapping

def gen_overlap(indices):
    consecutive_indices = []
    sidx = indices[0]
    eidx = sidx + 1
    for idx in indices[1:]:
        if idx == eidx:
            eidx += 1
        else:
            consecutive_indices.append((sidx, eidx))
            sidx = idx
            eidx = sidx+1
    consecutive_indices.append((sidx, eidx))
    return consecutive_indices

def check_overlap(idx_ranges, possibilities):
    matches = []
    for s,e in idx_ranges:
        for r, gt, ocr in possibilities:
            x, y = r
            if s >= x and e <= y:
                matches.append(((s,e), (x,y), gt))
    return matches

def find_ocr_toks(path):
    hathi_id = path.stem.split('_')[0]
    result_path = path.parent / '{}_hathi_batched_results.dt'.format(hathi_id)
    if not result_path.exists():
        return
    input_path = result_path.parent / '{}_hathi_input.dt'.format(hathi_id)
    aligned_df = pd.read_csv(result_path.parent / 'aligned_toks.csv',
                             converters={'sent_idx2':eval,'ctx1':eval,'ctx2':eval,'diff1':eval,'diff2':eval})
    inputs = torch.load(input_path)
    results = torch.load(result_path)

    bad_sents = defaultdict(list)
    for i, p in enumerate(aligned_df['sent_idx2']):
        x, y = p
        if x!=y:
            continue
        row = aligned_df.loc[i]
        idx1, idx2 = row['diff1']
        diff1 = idx2 - idx1
        gt = row['ctx1'][idx1:idx2]

        idx1, idx2 = row['diff2']
        diff2 = idx2 - idx1
        ocr = row['ctx2'][idx1:idx2]
        if abs(diff1-diff2) > 5:
            continue
        bad_sents[x].append((row['diff2'], tuple(gt), tuple(ocr)))
    preda = softmax(results, axis=-1)
    thresholds = [0.8,0.9,0.95,0.99]
    ocr_correct = defaultdict(set)
    ocr_wrong = defaultdict(int)
    for idx in range(len(inputs)):
        mask = np.where(np.array(inputs[idx]['attention_mask']) == 1)
        input_ids = np.array(inputs[idx]['input_ids'])[mask][1:-1]
        offsets = np.array(inputs[idx]['offset_mapping'])[mask]
        subtok_to_tok = np.array(map_subtok_to_tok_idx(offsets))
        preds = np.array(preda[idx])[mask][1:-1]
        for thres in thresholds:
            pos_idx = np.where(preds[:,1] > thres)[0]
            pred = np.zeros(len(preds))
            pred[pos_idx] = 1
            if 1 in pred:
                pred_mask = np.where(pred==1)
                indices = subtok_to_tok[pred_mask]
                indices = sorted(set(indices))
                guesses = gen_overlap(indices)
                if idx not in bad_sents:
                    ocr_wrong[thres] += len(guesses)
                    continue
                candidates = bad_sents[idx]
                tok_overlaps = check_overlap(guesses, candidates)
                for x, y, gt in tok_overlaps:
                    guess = tuple(inputs[idx]['sent'][x[0]:x[1]])
                    orig = tuple(inputs[idx]['sent'][y[0]:y[1]])
                    ocr_correct[thres].add((idx, x, y, guess, orig, gt))
                ocr_wrong[thres] += len(guesses) - len(tok_overlaps)
    num_bad_ocr = sum([len(y) for x,y in bad_sents.items()])
    results = (num_bad_ocr, bad_sents, ocr_correct, ocr_wrong)
    with open(path.parent / 'ocr_detect_tokwise_results.pkl', 'wb') as f:
        pickle.dump(results, f, 4)

paths = list(Path('/home/allekim/stonybook-data/guten_hathi_alignment').glob('*/*_new.txt'))
real_paths = []
for p in paths:
    with open(p) as f:
        l1, l2, overlap, match = map(int, next(f).split())
        if min(l1,l2) / max(l1,l2) < 0.8:
            continue
        real_paths.append(p)
print(len(real_paths))

with Pool(30) as pool:
    list(tqdm(pool.imap(find_ocr_toks, real_paths), total=len(real_paths)))

