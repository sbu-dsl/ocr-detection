import sys
import csv
import numpy as np
import torch
import pandas as pd
from scipy.special import softmax
from lxml import etree
from lcs import lcs_with_anchors
from compare_books import parse_hathi_body, count_all_diffs
from gen_ocr_comparison import *
from pathlib import Path
from ast import literal_eval
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from multiprocessing import Pool, current_process, Queue

total_gpus = 8
gpu_num = int(sys.argv[1])
if gpu_num < 0 or gpu_num >= total_gpus:
    raise ValueError("Invalid GPU number {}".format(gpu_num))

device = "cuda:{}".format(gpu_num)
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()

def write_lcs_hathi_results(output_path, htid1, htid2):
    orig_hathi_text = parse_hathi_body(htid1).split()
    hathi_text = parse_hathi_body(htid2).split()
    full_matches = lcs_with_anchors(orig_hathi_text, hathi_text)
    num_diffs = count_all_diffs(full_matches)
    with open(output_path, 'w') as f:
        f.write("{} {} {} {}\n".format(len(orig_hathi_text), len(hathi_text), len(full_matches), num_diffs))
        for idx1, idx2 in full_matches:
            f.write("{} {}\n".format(idx1, idx2))

def parse_hathi_diffs(lcs_path, base_hathi_id, hathi_id):
    with open(lcs_path, 'r') as f:
        full_matches = [[int(i) for i in x.strip().split()] for x in f.readlines()[1:]]
    base_hathi_text = parse_hathi_body(base_hathi_id)
    hathi_text = parse_hathi_body(hathi_id)
    return parse_match_diffs(base_hathi_text, hathi_text, full_matches)


def write_hathi_ocr(ocr_path, lcs_path, id1, id2):
    with open(ocr_path, 'w') as writefile:
        headers = [
            'base_hathi_id', 'hathi_id', 'base_hathi_context', 'hathi_context', 
            'base_hathi_full_context', 'hathi_full_context',
            'err_type', 'base_hathi_score', 'hathi_score'
            ]
        writer = csv.DictWriter(writefile, headers)
        writer.writeheader()
        diffs = parse_hathi_diffs(lcs_path, id1, id2)
        for windows, sent_windows in diffs:
            ctx1, ctx2 = windows
            fctx1, fctx2 = sent_windows
            base_hathi_context = unidecode(ctx1)
            hathi_context = unidecode(ctx2)
            row = {
                'base_hathi_id': id1, 
                'hathi_id': id2, 
                'base_hathi_context': base_hathi_context,
                'hathi_context': hathi_context, 
                'base_hathi_full_context': unidecode(fctx1),
                'hathi_full_context': unidecode(fctx2)
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


def score_sent(sent):
    inputs = tokenizer(sent, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    labels = inputs["input_ids"].clone()
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=inputs["attention_mask"]
    )
    loss, logits = outputs['loss'], outputs['logits']
    return loss.item()

def clean_new_lines(s):
    return ' '.join(s.split())

def write_ocr_model_results(output_path, ocr_result_path, id1, id2):
    df = pd.read_csv(ocr_result_path)
    df = df[df['err_type'] == "MISSPELL"]
    df['base_hathi_full_context'] = df['base_hathi_full_context'].apply(clean_new_lines)
    df['hathi_full_context'] = df['hathi_full_context'].apply(clean_new_lines)
    c = zip(df['base_hathi_full_context'], df['hathi_full_context'])
    c = list(c)

    with open(output_path, 'w') as csvfile:
        headers = ['id1', 'id2', 'ctx1', 'ctx2', 'loss1', 'loss2']
        writer = csv.DictWriter(csvfile, headers)
        writer.writeheader()
        for i, pair in enumerate(c):
            if pair[0] == '' or pair[1] == '':
                continue
            if pair[0].isspace() or pair[1].isspace():
                continue
            row = {
                'id1': id1, 'id2': id2,
                'ctx1': pair[0], 'ctx2': pair[1]
            }
            try:
                row['loss1'] = score_sent(pair[0])
                row['loss2'] = score_sent(pair[1])
            except Exception as e:
                continue
            writer.writerow(row) 

def gen_winner_probs(row):
    loss1 = row['loss1']
    loss2 = row['loss2']
    return 1 - softmax([loss1,loss2])

def determine_winner(csvpath):
    df = pd.read_csv(csvpath)
    vals = dict((df['loss1'] < df['loss2']).value_counts())
    num_win1, num_win2 = vals[True], vals[False]
    priors = [num_win1 / (num_win1+num_win2), num_win2 / (num_win1+num_win2)]
    priors = np.log(priors)
    probs = df.apply(gen_winner_probs,axis=1)
    log_probs = probs.apply(lambda x: np.log(x))
    final_logs = priors + sum(log_probs)
    return softmax(final_logs)

def write_tournament_results(all_ids):
    print(ids)
    base_id = all_ids[0]
    output_path = Path('/home/allekim/stonybook-data/ocr_tournament_results/') / base_id
    lcs_output_path = Path('/home/allekim/stonybook-data/lcs_results/tournament/') / base_id
    ocr_output_path = Path('/home/allekim/stonybook-data/ocr_results/tournament/') / base_id
    output_path.mkdir(parents=True, exist_ok=True)
    lcs_output_path.mkdir(parents=True, exist_ok=True)
    ocr_output_path.mkdir(parents=True, exist_ok=True)
    winners = set(all_ids)
    round_no = 1
    record_path = output_path / "tournament_record.txt"
    if record_path.exists():
        return
    with open(record_path, "w") as f:
        while len(winners) > 1:
            f.write("ROUND {} START\n".format(round_no))
            candidates = winners.copy()
            winners = set()
            while len(candidates) > 1:
                cand1 = candidates.pop()
                cand2 = candidates.pop()
                lcs_path = lcs_output_path / "{}_{}_{}.txt".format(round_no, cand1, cand2)
                ocr_path = ocr_output_path / "{}_{}_{}.csv".format(round_no, cand1, cand2)
                path = output_path / "{}_{}_{}.csv".format(round_no, cand1, cand2)
                write_lcs_hathi_results(lcs_path, cand1, cand2)
                write_hathi_ocr(ocr_path, lcs_path, cand1, cand2)
                write_ocr_model_results(path, ocr_path, cand1, cand2)
                winner_prob = determine_winner(path)
                winner_num = winner_prob.argmax()
                if winner_num == 0:
                    winners.add(cand1)
                    f.write("{} beat {}, {} to {}\n".format(cand1, cand2, winner_prob[0], winner_prob[1]))
                elif winner_num == 1:
                    winners.add(cand2)
                    f.write("{} beat {}, {} to {}\n".format(cand2, cand1, winner_prob[1], winner_prob[0]))
            if len(candidates) == 1:
                cand = candidates.pop()
                winners.add(cand)
            f.write("ROUND {} END\n\n".format(round_no))
            round_no += 1
        winner = winners.pop()
        f.write("{}\n".format(winner))

multi_hathi_books = []
with open('hathi_ocr_clusters.txt', 'r') as f:
    for line in f:
        ids = literal_eval(line)
        if len(ids) > 2:
            multi_hathi_books.append(ids)
print(len(multi_hathi_books))

remaining_books = []
tournament_path = Path("/home/allekim/stonybook-data/ocr_tournament_results")
for all_ids in multi_hathi_books:
    base_id = all_ids[0]
    loc_path = tournament_path / base_id
    if loc_path.exists():
        continue
    remaining_books.append(all_ids)
print(len(remaining_books))

num_per_gpu = len(multi_hathi_books) // total_gpus + 1
sidx = gpu_num * num_per_gpu
local_multi_hathi_books = remaining_books[sidx : sidx + num_per_gpu]
for ids in local_multi_hathi_books:
    write_tournament_results(ids)
