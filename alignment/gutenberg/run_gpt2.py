import sys
import csv
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from ast import literal_eval
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from multiprocessing import Pool, current_process, Queue

total_gpus = 6
gpu_num = int(sys.argv[1])
if gpu_num < 0 or gpu_num >= total_gpus:
    raise ValueError("Invalid GPU number {}".format(gpu_num))

device = "cuda:{}".format(gpu_num)
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()

def score_sent(sent):
    inputs = tokenizer(sent, is_split_into_words=True, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    labels = inputs["input_ids"].clone()
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=inputs["attention_mask"]
    )
    loss, logits = outputs['loss'], outputs['logits']
    return loss.item()

def write_ocr_model_results(ocr_result_path):
    output_path = ocr_result_path / 'model_scored_toks.csv'
    tok_csv_path = ocr_result_path / 'aligned_toks.csv'
    if not tok_csv_path.exists():
        return
    df = pd.read_csv(ocr_result_path / 'aligned_toks.csv', converters={'ctx1':literal_eval, 'ctx2':literal_eval})
    print(ocr_result_path.stem)
    with open(output_path, 'w') as csvfile:
        headers = ['gid', 'hid', 'ctx1', 'ctx2', 'diff1', 'diff2', 'loss1', 'loss2']
        writer = csv.DictWriter(csvfile, headers)
        writer.writeheader()
        for index, row in df.iterrows():
            row = dict(row)
            try:
                row['loss1'] = score_sent(row['ctx1'])
                row['loss2'] = score_sent(row['ctx2'])
            except:
                continue
            writer.writerow(row) 

cached_books = set([x.parent.name for x in Path('/home/allekim/stonybook-data/guten_hathi_alignment/').glob('*/model_scored_toks.csv')])
guten_hathi_books = list(Path('/home/allekim/stonybook-data/guten_hathi_alignment').glob('*'))
remaining_books = [x for x in guten_hathi_books if x.name not in cached_books]
print(len(remaining_books))
num_per_gpu = len(remaining_books) // total_gpus
sidx = gpu_num * num_per_gpu
local_multi_hathi_books = remaining_books[sidx : sidx + num_per_gpu]
for path in local_multi_hathi_books:
    write_ocr_model_results(path)
"""
guten_hathi_books = list(Path('/home/allekim/stonybook-data/guten_hathi_alignment').glob('*'))
num_per_gpu = len(guten_hathi_books) // total_gpus
sidx = gpu_num * num_per_gpu
local_books = guten_hathi_books[sidx : sidx + num_per_gpu]
for path in local_books:
    write_ocr_model_results(path)
"""
