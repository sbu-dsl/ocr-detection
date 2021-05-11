import os
import csv
import numpy as np
import pickle
import random
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from datasets import load_dataset, load_metric, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ast import literal_eval

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import set_seed, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0.dev0")

def generate_examples(row):
    hid1, hid2 = row['hid1'], row['hid2']
    loss1, loss2 = row['loss1'], row['loss2']
    diff1, diff2 = row['diff1'], row['diff2']
    ctx1, ctx2 = row['ctx1'], row['ctx2']
    ocr1, ocr2 = ctx1[diff1[0]:diff1[1]], ctx2[diff2[0]:diff2[1]]
    ex1 = ' '.join(ctx1[:diff1[0]]) + ' <ocr> ' + ' '.join(ocr1) + ' </ocr> ' + ' '.join(ctx1[diff1[1]:])
    ex2 = ' '.join(ctx2[:diff2[0]]) + ' <ocr> ' + ' '.join(ocr2) + ' </ocr> ' + ' '.join(ctx2[diff2[1]:])
    correct = "<blank>"
    if loss1 < loss2:
        if ocr1:
            correct = ' '.join(ocr1)
        return hid2, ex2, correct
    else:
        if ocr2:
            correct = ' '.join(ocr2)
        return hid1, ex1, correct

def preprocess_function(examples):
    inputs = examples['orig']
    targets = examples['corrected']
    inputs = [inp for inp in inputs]
    model_inputs = tokenizer(inputs, padding=True, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding=True, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


seed = 1729
set_seed(seed)
model_name = "ocr_correction_model"

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.add_tokens(["<ocr>", "</ocr>", "<blank>"], special_tokens=True)        
tokenizer.add_special_tokens({"additional_special_tokens": ["<ocr>", "</ocr>", "<blank>"]})

print("Loading test")
num_samples = 200000
testp = Path('/home/allekim/ocr-detection/ocr_data/test.csv')
df = pd.read_csv(testp, converters={'ctx1': eval, 'ctx2': eval, 'diff1': eval, 'diff2': eval}, nrows=num_samples)
df = df.sample(num_samples, random_state=seed)
df[['hid', 'orig','corrected']] = df.apply(generate_examples, axis=1, result_type="expand")
test_dataset = Dataset.from_pandas(df[['hid', 'orig', 'corrected']])

test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=None,
    num_proc=1,
)

print("Loading model")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
device = "cuda:4"
model.to(device)

print("Evaluating")
results = []
i = 0
batch_size = 64
for i in tqdm(range(0,len(test_dataset), batch_size)):
    x = test_dataset[i:i+batch_size]
    input_ids = torch.tensor(x['input_ids']).to(device)
    attention_mask = torch.tensor(x['attention_mask']).to(device)
    result = model.generate(input_ids=input_ids, attention_mask=attention_mask, output_scores=True, return_dict_in_generate=True)
    for j in range(len(x)):
        scores = np.array([y[j].detach().cpu().numpy() for y in result.scores])
        generated = result.sequences[j].detach().cpu().numpy()
        end_idx = np.where(generated==1)[0]
        if len(end_idx) > 0:
            outtoks = tokenizer.convert_ids_to_tokens(generated)
            final_string = tokenizer.convert_tokens_to_string(outtoks[1:end_idx[0]])
            results.append((x['hid'][j], x['orig'][j], x['corrected'][j], final_string, scores))

with open('results.pkl', 'wb') as f:
    pickle.dump(results, f, 4)
"""
df = pd.DataFrame(results, columns=['sent', 'truth', 'gen', 'scores'])
df.to_csv('new_results.csv')
"""
