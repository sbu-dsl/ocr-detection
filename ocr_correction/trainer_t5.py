import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import csv
import numpy as np
import random

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
        return ex2, correct
    else:
        if ocr2:
            correct = ' '.join(ocr2)
        return ex1, correct

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
model_name = "t5-base"

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.add_tokens(["<ocr>", "</ocr>", "<blank>"], special_tokens=True)        
tokenizer.add_special_tokens({"additional_special_tokens": ["<ocr>", "</ocr>", "<blank>"]})


print("Loading train")
num_samples = 1000000
train_path = '/home/allekim/ocr-detection/ocr_data/train.csv'
df = pd.read_csv(train_path, converters={'ctx1': eval, 'ctx2': eval, 'diff1': eval, 'diff2': eval}, nrows=num_samples)
df[['orig','corrected']] = df.apply(generate_examples, axis=1, result_type="expand")
train_data, val_data = train_test_split(df[['orig','corrected']], test_size=0.2, random_state=seed)
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=48,
)

val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=48,
)

print("Loading model")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

print("Data collator")
label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)


training_args = Seq2SeqTrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model('ocr_correction_model')

