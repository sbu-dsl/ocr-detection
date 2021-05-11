import torch
import pandas as pd
import numpy as np
import pickle
from lxml import etree
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, Trainer, TrainingArguments

def parse_hathi_content(hathi_id):
    hathi_path = Path('/home/allekim/stonybook-data/hathi/processed/{}/{}/parsed_content.pkl'.format(*hathi_id.split('.')))
    if not hathi_path.exists():
        return None
    with open(hathi_path, 'rb') as f:
        sents, tokens, tok_to_sent_idx = pickle.load(f)
    return sents 


print("Loading tokenizer")
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', add_prefix_space=True)

def encode(examples):
    model_inputs = tokenizer(examples['sent'], is_split_into_words=True, padding='max_length', truncation=True, return_offsets_mapping=True, return_tensors='np')
    return model_inputs

def create_hathi_dataset(base_path, hathi_id):
    outp = base_path / '{}_hathi_input.dt'.format(hathi_id)
    """
    if outp.exists():
        return torch.load(outp)
    """
    sents = parse_hathi_content(hathi_id)
    if sents == None:
        return None

    dataset = Dataset.from_dict({'sent_idx': range(len(sents)), 'sent': sents})
    dataset = dataset.map(encode, num_proc=20, batched=True)
    torch.save(dataset, outp)
    return dataset


print("Loading model")
# model = RobertaForTokenClassification.from_pretrained('/home/allekim/ocr-detection/ocr_detection/ocr_detection_model')
model = RobertaForTokenClassification.from_pretrained('/home/allekim/stonybook-dev/hathi_similarity/new_ocr_detection/tok_model')

trainer = Trainer(
    model=model
)

paths = Path("/home/allekim/stonybook-data/guten_hathi_alignment").glob('*/*.txt')
for book_path in paths:
    hathi_id = book_path.name[:-4]
    base_path = book_path.parent
    print(base_path)
    test_dataset = create_hathi_dataset(base_path, hathi_id)
    print("Loaded dataset")
    if test_dataset == None:
        continue

    print("Predicting")
    results = trainer.predict(test_dataset)
    torch.save(results, base_path / '{}_hathi_results.dt'.format(hathi_id))

