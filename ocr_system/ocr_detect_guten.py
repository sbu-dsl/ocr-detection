import torch
import pandas as pd
import numpy as np
from lxml import etree
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, Trainer, TrainingArguments

print("Loading tokenizer")
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', add_prefix_space=True)

def encode(examples):
    model_inputs = tokenizer(examples['sent'], is_split_into_words=True, padding='max_length', truncation=True, return_offsets_mapping=True)
    return model_inputs

def create_guten_dataset(gid):
    outp = Path('{}_guten_input.dt'.format(gid))
    guten_path = Path("/home/cpethe/stonybook-data/gutenberg/processed/{}/character_coref_annotated.xml".format(gid))
    root = etree.parse(str(guten_path))
    sents = []
    labels = []
    for s in root.iter('s'):
        sent = []
        for t in s.iter('t'):
            sent.append(t.text)
        sents.append(sent)
        labels.append([0]*len(sent))

    dataset = Dataset.from_dict({'sent': sents})
    dataset = dataset.map(encode, batched=True)
    torch.save(dataset, outp)
    return dataset


guten_id = 64317
print("Loading model")
# model = RobertaForTokenClassification.from_pretrained('/home/allekim/ocr-detection/ocr_detection/ocr_detection_model')
model = RobertaForTokenClassification.from_pretrained('/home/allekim/stonybook-dev/hathi_similarity/new_ocr_detection/tok_model')

print("Parsing book {}".format(guten_id))
test_dataset = create_guten_dataset(guten_id)


trainer = Trainer(
    model=model
)

results = trainer.predict(test_dataset)
torch.save(results, '{}_guten_results.dt'.format(guten_id))

