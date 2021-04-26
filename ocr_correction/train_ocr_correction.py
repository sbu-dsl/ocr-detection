import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import ReformerModelWithLMHead, ReformerConfig

class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.encodings['input_ids'][idx].clone().detach()
        return item

    def __len__(self):
        return len(self.encodings)


def roundUpToMultiple(number, multiple):
    num = number + (multiple - 1)
    return num - (num % multiple)

def encode(list_of_strings, pad_token_id=0):
    max_length = roundUpToMultiple(max([len(string) for string in list_of_strings]), 65536)

    attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
    input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

    for idx, string in enumerate(list_of_strings):
        if not isinstance(string, bytes):
            string = str.encode(string)

        input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
        attention_masks[idx, :len(string)] = 1

    return {'input_ids': input_ids, 'attention_mask': attention_masks}


def generate_examples(row):
    loss1, loss2 = row['loss1'], row['loss2']
    diff1, diff2 = row['diff1'], row['diff2']
    ctx1, ctx2 = row['ctx1'], row['ctx2']
    if any(['*' in x for x in ctx1]) or any(['*' in x for x in ctx2]):
        return np.nan
    ocr1, ocr2 = ctx1[diff1[0]:diff1[1]], ctx2[diff2[0]:diff2[1]]
    ex1 = ' '.join(ctx1[:diff1[0]]) + '*' + ' '.join(ocr1) + '*' + ' '.join(ctx1[diff1[1]:])
    ex2 = ' '.join(ctx2[:diff2[0]]) + '*' + ' '.join(ocr2) + '*' + ' '.join(ctx2[diff2[1]:])
    if loss1 < loss2:
        correct = '#' + ' '.join(ocr1) + '#'    
    else:
        correct = '#' + ' '.join(ocr2) + '#'
    return (ex1 + correct, ex2 + correct)
        

ocr_path = Path("/home/allekim/stonybook-data/hathi/ocr_model_results/double_books/")
result_paths = list(ocr_path.glob('*'))
df = pd.read_csv(result_paths[1], converters={'ctx1': eval, 'ctx2': eval, 'diff1': eval, 'diff2': eval})
df['examples'] = df.apply(generate_examples, axis=1)
result = [e for l in df['examples'].dropna() for e in l]

device = torch.device("cuda:1")
model = ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8")

model.to(device)
training_data = OCRDataset(encode(result[:450]))
test_data = OCRDataset(encode(result[450:]))

train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
optimizer = AdamW(model.parameters(), lr=1e-5)

model.train()
for batch in train_dataloader:
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    output = model(input_ids, attention_mask=attention_mask, labels=labels)
    output.loss.backward()
    optimizer.step()

