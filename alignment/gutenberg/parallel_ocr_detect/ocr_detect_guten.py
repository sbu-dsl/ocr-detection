import torch
import numpy as np
import pickle
import sys
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from torch.utils.data._utils.collate import default_convert
from transformers import RobertaTokenizerFast, RobertaForTokenClassification

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
    model_inputs = tokenizer(examples['sent'], is_split_into_words=True, padding='max_length', truncation=True, return_offsets_mapping=True)
    return model_inputs

def create_hathi_dataset(base_path, hathi_id):
    outp = base_path / '{}_hathi_input.dt'.format(hathi_id)
    if outp.exists():
        return torch.load(outp)
    sents = parse_hathi_content(hathi_id)
    if sents == None:
        return None

    dataset = Dataset.from_dict({'sent_idx': range(len(sents)), 'sent': sents})
    dataset = dataset.map(encode, batched=True)
    torch.save(dataset, outp)
    return dataset

if len(sys.argv) != 3:
    raise ValueError("Need GPU Num and Proc ID")

print("Loading model")
# model = RobertaForTokenClassification.from_pretrained('/home/allekim/ocr-detection/ocr_detection/ocr_detection_model')
model = RobertaForTokenClassification.from_pretrained('/home/allekim/stonybook-dev/hathi_similarity/new_ocr_detection/tok_model')

gpu_num = sys.argv[1]
print("GPU Num: {}".format(gpu_num))
device = 'cuda:{}'.format(gpu_num)
model.to(device)
model.eval()

with open('all_paths.txt') as f:
    paths = [Path(x) for x in f.readlines()]

proc_id = int(sys.argv[2])
total_proc = 9
num_per_proc = len(paths) // total_proc + 1
lidx = num_per_proc*proc_id
ridx = num_per_proc*(proc_id+1)
print("Processing lines {} to {}".format(lidx, ridx))
local_paths = paths[lidx:ridx]
for book_idx, book_path in enumerate(local_paths):
    print(book_path)
    hathi_id = book_path.name[:-9]
    print(hathi_id)
    base_path = book_path.parent
    outpath = base_path / '{}_hathi_batched_results.dt'.format(hathi_id)
    if outpath.exists():
        continue
    print(book_idx, len(local_paths), base_path)
    test_dataset = create_hathi_dataset(base_path, hathi_id)
    if test_dataset == None:
        print("Empty dataset?")
        continue
    print("Loaded dataset")
    test_dataset = test_dataset.remove_columns(['sent_idx', 'sent', 'offset_mapping'])
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, collate_fn=default_convert)
    results = []
    for i, batch in enumerate(dataloader):
        print("Batch {}".format(i), len(dataloader))
        input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch]).to(device)
        attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in batch]).to(device)
        x = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = x.logits.detach().cpu().numpy()
        results.append(logits)
    torch.save(np.concatenate(results), outpath)


