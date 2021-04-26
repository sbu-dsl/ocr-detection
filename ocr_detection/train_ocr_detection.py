import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, Trainer, TrainingArguments

rand_seed = 1729
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)

class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def generate_examples(row):
    loss1, loss2 = row['loss1'], row['loss2']
    diff1, diff2 = row['diff1'], row['diff2']

    ctx1, ctx2 = row['ctx1'], row['ctx2']
    ocr1, ocr2 = ctx1[diff1[0]:diff1[1]], ctx2[diff2[0]:diff2[1]]
    if loss1 < loss2:
        tags2 = np.zeros(len(ctx2), dtype=int)
        tags2[diff2[0]:diff2[1]] = 1
        return ctx2, tags2
    else:
        tags1 = np.zeros(len(ctx1), dtype=int)
        tags1[diff1[0]:diff1[1]] = 1
        return ctx1, tags1
    

def create_datasets():
    train_path = Path('tok_train_dataset')
    val_path = Path('tok_val_dataset')
    test_path = Path('tok_test_dataset')
    if train_path.exists() and val_path.exists() and test_path.exists():
        print("Getting cached train")
        tr = torch.load(train_path)
        print("Getting cached dev")
        val = torch.load(val_path)
        print("Getting cached test")
        test = torch.load(test_path)
        return tr, val, test

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', add_prefix_space=True)

    print("Reading train CSV")
    trainp = "/home/allekim/ocr-detection/ocr_data/train.csv"
    testp = "/home/allekim/ocr-detection/ocr_data/test.csv"
    train_df = pd.read_csv(trainp, converters={'ctx1': eval, 'ctx2': eval, 'diff1': eval, 'diff2': eval})
    train_df = train_df.sample(1000000, random_state=rand_seed)
    train_df[['text','sequence']] = train_df.apply(generate_examples,axis=1,result_type='expand')
    
    print("Reading test CSV")
    test_df = pd.read_csv(testp, converters={'ctx1': eval, 'ctx2': eval, 'diff1': eval, 'diff2': eval})
    test_df = test_df.sample(200000, random_state=rand_seed)
    test_df[['text','sequence']] = test_df.apply(generate_examples,axis=1,result_type='expand')

    print("Splitting CSV")
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=rand_seed)

    train_texts, val_texts, test_texts = map(list, [train_df['text'], val_df['text'], test_df['text']])
    train_tags, val_tags, test_tags = map(list, [train_df['sequence'], val_df['sequence'], test_df['sequence']])

    print("Tokenizing data")
    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

    def tok_to_subtok_idx(offsets):
        idx_mapping = []
        local_map = []
        for idx, pair in enumerate(offsets):
            s, e = pair
            if e == 0:
                continue
            if s == 1:
                if local_map:
                    idx_mapping.append(local_map)
                    local_map = []
            local_map.append(idx)
        if local_map:
            idx_mapping.append(local_map)
        return idx_mapping

    def encode_tags(tags, encodings):
        tok_mappings = [tok_to_subtok_idx(enc) for enc in encodings.offset_mapping]
        encoded_labels = []
        for tag_idx, tag in tqdm(enumerate(tags)):
            local_labels = np.ones(len(encodings.input_ids[tag_idx]),dtype=int)*-100
            for idx, lab in enumerate(tag):
                num_lab = 0 if lab == 0 else 1
                if idx < len(tok_mappings[tag_idx]):
                    local_labels[tok_mappings[tag_idx][idx]] = num_lab
            encoded_labels.append(local_labels.tolist())
        return encoded_labels

    print("Encoding labels")
    train_labels = encode_tags(train_tags, train_encodings)
    val_labels = encode_tags(val_tags, val_encodings)
    test_labels = encode_tags(test_tags, test_encodings)

    train_encodings.pop("offset_mapping")    
    val_encodings.pop("offset_mapping")
    test_encodings.pop("offset_mapping")
    train_dataset = OCRDataset(train_encodings, train_labels)
    val_dataset = OCRDataset(val_encodings, val_labels)
    test_dataset = OCRDataset(test_encodings, test_labels)
    torch.save(train_dataset, train_path)
    torch.save(val_dataset, val_path)
    torch.save(test_dataset, test_path)
    return train_dataset, val_dataset, test_dataset


train_dataset, val_dataset, test_dataset = create_datasets()

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = RobertaForTokenClassification.from_pretrained('roberta-large')

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
trainer.save_model('ocr_detection_model')

def compute_metrics(pred, labels):
    real_labels = labels != -100
    labels = labels[real_labels]
    preds = pred.argmax(-1)[real_labels]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

results = trainer.predict(test_dataset)
print(compute_metrics(results.predictions, results.label_ids))

