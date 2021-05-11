from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pickle
from collections import Counter

ocr_data_path = list(Path("/home/allekim/stonybook-data/hathi/ocr_model_results/double_books/").glob('*'))
def generate_examples(row):
    loss1, loss2 = row['loss1'], row['loss2']
    diff1, diff2 = row['diff1'], row['diff2']
    ctx1, ctx2 = row['ctx1'], row['ctx2']
    ocr1, ocr2 = tuple(ctx1[diff1[0]:diff1[1]]), tuple(ctx2[diff2[0]:diff2[1]])
    if loss1 < loss2:
        return (ocr2, ocr1)
    else:
        return (ocr1, ocr2)

c = Counter()
print(len(ocr_data_path))
for ocr_path in tqdm(ocr_data_path):
    df = pd.read_csv(ocr_path, converters={'ctx1': eval, 'ctx2': eval, 'diff1': eval, 'diff2': eval})
    ocr_pairs = df.apply(generate_examples, axis=1)
    c.update(ocr_pairs)
with open('ocr_pairs.pkl', 'wb') as f:
    pickle.dump(c, f, 4)
