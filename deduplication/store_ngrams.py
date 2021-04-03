import pickle
from pathlib import Path
from lxml import etree
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from nltk import ngrams, word_tokenize
from multiprocessing import Pool

def write_ngram(hathi_dir):
    with open(hathi_dir / "content.txt") as f:
        text = ' '.join(f.read().strip().split())
    sentences = sent_tokenize(text)
    tokens = []
    token_to_sentence_idx = []
    for idx, sentence in enumerate(sentences):
        sent_tokens = word_tokenize(sentence)
        tokens.extend(sent_tokens)
        token_to_sentence_idx.extend([idx]*len(sent_tokens))
    with open(hathi_dir / "parsed_content.pkl", "wb") as f:
        pickle.dump((sentences, tokens, token_to_sentence_idx), f, 4)
    tokens = [x for x in tokens if x.isalnum()]
    with open(hathi_dir / "fivegram.pkl", "wb") as f:
        fivegrams = set(ngrams(tokens, 5))
        pickle.dump(fivegrams, f, 4)

all_hathi_dirs = list(Path("/home/allekim/stonybook-data/hathi/ocr_detection").glob('*/*'))

"""
for idx, p in enumerate(book_paths):
    print(idx, p)
    write_ngram(p)
    break
"""
with Pool(48) as p:
    for _ in tqdm(p.imap(write_ngram, all_hathi_dirs), total=len(all_hathi_dirs)):
        pass
