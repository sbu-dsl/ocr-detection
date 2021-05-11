import pickle
from pathlib import Path
from lxml import etree
from tqdm import tqdm
from multiprocessing import Pool

def parse_xml_toks(path):
    if not path.exists():
        return None
    try:
        root = etree.parse(str(path))
    except:
        return None
    sents = []
    toks = []
    tok_to_sent_idx = []
    for s in root.iter('s'):
        sent = []
        for t in s.iter('t'):
            sent.append(t.text)
        sents.append(sent)
        toks.extend(sent)
        sent_idx = int(s.get('num'))
        tok_to_sent_idx.extend([sent_idx] * len(sent))
    return sents, toks, tok_to_sent_idx


def write_ngram(hathi_dir):
    xml_path = Path(hathi_dir / "character_coref_annotated.xml")
    vals = parse_xml_toks(xml_path)
    if vals == None:
        return None
    with open(hathi_dir / "parsed_content.pkl", "wb") as f:
        pickle.dump(vals, f, 4)

all_hathi_dirs = list(Path("/home/allekim/stonybook-data/hathi/processed").glob('*/*'))


# for idx, p in enumerate(all_hathi_dirs):
#     print(idx, p)
#     write_ngram(p)
#     break

with Pool(30) as p:
    for _ in tqdm(p.imap(write_ngram, all_hathi_dirs), total=len(all_hathi_dirs)):
        pass
