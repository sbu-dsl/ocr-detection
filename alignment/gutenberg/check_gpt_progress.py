from pathlib import Path

guten_hathi_books = list(Path('/home/allekim/stonybook-data/guten_hathi_alignment').glob('*/model_scored_toks.csv'))
print(len(guten_hathi_books))
