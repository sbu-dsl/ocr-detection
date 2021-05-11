from pathlib import Path
from multiprocessing import shared_memory

paths = list(Path("/home/allekim/stonybook-data/guten_hathi_alignment").glob('*/*_new.txt'))

with open('all_paths.txt', 'w') as f:
    for path in paths:
        f.write(str(path) + '\n')
    
