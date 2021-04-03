def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0 for _ in range(n+1)] for _ in range(m+1)] 
  
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0: 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1] + 1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
  
    index = L[m][n] 
    lcs = [(0,0)] * (index) 
    i = m 
    j = n 
    while i > 0 and j > 0: 
        if X[i-1] == Y[j-1]: 
            lcs[index-1] = (i-1, j-1)
            i-=1
            j-=1
            index-=1
        elif L[i-1][j] > L[i][j-1]: 
            i-=1
        else: 
            j-=1
    return lcs

def lcs_extended(X, Y, window_len=1000, overlap_ratio=0.8):
    full_match = []
    idx1 = 0
    idx2 = 0
    while True:
        curr_match = [(idx1+x, idx2+y) for x, y in lcs(X[idx1:idx1+window_len], Y[idx2:idx2+window_len])]
        if not curr_match:
            break
        overlap_idx = int(overlap_ratio * len(curr_match))
        idx1, idx2 = curr_match[overlap_idx][0] + 1, curr_match[overlap_idx][1] + 1
        full_match.extend(curr_match[:overlap_idx+1])
    return full_match

def unique_toks_idx(toks):
    seen = set()
    uniq = {}
    for idx, tok in enumerate(toks):
        if not tok.isalpha() or not tok.islower():
            continue
        if tok not in seen:
            uniq[tok] = idx
            seen.add(tok)
        else:
            if tok in uniq:
                del uniq[tok]
    return uniq

def anchor_idxs(dict1, dict2):
    lst1 = list(dict1)
    lst2 = list(dict2)
    matches = lcs(lst1, lst2)
    return [(dict1[lst1[x]], dict2[lst2[y]]) for x,y in matches]

def lcs_with_anchors(toks1, toks2, max_gap=2000):
    uniq1 = unique_toks_idx(toks1)
    uniq2 = unique_toks_idx(toks2)
    anchors = anchor_idxs(uniq1, uniq2)
    full_match = []
    idx1, idx2 = 0, 0
    for anchor1, anchor2 in anchors + [(len(toks1), len(toks2))]:
        diff1, diff2 = anchor1 - idx1, anchor2 - idx2
        if diff1 > max_gap or diff2 > max_gap:
            curr_match = [(idx1+x, idx2+y) for x, y in lcs_extended(toks1[idx1:anchor1], toks2[idx2:anchor2])]
        else:
            curr_match = [(idx1+x, idx2+y) for x, y in lcs(toks1[idx1:anchor1], toks2[idx2:anchor2])]
        idx1, idx2 = anchor1 + 1, anchor2 + 1
        full_match.extend(curr_match)
        full_match.append((anchor1, anchor2))
    # ignore last artifical anchors
    return full_match[:-1]

