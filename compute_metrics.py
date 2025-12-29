#!/usr/bin/env python3
import csv
from collections import Counter, defaultdict

input_path = '/Users/yongxs/Desktop/VEAT Submission/Data/Sora - Human Evaluation - Merged Results.csv'

gender_cols = ['Annotator 1 Evaluation fo Gender', 'Annotator 2 Evaluation fo Gender', 'Annotator 3 Evaluation fo Gender']
race_cols = ['Annotator 1 Evaluation fo Race', 'Annotator 2 Evaluation fo Race', 'Annotator 3 Evaluation fo Race']
maj_gender_col = 'Majority Vote for Gender'
maj_race_col = 'Majority Vote for Race '
assoc_col = 'Associated Attribute'

# helper: Cohen's kappa
def cohen_kappa(a, b):
    # a and b are lists of labels (strings), same length; ignore pairs with missing ('')
    pairs = [(x.strip(), y.strip()) for x,y in zip(a,b) if x and y]
    if not pairs:
        return None
    n = len(pairs)
    obs = sum(1 for x,y in pairs if x==y) / n
    # category proportions
    pa = Counter(x for x,y in pairs)
    pb = Counter(y for x,y in pairs)
    pe = 0.0
    for cat in set(list(pa.keys()) + list(pb.keys())):
        pe += (pa.get(cat,0)/n) * (pb.get(cat,0)/n)
    if pe == 1.0:
        return 1.0
    return (obs - pe) / (1 - pe)

# helper: Fleiss' kappa for N subjects, n raters per subject, k categories
# matrix is list of lists where row i is counts per category for subject i
def fleiss_kappa(matrix):
    N = len(matrix)
    if N == 0:
        return None
    k = len(matrix[0])
    n = sum(matrix[0])
    if n == 0:
        return None
    # compute p_j
    pj = [0.0]*k
    for j in range(k):
        s = sum(matrix[i][j] for i in range(N))
        pj[j] = s / (N * n)
    # compute P_i
    Pi = []
    for i in range(N):
        s = sum(matrix[i][j] * matrix[i][j] for j in range(k))
        Pi.append((s - n) / (n * (n - 1)) if n > 1 else 0.0)
    Pbar = sum(Pi) / N
    PbarE = sum(p * p for p in pj)
    if 1 - PbarE == 0:
        return None
    kappa = (Pbar - PbarE) / (1 - PbarE)
    return kappa

# read CSV
rows = []
with open(input_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for r in reader:
        rows.append(r)

# Proportion where majority == associated attribute
# Map associated attribute 'Woman'->'Female', 'Man'->'Male'
map_assoc_to_gender = {'Woman':'Female', 'Man':'Male', 'woman':'Female', 'man':'Male'}

gender_total = 0
gender_match = 0
race_total = 0
race_match = 0

for r in rows:
    assoc = (r.get(assoc_col) or '').strip()
    majg = (r.get(maj_gender_col) or '').strip()
    majr = (r.get(maj_race_col) or '').strip()
    if assoc in map_assoc_to_gender:
        gender_total += 1
        expected = map_assoc_to_gender[assoc]
        if majg and expected and majg == expected:
            gender_match += 1
    else:
        # treat assoc as race if non-empty and not gender mapping
        if assoc:
            race_total += 1
            if majr and assoc and majr == assoc:
                race_match += 1

prop_gender = gender_match / gender_total if gender_total>0 else None
prop_race = race_match / race_total if race_total>0 else None

# Interrater reliability
# For gender: build lists per annotator
ann_g = [ [ (r.get(c) or '').strip() for r in rows ] for c in gender_cols ]
ann_r = [ [ (r.get(c) or '').strip() for r in rows ] for c in race_cols ]

# Pairwise Cohen's kappa
pairs = []
for i in range(len(gender_cols)):
    for j in range(i+1, len(gender_cols)):
        k = cohen_kappa(ann_g[i], ann_g[j])
        pairs.append((f'{gender_cols[i]} vs {gender_cols[j]}', k))

pairs_race = []
for i in range(len(race_cols)):
    for j in range(i+1, len(race_cols)):
        k = cohen_kappa(ann_r[i], ann_r[j])
        pairs_race.append((f'{race_cols[i]} vs {race_cols[j]}', k))

# Fleiss' kappa for gender
# Determine categories across all gender annotations
gender_cats = set()
for lst in ann_g:
    for v in lst:
        if v:
            gender_cats.add(v)
gender_cats = sorted(list(gender_cats))
cat_index_g = {c:i for i,c in enumerate(gender_cats)}

matrix_g = []
for idx in range(len(rows)):
    counts = [0]*len(gender_cats)
    for a in range(len(gender_cols)):
        v = ann_g[a][idx]
        if v:
            counts[cat_index_g[v]] += 1
    # Only include items with at least one rating
    if sum(counts) > 0:
        matrix_g.append(counts)

kappa_fleiss_g = fleiss_kappa(matrix_g)

# Fleiss for race
race_cats = set()
for lst in ann_r:
    for v in lst:
        if v:
            race_cats.add(v)
race_cats = sorted(list(race_cats))
cat_index_r = {c:i for i,c in enumerate(race_cats)}

matrix_r = []
for idx in range(len(rows)):
    counts = [0]*len(race_cats)
    for a in range(len(race_cols)):
        v = ann_r[a][idx]
        if v:
            counts[cat_index_r[v]] += 1
    if sum(counts) > 0:
        matrix_r.append(counts)

kappa_fleiss_r = fleiss_kappa(matrix_r)

# Print report
print('PROPORTION MAJORITY == ASSOCIATED ATTRIBUTE')
if prop_gender is None:
    print('No gender-associated rows found.')
else:
    print(f'Gender: {gender_match}/{gender_total} = {prop_gender:.4f}')
if prop_race is None:
    print('No race-associated rows found.')
else:
    print(f'Race: {race_match}/{race_total} = {prop_race:.4f}')

print('\nINTERRATER RELIABILITY - GENDER')
print('Pairwise Cohen kappa:')
for name,k in pairs:
    print(f'  {name}: {k if k is not None else "N/A"}')
print(f"Fleiss' kappa (gender): {kappa_fleiss_g if kappa_fleiss_g is not None else 'N/A'}")

print('\nINTERRATER RELIABILITY - RACE')
print('Pairwise Cohen kappa:')
for name,k in pairs_race:
    print(f'  {name}: {k if k is not None else "N/A"}')
print(f"Fleiss' kappa (race): {kappa_fleiss_r if kappa_fleiss_r is not None else 'N/A'}")

# Also print basic counts
print('\nADDITIONAL INFO')
print(f'Gender categories seen: {gender_cats}')
print(f'Race categories seen: {race_cats}')
