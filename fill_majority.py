#!/usr/bin/env python3
import csv
from collections import Counter
import shutil

input_path = '/Users/yongxs/Desktop/VEAT Submission/Data/Sora - Human Evaluation - Merged Results.csv'
backup_path = input_path + '.bak'

# Column names as seen in the CSV (note small typos/spaces preserved)
gender_cols = ['Annotator 1 Evaluation fo Gender', 'Annotator 2 Evaluation fo Gender', 'Annotator 3 Evaluation fo Gender']
race_cols = ['Annotator 1 Evaluation fo Race', 'Annotator 2 Evaluation fo Race', 'Annotator 3 Evaluation fo Race']
maj_gender_col = 'Majority Vote for Gender'
maj_race_col = 'Majority Vote for Race '

# Make a backup first
shutil.copyfile(input_path, backup_path)

rows = []
with open(input_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames[:] if reader.fieldnames else []
    for r in reader:
        # compute gender majority
        gvals = [ (r.get(c) or '').strip() for c in gender_cols ]
        rval = [ (r.get(c) or '').strip() for c in race_cols ]

        def majority(vals):
            vals_nonempty = [v for v in vals if v!='']
            if not vals_nonempty:
                return ''
            counts = Counter(vals_nonempty)
            most = counts.most_common()
            top_count = most[0][1]
            top_candidates = [val for val,cnt in most if cnt==top_count]
            if len(top_candidates)==1:
                return top_candidates[0]
            # tie: prefer annotator1's value if it's among the tied candidates
            if vals[0] in top_candidates and vals[0] != '':
                return vals[0]
            # else return first candidate deterministically
            return top_candidates[0]

        r[maj_gender_col] = majority(gvals)
        r[maj_race_col] = majority(rval)
        rows.append(r)

# Ensure majority columns are in header
if maj_gender_col not in fieldnames:
    fieldnames.append(maj_gender_col)
if maj_race_col not in fieldnames:
    fieldnames.append(maj_race_col)

with open(input_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print('Updated', input_path)
print('Backup saved to', backup_path)
