#!/usr/bin/env python3
"""Candidate table: Sheth 1923 Sanskrit equivalents -> CDIAL headwords, plus their Prakrit senses.

Reads the compiled graph (cldf/forms.csv, cldf/edges.csv) and the identity registry, writes
candidates.csv (one row per Sheth Sanskrit form) and prints tier statistics.
"""
import csv, collections, re, sys, unicodedata, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
csv.field_size_limit(10**9)

SUPER = dict.fromkeys(map(ord, '¹²³⁴⁵⁶⁷⁸⁹ˋ'), None)

def canon(s: str) -> str:
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if c not in '́̀')
    s = s.translate(SUPER)
    s = s.replace('r̥', 'ṛ').replace('r̩', 'ṛ').replace('l̥', 'ḷ').replace('l̩', 'ḷ')
    s = unicodedata.normalize('NFC', s)
    s = s.replace('ʰ', 'h').replace('ŋ', 'ṅ').replace('ṃ', 'ṁ').replace('ḣ', 'ḥ')
    return s.strip()

# ---------------------------------------------------------------- CDIAL heads
heads = collections.defaultdict(list)      # canon -> [(id, form, gloss, starred)]
head_info = {}
with open(ROOT / 'cldf/forms.csv', encoding='utf-8', newline='') as f:
    for r in csv.DictReader(f):
        if r['Language_ID'] == 'Indo-Aryan' and r['Source'].startswith('CDIAL') and r['Status'] == 'entry' and not r['Redirect']:
            orig = r['Original'] or r['Form']
            starred = orig.startswith('*')
            key = canon(orig.lstrip('*').rstrip('-'))
            heads[key].append(r['ID'])
            head_info[r['ID']] = (orig, r['Gloss'], starred)
print('cdial entries', len(head_info), 'distinct canon', len(heads), file=sys.stderr)

# ---------------------------------------------------------------- existing rank-1 edges
rank1 = {}
with open(ROOT / 'cldf/edges.csv', encoding='utf-8', newline='') as f:
    for r in csv.DictReader(f):
        if r['Rank'] == '1' and r['Kind'] in ('reflex', 'borrowed', 'variant', 'derived'):
            rank1[r['Child_ID']] = (r['Parent_ID'], r['Kind'], r['Source'])

# ---------------------------------------------------------------- Sheth rows
sk_rows, pk_rows = {}, collections.defaultdict(list)
with open(ROOT / 'cldf/forms.csv', encoding='utf-8', newline='') as f:
    for r in csv.DictReader(f):
        if not r['Source'].startswith('sheth1923') or r['Redirect']:
            continue
        if r['Language_ID'] == 'Sk':
            sk_rows[r['ID']] = r
        elif r['Language_ID'] in ('Pk', 'Ap', 'As'):
            pk_rows[r['ID']] = r
key_of = {}
with open(ROOT / 'data/form-identities.csv', encoding='utf-8', newline='') as f:
    for r in csv.DictReader(f):
        if r['Source_Key'].startswith('sheth1923:') and r['Status'] == 'active':
            key_of[r['Form_ID']] = r['Source_Key']
print('sk', len(sk_rows), 'pk', len(pk_rows), 'keyed', len(key_of), file=sys.stderr)

# sense key -> pk form ids
sense_pk = collections.defaultdict(list)
for fid in pk_rows:
    k = key_of.get(fid)
    if k:
        sense_pk[re.sub(r':v\d+$', '', k)].append(fid)

def match(form):
    c = canon(form)
    if c in heads:
        return heads[c], 'exact'
    if c.endswith('ḥ') and c[:-1] in heads:
        return heads[c[:-1]], 'visarga'
    return [], 'none'

fields = ['Sk_ID', 'Sk_Form', 'Native', 'Sense_Key', 'Tier', 'Candidates', 'Parent', 'Parent_Form', 'Parent_Gloss',
          'Starred', 'Pk_IDs', 'Pk_Forms', 'Pk_Gloss', 'Existing_Parent', 'Existing_Pk_Parents']
tiers = collections.Counter(); consist = collections.Counter()
rows = []
for fid, r in sk_rows.items():
    k = key_of.get(fid, '')
    sense = k[:-len(':sanskrit')] if k.endswith(':sanskrit') else ''
    cands, how = match(r['Form'])
    pk_ids = sense_pk.get(sense, [])
    pk_forms = '; '.join(pk_rows[p]['Form'] for p in pk_ids)
    pk_gloss = ' | '.join(dict.fromkeys(pk_rows[p]['Gloss'] for p in pk_ids if pk_rows[p]['Gloss']))
    ex_pk = {rank1[p][0] for p in pk_ids if p in rank1}
    if len(cands) == 1:
        pid = cands[0]; orig, gloss, starred = head_info[pid]
        tier = ('unique' if how == 'exact' else 'unique-visarga') + ('-starred' if starred else '')
        for p in pk_ids:
            if p in rank1:
                consist['agree' if rank1[p][0] == pid else 'disagree'] += 1
    elif cands:
        pid, orig, gloss, starred, tier = '', '', '', False, 'ambiguous'
    else:
        pid, orig, gloss, starred, tier = '', '', '', False, 'none'
    tiers[tier] += 1
    rows.append(dict(Sk_ID=fid, Sk_Form=r['Form'], Native=r['Native'], Sense_Key=sense, Tier=tier,
                     Candidates=' '.join(f"{c}={head_info[c][0]}«{head_info[c][1][:40]}»" for c in cands),
                     Parent=pid, Parent_Form=orig, Parent_Gloss=gloss, Starred='*' if starred else '',
                     Pk_IDs=' '.join(pk_ids), Pk_Forms=pk_forms, Pk_Gloss=pk_gloss,
                     Existing_Parent=rank1.get(fid, ('',))[0], Existing_Pk_Parents=' '.join(sorted(ex_pk))))
with open(OUT / 'candidates.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
print('tiers', dict(tiers), file=sys.stderr)
print('pk consistency with existing rank-1 edges', dict(consist), file=sys.stderr)
pk_cov = sum(len(sense_pk.get(r['Sense_Key'], [])) for r in rows if r['Tier'].startswith('unique'))
print('pk rows covered by unique tier', pk_cov, file=sys.stderr)
