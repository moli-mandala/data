#!/usr/bin/env python3
"""Turn the Sheth->CDIAL candidate table plus the hand review table into decisions.json.

Rules (see REVIEW.md):
* a Sheth Sanskrit equivalent whose accent-stripped form matches exactly one CDIAL entry headword
  is linked to that entry as a reflex, unless the hand review table holds it;
* homograph matches (several CDIAL heads), bound-morpheme heads (Prakrit °…), one- or
  two-letter Sanskrit forms, and forms coinciding with a Sheth work abbreviation were reviewed by
  hand in review-decisions.tsv (group -> CDIAL id | hold, reason), grouped by review-groups.tsv;
* the Prakrit/Apabhramsha/Ashokan rows of the same Sheth sense inherit the link as reflexes,
  except alternate heads that already carry a variant edge to their primary head.
"""
import csv, collections, json, re, subprocess, sys, unicodedata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from skeleton import similarity
SKELETON_THRESHOLD = 0.5

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
csv.field_size_limit(10**9)
subprocess.run([sys.executable, str(HERE / 'build_candidates.py'), str(HERE)], check=True)

def canon(s):
    s = unicodedata.normalize('NFD', s); s = ''.join(c for c in s if c not in '́̀')
    s = s.replace('r̥', 'ṛ').replace('r̩', 'ṛ'); s = unicodedata.normalize('NFC', s)
    return s.replace('ʰ', 'h').replace('ŋ', 'ṅ').replace('ṃ', 'ṁ').replace('ḣ', 'ḥ').replace('-', '').replace('°', '').strip()

cands = list(csv.DictReader((HERE / 'candidates.csv').open(encoding='utf-8', newline='')))
groups = {}
for line in (HERE / 'review-groups.tsv').read_text(encoding='utf-8').splitlines():
    cols = line.split('\t'); groups[cols[0]] = cols[6].split()
review = {}
for line in (HERE / 'review-decisions.tsv').read_text(encoding='utf-8').splitlines():
    g, decision, reason = line.split('\t')
    for sk_id in groups[g]:
        review[sk_id] = (decision, reason)

records, rank1 = {}, {}
with (ROOT / 'cldf/forms.csv').open(encoding='utf-8', newline='') as f:
    for r in csv.DictReader(f):
        if r['Source'].startswith('sheth1923') and r['Language_ID'] in ('Sk', 'Pk', 'Ap', 'As'):
            records[r['ID']] = {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Native', 'Source', 'Tags')}
child_parent = {}
with (ROOT / 'cldf/edges.csv').open(encoding='utf-8', newline='') as f:
    for r in csv.DictReader(f):
        if r['Rank'] == '1':
            child_parent[r['Child_ID']] = r['Parent_ID']
            if r['Child_ID'] in records: rank1[r['Child_ID']] = r['Kind']
cdial_pk = collections.defaultdict(set)
with (ROOT / 'cldf/forms.csv').open(encoding='utf-8', newline='') as f:
    for r in csv.DictReader(f):
        if r['Language_ID'] == 'Pk' and r['Source'].startswith('CDIAL') and r['ID'] in child_parent:
            cdial_pk[child_parent[r['ID']]].add(r['Original'] or r['Form'])

accepted, held, stats = [], [], collections.Counter()
for c in cands:
    sk = records[c['Sk_ID']]
    pk_ids = [p for p in c['Pk_IDs'].split() if p in records]
    pk_desc = '; '.join(dict.fromkeys(records[p]['Form'] for p in pk_ids))
    gloss = next((records[p]['Gloss'] for p in pk_ids if records[p]['Gloss']), '')
    if c['Sk_ID'] in review:
        decision, reason = review[c['Sk_ID']]
        if decision == 'hold':
            stats['held:review'] += 1
            held.append({'record': sk, 'candidates': c['Candidates'], 'reason': reason}); continue
        parent, how = decision, f"homograph resolved by hand review: {reason}"
        stats['accepted:review'] += 1
    elif c['Tier'].startswith('unique'):
        parent, how = c['Parent'], f"unique CDIAL headword match ({c['Parent_Form']})"
        stats['accepted:' + c['Tier']] += 1
    else:
        stats['not-linked:' + c['Tier']] += 1; continue
    if c['Starred']:
        how += "; the CDIAL head is reconstructed, so Sheth's Sanskrit equivalent is a Sanskritisation"
    base = f"Sheth prints the Sanskrit equivalent {sk['Native']} for Prakrit {pk_desc}" + (f" '{gloss}'" if gloss else '') + '. '
    sk_record = {k: v for k, v in sk.items() if k != 'Native'}
    if c['Sk_ID'] not in rank1:
        accepted.append({'record': sk_record, 'parent': parent, 'kind': 'reflex',
                         'citation': f"{sk['Source']};CDIAL[{parent}]", 'evidence': base + how + '.'})
    own = cdial_pk.get(parent, set())
    for p in pk_ids:
        if p in rank1:
            stats[f'pk-skipped:{rank1[p]}-edge'] += 1; continue
        pk = records[p]
        hit = [o for o in own if canon(o) == canon(pk['Form'])]
        if not hit:
            score = similarity(sk['Form'], pk['Form'])
            if score < SKELETON_THRESHOLD:
                stats['pk-held:not-shaped-like-reflex'] += 1
                held.append({'record': {k: v for k, v in pk.items() if k != 'Native'}, 'candidates': f"{parent}={c['Parent_Form'] or parent}",
                             'reason': f"Sheth's Sanskrit equivalent {sk['Native']} ({sk['Form']}) is not shaped like the etymon of {pk['Form']} "
                                       f"(skeleton similarity {score:.2f}); Sheth may print a semantic equivalent here. The Sanskrit form itself is linked."})
                continue
        support = (f" CDIAL {parent} itself cites Pk. {', '.join(sorted(hit))}." if hit
                   else f" CDIAL {parent} cites Pk. {', '.join(sorted(own))}." if own
                   else f" CDIAL {parent} cites no Prakrit reflex; the link rests on Sheth's equivalent.")
        stats['pk-accepted' + (':cdial-cites-same-form' if hit else '')] += 1
        accepted.append({'record': {k: v for k, v in pk.items() if k != 'Native'}, 'parent': parent, 'kind': 'reflex',
                         'citation': f"{pk['Source']};CDIAL[{parent}]",
                         'evidence': base + how + '.' + support})
(HERE / 'decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
stats['accepted-total'] = len(accepted); stats['held-total'] = len(held)
(HERE / 'summary.json').write_text(json.dumps(dict(sorted(stats.items())), indent=1) + '\n', encoding='utf-8')
print(json.dumps(dict(sorted(stats.items())), indent=1))
