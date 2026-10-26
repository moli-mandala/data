"""Render pending Tamil review manifests; no source or graph writes."""
import collections
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
DEST = P.parent / 'Tamil'
DEST.mkdir(exist_ok=True)
decisions = json.loads((P / 'proposed-decisions.json').read_text())
articles = json.loads((P / 'primary-evidence.json').read_text())
accepted, held = decisions['accepted'], decisions['held']
assert decisions['status'] == 'pending-review'
ids = {x['parent'] for x in accepted}
parents = {r['ID']: r for r in csv.DictReader((DATA / 'cldf/forms.csv').open()) if r['ID'] in ids}
assert set(parents) == ids
groups = collections.OrderedDict()
for item in accepted:
    groups.setdefault(item['parent'], []).append(item)
existing = [int(f.stem.split('-')[1]) for f in DEST.glob('batch-*.json') if f.stem.split('-')[1].isdigit()]
start = max(existing, default=0) + 1
owned = []
links = []
for offset in range(0, len(groups), 5):
    number = start + offset // 5
    subset = list(groups.items())[offset:offset + 5]
    path = DEST / f'batch-{number:03d}.json'
    assert not path.exists(), path
    props = []
    lines = [f'**Next {len(subset)} proposals, ready for review. Nothing in this batch saved yet.**', '',
             f'Tamil · DravLex residual records · batch {number:03d}. Pending user review; autonomous research continues.', '',
             '| # | Tamil | Proposed etymology | Evidence |', '|---|---|---|---|']
    for j, (parent, items) in enumerate(subset, 1):
        article = articles[parent]
        page = article['Page']
        url = f'https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={page}'
        form_label = ', '.join('**' + x['record']['Form'].replace('|', '\\|') + '** ‘' + x['record']['Gloss'] + '’' for x in items)
        evidence = items[0]['evidence']
        lines.append(f'| {j} | {form_label} | Proposed reflex of [**DEDR {parent[1:]}**]({url}) | {evidence} ({len(items)} record' + ('s' if len(items) != 1 else '') + '.) |')
        assignments = [dict(Form_ID=x['record']['ID'], Etymon_ID=parent, Kind=x['kind'], Rank='1',
                            Status='proposed', Source=x['citation'], Notes=x['evidence'], Pos='') for x in items]
        props.append(dict(number=j, status='pending-review', parentId=parent, parentForm=parents[parent]['Form'],
                          kind='reflex', citation=items[0]['citation'], evidence=evidence,
                          formIds=sorted(x['record']['ID'] for x in items), records=[x['record'] for x in items],
                          assignments=assignments))
    affected = sum(len(p['records']) for p in props)
    lines += ['', f'Proposed assignment rows: **{affected}**. Affected records: **{affected}**. Relation kinds: **reflex**.',
              '', f'Exact persistent IDs and complete records: [{path.name}]({path}).', '',
              'Ambiguous loans, unparsed responses, and unsupported cognate suggestions are excluded; see the [research holds](../tamil-dravlex-20260925/research-holds.json). DEDR nodes denote whole article families.']
    path.write_text(json.dumps(dict(language='Tamil', batch=number, status='pending-review', date='2026-09-25',
                                    researchDirectory=str(P), proposals=props), ensure_ascii=False, indent=2) + '\n')
    review = path.with_name(path.stem + '-review.md')
    review.write_text('\n'.join(lines) + '\n')
    owned.append(str(path))
    links.append(f'[{number:03d}]({review})')

(P / 'review-manifest-paths.json').write_text(json.dumps({'Tamil': owned}, indent=2) + '\n')
(P / 'research-holds.json').write_text(json.dumps(held, ensure_ascii=False, indent=2) + '\n')
allrows = json.loads((P / 'survey-inventory.json').read_text())
external = json.loads((P / 'externally-assigned-records.json').read_text())
prior_p = json.loads((P / 'previously-proposed-records.json').read_text())
prior_h = json.loads((P / 'previously-held-records.json').read_text())
covered = {x['record']['ID'] for x in accepted + held + prior_p + prior_h} | {x['ID'] for x in external}
assert len(covered) == len(allrows) == 145
unexamined = [r for r in allrows if r['ID'] not in covered]
assert not unexamined
(P / 'unexamined-inventory.json').write_text('[]\n')
index = ['# Tamil DravLex residual research', '', '**Pending review. No new assignments saved. No rebuild run.**', '',
         f'The 145-record survey has {len(prior_p)} earlier proposals, {len(prior_h)} earlier holds, '
         f'{len(accepted)} new proposed assignment rows across {len(groups)} DEDR article families, and {len(held)} new research holds. '
         'Every source record has a review state; no Tamil DravLex records remain unexamined.', '',
         'Review batches: ' + ', '.join(links) + '.', '',
         'The upstream DravLex annotations, current homepage search, and full shortlisted DEDR articles were checked. '
         'The [primary evidence](primary-evidence.json), [source records](source-record-evidence.json), '
         '[upstream annotations](upstream-audit-evidence.json), [decisions](proposed-decisions.json), and '
         '[holds](research-holds.json) preserve the basis for each decision.', '',
         'Notable holds include the two-word “die” response, an unresolved know+ko formation, and source-flagged loans without verified immediate donors. '
         'The DEDR article-level treatment includes Tamil demonstrative variants whose exact survey spellings are not printed as article heads. No source text, parent heading, or accepted sidecar was edited.']
(P / 'REVIEW.md').write_text('\n'.join(index) + '\n')
print(json.dumps({'proposed': len(accepted), 'assignmentRows': len(accepted), 'held': len(held), 'unexamined': len(unexamined), 'batches': len(owned)}))
