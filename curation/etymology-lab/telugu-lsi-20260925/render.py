"""Render the pending Telugu LSI checkpoint without touching accepted sources."""
import collections
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
DEST = P.parent / 'Telugu'
DEST.mkdir(exist_ok=True)
decisions = json.loads((P / 'proposed-decisions.json').read_text())
articles = json.loads((P / 'primary-evidence.json').read_text())
accepted, held = decisions['accepted'], decisions['held']
assert decisions['status'] == 'pending-review'
ids = {part for x in accepted for part in x.get('components', [x['parent']])}
parents = {r['ID']: r for r in csv.DictReader((DATA / 'cldf/forms.csv').open()) if r['ID'] in ids}
assert set(parents) == ids
groups = collections.OrderedDict()
for item in accepted:
    groups.setdefault((tuple(item.get('components', [item['parent']])), item['evidence']), []).append(item)
old_owned = json.loads((P / 'review-manifest-paths.json').read_text()).get('Telugu', []) if (P / 'review-manifest-paths.json').exists() else []
existing = [int(f.stem.split('-')[1]) for f in DEST.glob('batch-*.json') if f.stem.split('-')[1].isdigit()]
start = max(existing, default=0) + 1
owned = []
links = []
for offset in range(0, len(groups), 5):
    batch_index = offset // 5
    number = int(Path(old_owned[batch_index]).stem.split('-')[1]) if batch_index < len(old_owned) else start + batch_index - len(old_owned)
    subset = list(groups.items())[offset:offset + 5]
    path = DEST / f'batch-{number:03d}.json'
    if path.exists():
        old = json.loads(path.read_text())
        assert old.get('researchDirectory') == str(P), path
    props = []
    lines = [f'**Next {len(subset)} proposals, ready for review. Nothing in this batch saved yet.**', '',
             f'Telugu · LSI 1928 residual records · batch {number:03d}. Pending user review; autonomous research continues.', '',
             '| # | Telugu | Proposed etymology | Evidence |', '|---|---|---|---|']
    for j, ((parent_parts, evidence), items) in enumerate(subset, 1):
        parent = parent_parts[0]
        article = articles[parent]
        url = f'https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={article["Page"]}'
        form_label = ', '.join('**' + x['record']['Form'].replace('|', '\\|') + '** ‘' + x['record']['Gloss'] + '’' for x in items)
        kind = items[0]['kind']
        if kind == 'component':
            proposed = 'Ordered components: ' + ' + '.join(
                f'[**DEDR {part[1:]}**](https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={articles[part]["Page"]})'
                for part in parent_parts)
        else:
            proposed = f'Proposed reflex of [**DEDR {parent[1:]}**]({url})'
        lines.append(f'| {j} | {form_label} | {proposed} | {evidence} ({len(items)} record' + ('s' if len(items) != 1 else '') + '.) |')
        assignments = [dict(Form_ID=x['record']['ID'], Etymon_ID=part, Kind=x['kind'], Rank='1',
                            Status='proposed', Source=x['citation'], Notes=x['evidence'], Pos=str(pos) if kind == 'component' else '')
                       for x in items for pos, part in enumerate(parent_parts, 1)]
        props.append(dict(number=j, status='pending-review', parentId=parent, parentForm=parents[parent]['Form'],
                          kind=kind, citation=items[0]['citation'], evidence=evidence,
                          formIds=sorted(x['record']['ID'] for x in items), records=[x['record'] for x in items],
                          assignments=assignments))
        if kind == 'component':
            props[-1].update(componentIds=list(parent_parts), componentForms=[parents[part]['Form'] for part in parent_parts])
    affected = sum(len(p['records']) for p in props)
    edge_count = sum(len(p['assignments']) for p in props)
    kinds = ', '.join(sorted({p['kind'] for p in props}))
    lines += ['', f'Proposed assignment rows: **{edge_count}**. Affected records: **{affected}**. Relation kinds: **{kinds}**.',
              '', f'Exact persistent IDs and complete records: [{path.name}]({path}).', '',
              'Multiword responses without full component analysis remain excluded; see the [research holds](../telugu-lsi-20260925/research-holds.json). Ordinary case and plural forms of one cited stem retain one reflex link.']
    path.write_text(json.dumps(dict(language='Telugu', batch=number, status='pending-review', date='2026-09-25',
                                    researchDirectory=str(P), proposals=props), ensure_ascii=False, indent=2) + '\n')
    review = path.with_name(path.stem + '-review.md')
    review.write_text('\n'.join(lines) + '\n')
    owned.append(str(path))
    links.append(f'[{number:03d}]({review})')

(P / 'review-manifest-paths.json').write_text(json.dumps({'Telugu': owned}, indent=2) + '\n')
(P / 'research-holds.json').write_text(json.dumps(held, ensure_ascii=False, indent=2) + '\n')
allrows = json.loads((P / 'survey-inventory.json').read_text())
external = json.loads((P / 'externally-assigned-records.json').read_text())
prior_p = json.loads((P / 'previously-proposed-records.json').read_text())
prior_h = json.loads((P / 'previously-held-records.json').read_text())
covered = {x['record']['ID'] for x in accepted + held + prior_p + prior_h} | {x['ID'] for x in external}
assert covered <= {x['ID'] for x in allrows}
unexamined = [r for r in allrows if r['ID'] not in covered]
assert len(allrows) == 167 and len(unexamined) == 83
(P / 'unexamined-inventory.json').write_text(json.dumps(unexamined, ensure_ascii=False, indent=2) + '\n')
assignment_count = sum(len(x.get('components', [x['parent']])) for x in accepted)
index = ['# Telugu LSI 1928 residual research — checkpoint 001', '',
         '**Pending review. No new assignments saved. No rebuild run.**', '',
         f'Of 167 source records, {len(prior_p)} have earlier pending proposals, {len(prior_h)} earlier holds, '
         f'{len(accepted)} have new pending proposals ({assignment_count} assignment rows) across {len(groups)} etyma, '
         f'{len(held)} have new research holds, and {len(unexamined)} remain unexamined.', '',
         'Review batches: ' + ', '.join(links) + '.', '',
         'The [source records](source-record-evidence.json), [homepage discovery](discovery-001-results.json), '
         '[full DEDR evidence](primary-evidence.json), [decisions](proposed-decisions.json), and '
         '[holds](research-holds.json) preserve the research basis. Case and plural morphology is distinguished from elicited multiword phrases.', '',
         'The remaining 83 source records are listed in [unexamined inventory](unexamined-inventory.json). '
         'The horse article itself entertains an Indo-Aryan origin. Male animal compounds with both members verified have ordered component proposals; female animal phrases remain held pending their first component. '
         'No source text, parent heading, or accepted sidecar was edited.']
(P / 'REVIEW.md').write_text('\n'.join(index) + '\n')
print(json.dumps({'proposed': len(accepted), 'assignmentRows': assignment_count, 'held': len(held), 'unexamined': len(unexamined), 'batches': len(owned)}))
