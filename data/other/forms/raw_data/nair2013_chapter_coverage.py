"""Inventory all thirteen chapter packages; does not generate lexical data or build CLDF."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

DATA = next(p for p in Path(__file__).resolve().parents if (p/'source_meta.py').exists())
CHAPTERS = [('adiya',4,8),('aranadan',9,11),('betta-kurumba',12,18),
    ('cholanaickan',19,21),('irula',22,27),('kakkala',28,31),('kani',32,33),
    ('kurichiya',34,36),('malamuthan',37,39),('malaveda',40,44),
    ('mannan',45,49),('muduga',50,54),('mullu-kurumba',55,58)]
forms = DATA/'data/other/forms'
coverage=[]
for name,start,end in CHAPTERS:
    path=forms/f'20260927-nair-{name}.csv'
    settings=path.with_suffix('.yaml')
    item=dict(chapter=name,printed_pages=[start,end],canonical_csv=str(path.relative_to(DATA)),
        canonical_settings=str(settings.relative_to(DATA)),installed_source_files=path.exists() and settings.exists())
    if path.exists():
        counts=Counter(); keys=set(); total=0
        with path.open(newline='') as stream:
            for row in csv.reader(stream):
                assert len(row)==15 and row[10] and row[10] not in keys
                keys.add(row[10]); counts[row[0]]+=1; total+=1
        item.update(rows=total,by_language=dict(sorted(counts.items())),csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    coverage.append(item)
report=dict(source='Nair2013 Tribal Languages of Kerala',chapters=coverage,
    chapters_with_canonical_source_files=sum(c['installed_source_files'] for c in coverage),
    source_rows=sum(c.get('rows',0) for c in coverage),
    interpretation='Exact source-file presence/count inventory, not proof that all source or integration gates passed. Per-chapter audits and handoffs remain authoritative.',
    remaining_chapters=[c['chapter'] for c in coverage if not c['installed_source_files']],
    deferred=['Full CLDF/build and compiled graph/identity/reference survival','Full suite','User-triggered browser/database refresh'],
    restrictions='No remote work, database generation or publication performed by this inventory.')
target=DATA/'source_checklists/20260927-nair2013-chapter-coverage.json'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='chapters'},ensure_ascii=False))
