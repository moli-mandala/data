"""Root-selected reproducible whole-output audit sample; not an audit verdict."""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

BASE=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    pass2='--pass2' in sys.argv
    freeze=BASE/('whole-paired-output-freeze-20260927-pass2.json' if pass2 else 'whole-paired-output-freeze-20260927.json')
    manifest=json.loads(freeze.read_text())
    for filename,sha in manifest['input_sha256'].items():
        assert digest(BASE/filename)==sha,filename
    packages={}
    for name,folder in [('rampur',BASE),('baghi',BASE.parent/'bailey_baghi_1920')]:
        for filename,sha in manifest['packages'][name]['files'].items():
            assert digest(folder/filename)==sha,(name,filename)
        audit=[json.loads(line) for line in (folder/'whole-proposed-audit.jsonl').read_text().splitlines()]
        with (folder/'whole-proposed.csv').open(newline='') as stream:
            csv_rows=list(csv.reader(stream))
        packages[name]={'audit':audit,'csv_rows':csv_rows}
    root=Path.cwd()
    assert digest(root/manifest['source_pdf'])==manifest['source_pdf_sha256']
    rng=random.Random('root-independent-paired-output-20260927-v2' if pass2 else 'root-independent-paired-output-20260927-v1')
    prior_indices=set()
    if pass2:
        prior=json.loads((BASE/'root-frozen-sample-20260927.json').read_text())
        prior_indices={u['audit_index'] for u in prior['selected_units']}
    audit=packages['rampur']['audit']
    selected=[]
    for kind,count in [('glossary',4),('grammar',8),('sentence',3),('introduction',3),('commentary',2)]:
        pool=[i for i,row in enumerate(audit) if i not in prior_indices and row.get('source_kind','glossary')==kind
              and (row.get('exported_entry_keys') or row.get('entry_keys') or packages['baghi']['audit'][i].get('exported_entry_keys'))]
        selected.extend(rng.sample(pool,count))
    records=[]
    for index in selected:
        paired={}
        for name,package in packages.items():
            unit=package['audit'][index]
            keys=unit.get('exported_entry_keys',unit.get('entry_keys',[]))
            # Jambu raw forms CSV is headerless; stable entry key is column 11.
            rows=[r for r in package['csv_rows'] if r[10] in keys]
            assert len(rows)==len(keys),(name,index,keys)
            paired[name]={'audit_unit':unit,'actual_csv_rows':rows}
        records.append({'audit_index':index,'printed_page':audit[index]['printed_page'],
                        'source_kind':audit[index].get('source_kind','glossary'),
                        'packages':paired,'verdict':'pending original-image review'})
    report={'status':'selected and hash-verified; no audit verdict yet',
            'freeze_sha256':digest(freeze),'source_pdf_sha256':manifest['source_pdf_sha256'],
            'method':'Seeded stratified 20 source units across glossary, grammar, sentences, introduction, commentary; inspect both lect outputs.',
            'selected_units':records}
    output='root-frozen-sample-20260927-pass2.json' if pass2 else 'root-frozen-sample-20260927.json'
    (BASE/output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for r in records:
        print(r['audit_index'],r['printed_page'],r['source_kind'],
              [p['audit_unit'].get('source_unit_key',p['audit_unit'].get('source_cell_key')) for p in r['packages'].values()])


if __name__=='__main__':
    main()
