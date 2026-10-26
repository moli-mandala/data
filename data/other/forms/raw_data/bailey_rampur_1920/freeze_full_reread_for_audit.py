"""Pin full-reread artifacts and select a third fresh paired output sample."""
import csv
import hashlib
import json
import random
from pathlib import Path

P=Path(__file__).resolve().parent


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def main():
    prior_manifest=P/'whole-paired-output-freeze-20260927-pass2.json'
    prior=json.loads(prior_manifest.read_text())
    packages={};manifest={}
    for name in ['rampur','baghi']:
        folder=P.parent/f'bailey_{name}_1920'
        filenames=['full-reread-preview.csv','full-reread-preview-audit.jsonl','full-reread-profile.txt',
                   'full-reread-preview-report.json','full-reread-parser-validation.json',
                   'whole-proposed.csv','whole-proposed-audit.jsonl']
        manifest[name]={filename:sha(folder/filename) for filename in filenames}
        with (folder/'full-reread-preview.csv').open(newline='') as stream:rows=list(csv.reader(stream))
        audit=[json.loads(line) for line in (folder/'full-reread-preview-audit.jsonl').read_text().splitlines()]
        packages[name]=(rows,audit)
    assert sha(Path.cwd()/prior['source_pdf'])==prior['source_pdf_sha256']
    frozen=dict(status='frozen for independent review only; installation not approved',
                source_pdf=prior['source_pdf'],source_pdf_sha256=prior['source_pdf_sha256'],
                prior_freeze_sha256=sha(prior_manifest),packages=manifest,
                input_sha256={name:sha(P/name) for name in ['whole-legacy-full-rereading-20260927.json',
                             'prepare_full_reread_preview.py','validate_full_reread_preview.py',
                             'freeze_full_reread_for_audit.py']})
    freeze=P/'whole-paired-full-reread-freeze-20260927.json'
    freeze.write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n')
    previous=set()
    for filename in ['root-frozen-sample-20260927.json','root-frozen-sample-20260927-pass2.json']:
        previous.update(r['audit_index'] for r in json.loads((P/filename).read_text())['selected_units'])
    rng=random.Random('root-independent-paired-output-20260927-v3')
    chosen=[];audit=packages['rampur'][1]
    for kind,count in [('glossary',4),('grammar',8),('sentence',3),('introduction',3),('commentary',2)]:
        pool=[i for i,row in enumerate(audit) if i not in previous and row.get('source_kind','glossary')==kind
              and (row.get('exported_entry_keys') or row.get('entry_keys') or packages['baghi'][1][i].get('exported_entry_keys'))]
        chosen.extend(rng.sample(pool,count))
    records=[]
    for index in chosen:
        paired={}
        for name,(rows,units) in packages.items():
            unit=units[index];keys=unit.get('exported_entry_keys',unit.get('entry_keys',[]))
            selected=[row for row in rows if row[10] in keys]
            assert len(selected)==len(keys)
            paired[name]=dict(audit_unit=unit,actual_csv_rows=selected)
        records.append(dict(audit_index=index,printed_page=audit[index]['printed_page'],
                            source_kind=audit[index].get('source_kind','glossary'),packages=paired,
                            verdict='pending original-image review'))
    report=dict(status='fresh sample selected; no audit verdict',freeze_sha256=sha(freeze),
                excluded_prior_indices=sorted(previous),selected_units=records)
    (P/'root-frozen-sample-20260927-pass3.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Frozen1345 rows; selected20freshpairedunits excluding both earlier samples:')
    for row in records:print(row['audit_index'],row['printed_page'],row['source_kind'])


if __name__=='__main__':main()
