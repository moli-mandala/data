"""Reproduce and install the independently reviewed whole Dhakal 2011 thesis.

No database build occurs. Historical Chapter 3 code/evidence remains intact in
import_chapter3_legacy.py; immutable old data drives identity preservation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
STEM='20260921-dhakal-darai'
LITERAL_STEM='20260927-dhakal-darai-literal'
SOURCE='dhakal2011darai'
ROUTES={'author-transcription':STEM,'literal-preservation':LITERAL_STEM}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_approval():
    approval=json.loads((HERE/'root-whole-independent-review-20260927206.json').read_text())
    assert approval['result']=='pass' and approval['reviewed']==20 and approval['material_errors']==0
    for name,expected in approval['verified_output_hashes'].items():
        assert digest(HERE/name)==expected, ('Reviewed output changed',name)
    return approval


def build():
    verify_approval()
    rows=[]
    for route in ROUTES:
        with (HERE/f'whole-proposed-{route}.csv').open(newline='') as stream:
            rows.extend(csv.reader(stream))
    audit=[json.loads(line) for line in (HERE/'whole-proposed-audit.jsonl').open()]
    assert len(rows)==8373 and len(audit)==11304
    assert len({row[10] for row in rows})==len(rows)
    return rows,audit


def generate():
    # All inputs are reviewed source-local artifacts; no original PDF, network,
    # live installed CSV or compiled table is needed to reproduce the export.
    for script in ['prepare_whole_metadata.py','expand_whole_notation.py',
                   'prepare_whole_transcription.py','prepare_whole_csv_preview.py',
                   'prepare_whole_reuse_candidates.py','prepare_whole_staging.py']:
        subprocess.run([sys.executable,str(HERE/script)],check=True)
    verify_approval()
    return build()


def install():
    from pybtex.database import parse_file
    generate()
    bib_path=ROOT/'cldf/sources.bib'
    text=bib_path.read_text()
    blocks=re.split(r'(?=^@)',text,flags=re.M)
    matches=[i for i,b in enumerate(blocks) if re.match(r'@\w+\{'+SOURCE+r'\s*,',b)]
    assert len(matches)==1
    replacement=(HERE/'whole-source-proposed.bib').read_text().rstrip()+'\n\n'
    assert set(parse_file(str(HERE/'whole-source-proposed.bib')).entries)=={SOURCE}
    old_other={k:v for k,v in parse_file(str(bib_path)).entries.items() if k!=SOURCE}
    blocks[matches[0]]=replacement
    dialect_path=ROOT/'cldf/dialects.csv'
    with dialect_path.open(newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;existing={r['ID']:r for r in reader}
    with (HERE/'whole-dialects-proposed.csv').open(newline='') as stream:
        additions=list(csv.DictReader(stream))
    for row in additions:
        assert set(row)==set(fields)
        assert row['ID'] not in existing or existing[row['ID']]==row
    identity_path=ROOT/'data/form-identities.csv'
    identity_before=digest(identity_path)
    canonical={}
    for route,stem in ROUTES.items():
        for extension,source in [('csv',f'whole-proposed-{route}.csv'),('yaml',f'whole-installed-{route}.yaml')]:
            target=ROOT/f'data/other/forms/{stem}.{extension}'
            shutil.copyfile(HERE/source,target)
            canonical[str(target.relative_to(ROOT))]=digest(target)
    profile=ROOT/'conversion/dhakal-darai.txt'
    shutil.copyfile(HERE/'whole-profile-proposed.txt',profile)
    canonical[str(profile.relative_to(ROOT))]=digest(profile)
    shutil.copyfile(HERE/'whole-proposed-audit.jsonl',HERE/'whole-installed-audit.jsonl')
    bib_path.write_text(''.join(blocks))
    assert {k:v for k,v in parse_file(str(bib_path)).entries.items() if k!=SOURCE}==old_other
    with dialect_path.open('a',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writerows(row for row in additions if row['ID'] not in existing)
    assert digest(identity_path)==identity_before
    report={'status':'Complete reviewed source-stage installed; compiled/database/browser gates deferred.',
            'rows':8373,'author_transcription':8361,'literal_preservation':12,
            'audit_units':11304,'exact_reuses':2431,'controls':500,'old_keys_preserved':485,
            'canonical_sha256':canonical,'installed_audit_sha256':digest(HERE/'whole-installed-audit.jsonl'),
            'durable_identity_sha256_unchanged':identity_before,
            'independent_review':'root-whole-independent-review-20260927206.json',
            'unrelated_bibliography_entries_unchanged':len(old_other),
            'dialect_records_added':[row['ID'] for row in additions if row['ID'] not in existing],
            'deferred':['Full CLDF/database build','Compiled identity/graph survival','Full test suite','Browser QA']}
    (HERE/'whole-installation-report-20260927.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--install',action='store_true');args=parser.parse_args()
    install() if args.install else generate()
