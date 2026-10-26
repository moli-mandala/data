"""Install only the independently approved, reproducible whole-source freeze."""
import argparse
import csv
import json
import shutil
from pathlib import Path
from verify_reviewed_regeneration import verify_freeze, main as regenerate
from freeze_reviewed_for_audit import sha

P=Path(__file__).resolve().parent
D=P.parents[4]
STEM='whole-reviewed-preview-20260927'
CSV=D/'data/other/forms/20260925-bailey-north-jubbal.csv'
PROFILE=D/'conversion/bailey-north-jubbal-1920.txt'
ARCHIVE=P/'historical-glossary-installation'


def generate():
    freeze=verify_freeze()
    regeneration=json.loads((P/'whole-reviewed-regeneration-validation.json').read_text())
    assert regeneration['freeze_sha256']==sha(P/'whole-reviewed-output-freeze-20260927.json')
    assert regeneration['producer_sha256']==sha(P/'verify_reviewed_regeneration.py')
    for name,digest in regeneration['compared'].items():assert sha(P/name)==digest,name
    with (P/f'{STEM}.csv').open(newline='') as stream:rows=list(csv.reader(stream))
    audit=[json.loads(s) for s in (P/f'{STEM}-audit.jsonl').read_text().splitlines()]
    assert len(rows)==837 and len(audit)==688
    assert len({r[10] for r in rows})==837
    assert {k for a in audit for k in a['entry_keys']}=={r[10] for r in rows}
    return rows,audit


def main():
    args=argparse.ArgumentParser();args.add_argument('--install',action='store_true');options=args.parse_args()
    if options.install:regenerate()
    rows,audit=generate()
    if not options.install:
        print('Approved 837-row source freeze verified; no writes.');return
    ARCHIVE.mkdir(exist_ok=True)
    historical=[CSV,CSV.with_suffix('.yaml'),PROFILE,P/'audit.jsonl']
    if not (ARCHIVE/'manifest.json').exists():
        old=list(csv.reader(CSV.open(newline='')))
        assert len(old)==330
        assert {r[10] for r in old}<={r[10] for r in rows}
        manifest={}
        for source in historical:
            target=ARCHIVE/source.name
            shutil.copyfile(source,target)
            manifest[str(source.relative_to(D))]={'archived':target.name,'sha256':sha(source)}
        (ARCHIVE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    else:
        for info in json.loads((ARCHIVE/'manifest.json').read_text()).values():
            assert sha(ARCHIVE/info['archived'])==info['sha256']
    for source,target in [(P/f'{STEM}.csv',CSV),(P/f'{STEM}-audit.jsonl',P/'audit.jsonl'),(P/f'{STEM}-profile.txt',PROFILE)]:
        shutil.copyfile(source,target);assert sha(source)==sha(target)
    report=dict(status='approved source-stage files installed; metadata and installed validation recorded separately',
                rows=837,audit_units=688,preserved_legacy_keys=330,new_rows=507,
                freeze_sha256=sha(P/'whole-reviewed-output-freeze-20260927.json'),
                accepted_audit_sha256=sha(P/'root-whole-output-audit-20260927-pass1.json'),
                installed_sha256={str(x.relative_to(D)):sha(x) for x in [CSV,P/'audit.jsonl',PROFILE]},
                historical_importer='import_source.py is hash-pinned historical evidence. Never run it against the live installation; regeneration verifier runs it only in an isolated sandbox.',
                deferred=['Full CLDF/database build','Compiled graph/durable-ID verification','Regenerated references','Full suite','Browser QA'])
    (P/'whole-installation-result-20260927.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Installed 837 rows / 688 audit units; all330 legacy keys retained.')


if __name__=='__main__':main()
