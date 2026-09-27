"""Prepare audited Belari source rows, without database builds."""
import argparse
import csv
import json
import unicodedata
import shutil
import yaml
from pathlib import Path
from prepare_analysis import prepare, ROOT

SOURCE='bhat1971koraga'
STEM='20260922-bhat-belari'

def build():
    candidates={r['entry_key']:r for r in map(json.loads,(ROOT/'lexical-candidates.jsonl').read_text().splitlines())}
    reviews={r['entry_key']:r for r in map(json.loads,(ROOT/'glyph-review.jsonl').read_text().splitlines())}
    relations=json.loads((ROOT/'relationship-decisions.json').read_text())
    parents={r['entry_key']:r['parent_key'] for r in relations['relationships']}
    rows=[];audit=[]
    for analysis in prepare():
        origin=candidates[analysis['source_occurrence_key']]
        assert analysis['glyph_reviewed']
        section=origin['section'];etymology=''
        if section=='4':
            etymology='Bhat compares this Belari form with Tulu under an initial Tulu p / Belari h correspondence. The preceding discussion leaves borrowing versus native inheritance open.'
        elif section=='5':
            etymology='Bhat cites this form as an example of Proto-Dravidian *ṟ corresponding to Belari ḷ; no reconstructed lexical etymon is supplied.'
        locator=f'p. {origin["printed_page"]}, section {section}, col. {origin["column"]}, item {origin["row_in_section_column"]}'
        row=['Belari','',unicodedata.normalize('NFC',analysis['form']),analysis['gloss'],'','',
             '\n'.join(analysis['notes']),f'{SOURCE}[{locator}]','',etymology,analysis['entry_key'],
             parents.get(analysis['entry_key'],''),'','',' '.join(analysis['tags'])]
        rows.append(row)
        audit.append({'entry_key':analysis['entry_key'],'source_occurrence':origin,
                      'glyph_review':reviews[analysis['source_occurrence_key']],
                      'analysis':analysis,'row':row,'status':'proposed-source-row'})
    keys={r[10] for r in rows}
    assert len(rows)==len(keys)==192
    assert all(not r[11] or r[11] in keys for r in rows)
    return rows,audit

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);parser.add_argument('--install',action='store_true');args=parser.parse_args()
    if not args.output and not args.install: parser.error('provide --output or --install')
    args.output=args.output or ROOT
    rows,audit=build();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/f'{STEM}.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
    (args.output/'audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
    if args.install:
        acceptance=json.loads((ROOT/'output-audit-2026092203.json').read_text())
        assert acceptance['status']=='accepted source-output sample' and acceptance['material_errors']==0
        shutil.copyfile(args.output/f'{STEM}.csv',ROOT.parent.parent/f'{STEM}.csv')
        # The shared citation key is owned by selected-koraga.yaml, not duplicated here.
        settings={'file':f'{STEM}.csv','defaults':{'identity':{'legacy_ids':'stem','append_order':38},'transcription':{'profile':'bhat-belari'},'importer':{'commands':[['data/other/forms/raw_data/bhat_belari_1971/import_source.py','--install']],'note':'Source-file installation only; does not build CLDF or the browser database.'}}}
        (ROOT.parent.parent/f'{STEM}.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
        shutil.copyfile(ROOT/'bhat-belari.txt',ROOT.parents[4]/'conversion/bhat-belari.txt')
    print(f'{len(rows)} rows, {sum(bool(r[11]) for r in rows)} explicit paradigm links; installed={args.install}; no database build')
