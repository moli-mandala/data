"""Read-only manual-source overlap inventory, not a compiled lexical total."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'data/source_meta.py').exists())
DATA=ROOT/'data'


def main():
    overlaps=[]
    for path in sorted((DATA/'data/other/forms').glob('*.csv')):
        matches=[]
        with path.open(newline='') as stream:
            for row in csv.reader(stream):
                if row and row[0] in ('Kolami','LSI-KOLAMI','Naikri'):
                    matches.append(dict(language_label=row[0],form=row[2],gloss=row[3],source=row[7],entry_key=row[10] if len(row)>10 else None))
        if matches:
            overlaps.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                count=len(matches),counts_by_label=dict(Counter(r['language_label'] for r in matches)),
                keys=[r['entry_key'] for r in matches],representative_rows=matches[:3]))
    held=[]
    source=DATA/'data/other/forms/raw_data/grierson_kui_1906/outside-comparands-routing-reuse-review-20260927.json'
    record=json.loads(source.read_text())
    for decision in record['decisions']:
        if decision.get('source_label')=='Bhili of Basim' and ':p566:' in decision['source_unit_key']:
            held.append(decision)
    registry=list(csv.DictReader((DATA/'cldf/languages.csv').open(newline='')))
    bases=[r for r in registry if r['ID'] in ('Kolami','Naikri','Naiki')]
    dialects=[r for r in csv.DictReader((DATA/'cldf/dialects.csv').open(newline='')) if r.get('Language_ID') in ('Kolami','Naikri')]
    report=dict(status='Read-only source-input overlap leads; not complete compiled language totals',manual_sources=overlaps,
        current_bases=bases,current_dialects=dialects,existing_kui_basim_identity_holds=held,
        additional_overlap='DEDR-origin rows and entry commentary remain a separate source layer; original Grierson physical witnesses are not identical merely because a form/gloss agrees.',
        reuse_policy='Only the exact same original occurrence may reuse an existing key/citation. Preserve other editions/transcriptions and independent attestation identities.',canonical_changes=False)
    (HERE/'overlap-inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(source_files=len(overlaps),rows=sum(r['count'] for r in overlaps),basim_kui_holds=len(held))))


if __name__=='__main__':main()
