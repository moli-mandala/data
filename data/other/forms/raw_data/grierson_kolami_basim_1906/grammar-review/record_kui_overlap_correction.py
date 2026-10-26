"""Propose an exact existing-key spelling repair without editing Kui assets."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'data/source_meta.py').exists())
KEY='grierson1906kui:p566:outside-comparand:06'

def main():
    path=ROOT/'data/data/other/forms/20260927-grierson-kui.csv'
    with path.open(newline='') as handle:rows=list(csv.reader(handle))
    found=[(i,r) for i,r in enumerate(rows,1) if r[10]==KEY]
    assert len(found)==1
    line,row=found[0]
    assert len(row)==15 and row[0]=='Kolami' and row[2]=='and'
    proposed=list(row);proposed[2]='amd'
    evidence=json.loads((HERE.parent/'scope/grammar-control-locators.json').read_text())
    crop=next(r for r in evidence if r['crop'].endswith('p566-kolami-amd-avnd-owner-control.png'))
    status=dict(status='Exact old-key correction proposal; no canonical mutation or activated importer change',
        source_csv=str(path.relative_to(ROOT)),source_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        row_number=line,entry_key=KEY,old_row=row,proposed_row=proposed,
        changed_fields=[dict(column=2,field='Form / literal transcription',old='and',new='amd')],
        evidence=crop,adjudication='Printed m has two shoulders; source hypothetical avnd retains plain n. Same physical occurrence as new chapter inventory, different old transcription.',
        new_chapter_witness_key='grierson1906kolamibasim:p566:pronunciation-n-nd-d:1:independent:w10',
        accounting='Same-occurrence reuse pending reviewed canonical spelling addendum, never exact-spelling reuse at current baseline; do not emit a duplicate chapter key.',
        importer_recovery_plan={'required':'A separately approved Kui correction addendum routed through its existing guarded coordinator, preserving previous installer/frozen candidate history.',
            'guards':['Current15fieldoldrow exact','CurrentCSVhash','Rootapproval ofamdwith crop hash','Existingdurableentrykey unchanged',
                'All other1831rows and all14other fields unchanged','Repeat canonical import retains the approved correction instead of restoring historicaland'],
            'canonical_writes':False,'current_scripts_unchanged':True},accepted_for_installation=False)
    (HERE/'existing-kui-key-spelling-correction-proposal.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(status,ensure_ascii=False))

if __name__=='__main__':main()
