"""Exact two source-defined suffix Notes repairs; frozen r5 remains historical."""
import copy,csv,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
OLD=HERE/'revision5'
NEW=HERE/'revision6'
DECISIONS={
 'grierson1906kolamibasim:p562:noun-plural:1:independent:w1':('l','Grierson identifies l as the plural suffix.'),
 'grierson1906kolamibasim:p562:accusative-dative:1:independent:w1':('n','Grierson identifies n as the accusative and dative suffix.'),
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert sha(OLD/'combined-review-candidate.csv')=='71e2988edaede8e570f46260e879b46c9563acdcfe7f8e4102fd9c5debe70259'
    frozen=json.loads((OLD/'review-freeze.json').read_text())
    for p,h in frozen['files'].items():assert sha(PACKAGE/p)==h,p
    with (OLD/'combined-review-candidate.csv').open() as f:before=list(csv.reader(f))
    after=copy.deepcopy(before);by={r[10]:r for r in after};delta=[]
    for key,(form,note) in DECISIONS.items():
        r=by[key];assert r[2]==form and r[6]=='No invented leading hyphen'
        delta.append({'entry_key':key,'field':'Notes','old':r[6],'new':note,'form_guard':form,'source_page':562,
          'source_evidence':'Printed prose explicitly identifies the suffix function; no leading hyphen is supplied to the literal form.'})
        r[6]=note
    assert len(after)==827 and sum(a!=b for a,b in zip(before,after))==2
    assert all(a[:6]+a[7:]==b[:6]+b[7:] for a,b in zip(before,after))
    NEW.mkdir(exist_ok=True)
    for name in ['grierson-kolami-basim-1906.txt','unicode-inventory.json','combined-review-candidate.yaml','specimen-metadata-proposal.jsonl','references-review-candidate.bib']:
        shutil.copyfile(OLD/name,NEW/name)
    with (NEW/'combined-review-candidate.csv').open('w',newline='') as f:csv.writer(f).writerows(after)
    audit=[json.loads(s) for s in (OLD/'combined-review-audit.jsonl').read_text().splitlines()]
    for a in audit:
        if a['source_unit_key'] in DECISIONS:
            assert a['actual_row']==next(r for r in before if r[10]==a['source_unit_key'])
            a['historical_pre_scholarly_notes_row']=a['actual_row'][:]
            a['actual_row']=by[a['source_unit_key']]
            a['scholarly_notes_revision']=next(d for d in delta if d['entry_key']==a['source_unit_key'])
    (NEW/'combined-review-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
    sample=json.loads((OLD/'fresh-output-selection-20260927.json').read_text())
    sample['historical_r5_selection_sha256']=sha(OLD/'fresh-output-selection-20260927.json')
    sample['candidate_sha256']=sha(NEW/'combined-review-candidate.csv');sample['audit_sha256']=sha(NEW/'combined-review-audit.jsonl')
    for field in ['sample','targeted_controls']:
        for item in sample[field]:item['actual_row']=by[item['source_unit_key']]
    for item in sample['literal_repair_regression_controls']:item['actual_emitted_row']=by.get(item['source_unit_key'])
    sample['revision_note']='Same independently accepted20 source keys; exactly two scholarly suffix Notes replacements, all other row fields unchanged.'
    (NEW/'fresh-output-selection-20260927.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
    report={'status':'Exact two Notes-only source-local derivative; installation remains disabled',
      'old_csv_sha256':sha(OLD/'combined-review-candidate.csv'),'new_csv_sha256':sha(NEW/'combined-review-candidate.csv'),
      'old_audit_sha256':sha(OLD/'combined-review-audit.jsonl'),'new_audit_sha256':sha(NEW/'combined-review-audit.jsonl'),
      'rows':827,'changed_notes':2,'all_other_fields_and_order_unchanged':True,'decisions':delta,'canonical_changes':False}
    (NEW/'scholarly-notes-continuity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    status=json.loads((OLD/'status.json').read_text());status['outputs']={p.name:sha(p) for p in NEW.iterdir() if p.is_file()}
    (NEW/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
