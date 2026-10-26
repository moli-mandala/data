"""One source-evidenced uncertainty propagation addendum; no canonical writes."""
import csv,json,hashlib,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;F=P/'whole-proposal-r2';O=P/'uncertainty-addendum'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((F/'review-manifest.json').read_text())
    for name,digest in manifest['output_sha256'].items():assert sha(F/name)==digest,name
    rows=list(csv.reader((F/'proposal.csv').open()));audit=[json.loads(s) for s in(F/'proposal-audit.jsonl').read_text().splitlines()];keyed={r[10]:r for r in rows}
    found=[]
    def walk(value,path):
        if isinstance(value,dict):
            for k,v in value.items():
                if ('uncert' in k.lower() or k=='review_flags') and v:found.append({'path':path+[k],'value':v})
                walk(v,path+[k])
        elif isinstance(value,list):
            for i,v in enumerate(value):walk(v,path+[str(i)])
    for i,unit in enumerate(audit):walk(unit,[str(i)])
    substantive=[f for f in found if f['path'][-1]=='source_glyph_uncertainty']
    assert len(substantive)==1
    key='bailey1920kagani:p97:substantive-past:3-pl-f';row=keyed[key];before=row[:]
    assert row[2]=='äsīā̃' and 'uncertain' not in row[14].split()
    note='Source glyph uncertain: the initial upper mark shows two separated points, provisionally transcribed ä; damaged ā remains possible (alternative āsīā̃). No phonetic interpretation is assigned.'
    row[6]=row[6]+' '+note;row[14]+=' uncertain'
    unit=next(u for u in audit if u.get('source_unit_key')==key)
    unit['uncertainty_propagation_addendum']={'entry_key':key,'source_field':'input.source_glyph_uncertainty','exported_notes':row[6],'exported_tags':row[14],'reason':'Carry existing unresolved source reading into visible structured uncertainty; no new reading or interpretation.'}
    O.mkdir(exist_ok=True)
    with (O/'proposal.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
    (O/'proposal-audit.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in audit))
    shutil.copyfile(F/'profile-proposal.txt',O/'profile-proposal.txt')
    report={'status':'pending root targeted approval; no canonical writes','base_freeze_sha256':sha(F/'review-manifest.json'),'base_output_sha256':{n:sha(F/n) for n in ['proposal.csv','proposal-audit.jsonl','profile-proposal.txt']},'census':{'all_audit_units':len(audit),'all_rows':len(rows),'recursive_nonempty_uncertainty_fields':found,'affected_rows':1},'entry_key':key,'before_row':before,'after_row':row,'changed_fields':['Notes','Tags'],'all_keys_forms_glosses_unchanged':True,'source_evidence':unit['input']['source_glyph_uncertainty'],'output_sha256':{n:sha(O/n) for n in ['proposal.csv','proposal-audit.jsonl','profile-proposal.txt']},'producer_sha256':sha(Path(__file__))}
    (O/'review-proposal.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'affected_rows':1,'entry_key':key,'output_sha256':report['output_sha256']},ensure_ascii=False))
if __name__=='__main__':main()
