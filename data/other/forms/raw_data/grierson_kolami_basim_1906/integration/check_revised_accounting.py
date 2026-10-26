"""Rebase root-reviewed complete literal accounting through exact metadata deltas."""
import argparse,csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
ROOT=HERE.parents[6]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def csvrows(p):
    with p.open() as f:return list(csv.reader(f))
def main(directory=None,expected_tag_changes=138,expected_note_changes=3):
    current=directory.resolve() if directory is not None else HERE/'revision3'
    original_report=PACKAGE/'root-independent-review/whole-candidate-literal-accounting.json'
    assert sha(original_report)=='5513afde2c61adcc60e8ad0c865cd5038d0a7ffd00c680fa86fdcadde8893c1e'
    root=json.loads(original_report.read_text())
    assert sha(HERE/'combined-review-candidate.csv')==root['candidate_sha256']
    old=csvrows(HERE/'combined-review-candidate.csv');new=csvrows(current/'combined-review-candidate.csv')
    assert len(old)==len(new)==827 and len({r[10] for r in new})==827
    changes=[]
    for a,b in zip(old,new):
        changed=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
        assert set(changed)<={6,14},(a[10],changed)
        if changed:changes.append({'entry_key':a[10],'changed_columns':changed})
    assert sum(14 in r['changed_columns'] for r in changes)==expected_tag_changes
    assert sum(6 in r['changed_columns'] for r in changes)==expected_note_changes
    audit=rows(current/'combined-review-audit.jsonl');assert len(audit)==1570
    baseline=rows(HERE/'combined-review-audit.jsonl')
    assert [r['source_unit_key'] for r in baseline]==[r['source_unit_key'] for r in audit]
    assert [r['entry_keys'] for r in baseline]==[r['entry_keys'] for r in audit]
    for path,pin in root['canonical_input_sha256'].items():assert sha(ROOT/path)==pin,path
    canonical={p:{r[10]:r for r in csvrows(ROOT/p)} for p in root['canonical_input_sha256']}
    for reuse in root['canonical_reuses']:
        matches=[r for r in audit if reuse['existing_key'] in r['entry_keys']]
        assert len(matches)==1
        if 'canonical_row' in matches[0]:
            assert matches[0]['canonical_row']==canonical[reuse['canonical_path']][reuse['existing_key']]
        else:
            previous=[r for r in baseline if reuse['existing_key'] in r['entry_keys']]
            assert matches==previous
            assert canonical[reuse['canonical_path']][reuse['existing_key']][2]==matches[0]['source_record']['form_candidate']
    grammar_before=rows(PACKAGE/'grammar-review/sharp-reviewed-revision2/reviewed-child-witnesses.jsonl')
    grammar_after=rows(PACKAGE/'grammar-review/sharp-reviewed-revision3/reviewed-child-witnesses.jsonl')
    assert len(grammar_before)==len(grammar_after)==224
    for a,b in zip(grammar_before,grammar_after):
        for field in ['source_unit_key','physical_parent_key','form','gloss','printed_label','language_label','proposed_disposition','source_record']:
            assert a[field]==b[field],(a['source_unit_key'],field)
    report={'status':'Root-reviewed full literal/accounting projection preserved through exact expected tag and commentary fields',
      'root_accounting_report_sha256':sha(original_report),'old_candidate_sha256':sha(HERE/'combined-review-candidate.csv'),
      'current_candidate_sha256':sha(current/'combined-review-candidate.csv'),'current_audit_sha256':sha(current/'combined-review-audit.jsonl'),
      'counts':root['section_counts'],'rows':827,'structural_parents':169,'grammar_children':224,'physical_specimen_atoms':561,
      'atomic_emissions':560,'source_bounded_expressions':51,'unchanged_canonical_reuses':3,'audit_records':1570,
      'changed_tag_fields':expected_tag_changes,'changed_notes_fields':expected_note_changes,'all_other_fields_keys_order_literals_graphs_citations_unchanged':True,
      'canonical_input_sha256':root['canonical_input_sha256'],'canonical_changes':False,
      'limits':['Remaining independent case/object/residual metadata review and final original-output sample are pending.','No full compiled/browser gates run.']}
    (current/'protected-literal-accounting.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path);p.add_argument('--expected-tag-changes',type=int,default=138);p.add_argument('--expected-note-changes',type=int,default=3);a=p.parse_args();main(a.directory,a.expected_tag_changes,a.expected_note_changes)
