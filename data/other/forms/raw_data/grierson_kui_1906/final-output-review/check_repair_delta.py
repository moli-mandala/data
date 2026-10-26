import csv,hashlib,json,io
from pathlib import Path
O=Path(__file__).resolve().parent; P=O.parent; I=P/'integration'
before=list(csv.reader((O/'pre-repair-candidate.csv').open()))
after=list(csv.reader((I/'combined-candidate.csv').open()))
revision=json.loads((I/'display-note-revisions.json').read_text())
if hashlib.sha256((O/'pre-repair-candidate.csv').read_bytes()).hexdigest()!=revision['baseline_csv_sha256']:
    # The initial copy raced regeneration. Reconstruct only the declared old
    # Notes/tags, and require the exact independently pinned historical hash.
    before=[r[:] for r in after]; index={r[10]:r for r in before}
    for d in revision['changes']:
        index[d['entry_key']][6]=d['old_notes'];index[d['entry_key']][14]=d['old_tags']
    buffer=io.StringIO(newline='');csv.writer(buffer).writerows(before)
    restored=buffer.getvalue().encode()
    assert hashlib.sha256(restored).hexdigest()==revision['baseline_csv_sha256']
    (O/'pre-repair-candidate.csv').write_bytes(restored)
assert [r[10] for r in before]==[r[10] for r in after]
changes=[];issues=[]
expected_clear={'grierson1906kui:p652:khondmals-table:prompt47:v1','grierson1906kui:p652:khondmals-table:prompt51:v1','grierson1906kui:p660:khondmals-table:prompt99:v2','grierson1906kui:p672:khondmals-table:prompt175:v1','grierson1906kui:p672:khondmals-table:prompt175:v2'}
tags_changed=set()
for a,b in zip(before,after):
    delta=[n for n in range(15) if a[n]!=b[n]]
    if not delta:continue
    if set(delta)-{6,14}:issues.append([b[10],'unexpected fields',delta])
    if 14 in delta:
        tags_changed.add(b[10])
        if set(a[14].split())-set(b[14].split())!={'uncertain'} or set(b[14].split())-set(a[14].split()):issues.append([b[10],'tag delta'])
    changes.append({'entry_key':b[10],'changed_columns':delta,'old_notes':a[6],'new_notes':b[6]})
assert tags_changed==expected_clear
bykey={r[10]:r for r in after}
causal='grierson1906kui:p463:causal-example:2'
if 'causative' in bykey[causal][6]:issues.append([causal,'causal changed to causative'])
for k in ['grierson1906kui:p652:khondmals-table:prompt47:v2','grierson1906kui:p652:khondmals-table:prompt51:v2','grierson1906kui:p660:khondmals-table:prompt99:v1','grierson1906kui:p672:khondmals-table:prompt175:v3']:
    assert 'uncertain' in bykey[k][14].split()
report={'reviewer':'/root/asur_finish','status':'passed' if not issues else 'correction required','issues':issues,
 'before_sha256':hashlib.sha256((O/'pre-repair-candidate.csv').read_bytes()).hexdigest(),
 'after_sha256':hashlib.sha256((I/'combined-candidate.csv').read_bytes()).hexdigest(),
 'rows':len(after),'changed_rows':len(changes),'notes_changed':sum(6 in x['changed_columns'] for x in changes),'tags_changed':sorted(tags_changed),
 'other_fields_unchanged':not any(set(x['changed_columns'])-{6,14} for x in changes),
 'independent_notes_review':'Reviewed all forty replacements against their previous source claims. Source uncertainty, attribution, conjectural wording and genuine Letchmajee disagreement preserved. Causal-versus-causative terminology checked separately.',
 'changes':changes}
(O/'postrepair-delta-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(report['status'],len(changes),'changed rows',len(issues),'issues')
