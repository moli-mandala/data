"""Apply recorded original-page decisions without discarding any text candidate."""
import json,hashlib,copy
from pathlib import Path
P=Path(__file__).resolve().parent

def build():
    review=json.loads((P/'appendix5-original-review-20260927.json').read_text())
    sources=['appendix5-text-first-reading.jsonl','appendix5-interlinear-column-candidates.jsonl']
    rows=[json.loads(s) for f in sources for s in (P/f).open()]
    decisions={r['source_unit_key']:r for r in review['decisions']}
    pages={r['printed_page'] for r in review['pages']}
    assert set(decisions)<= {r['source_unit_key'] for r in rows}
    for row in rows:
        needed={l['printed_page'] for l in row['source_lines']} if row.get('kind')=='whole-expression' else {row['printed_page']}
        if needed<=pages:
            row['status']='Owner original-page review passed; independent review and metadata integration pending'
            row['owner_review_pages']=sorted(needed)
            if 'needs_manual_alignment' in row:
                row['candidate_alignment_warning']=row['needs_manual_alignment']
                row['needs_manual_alignment']=False
        key=row['source_unit_key']
        if key in decisions:
            change=decisions[key];row['review_input']=copy.deepcopy(row)
            row.update(change.get('set',{}));row['review_decision']=change['decision']
    return rows

if __name__=='__main__':
    rows=build();(P/'appendix5-reviewed-records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(len(rows),'total records;',sum('owner_review_pages' in r for r in rows),'owner reviewed so far; rest explicitly unreviewed')
