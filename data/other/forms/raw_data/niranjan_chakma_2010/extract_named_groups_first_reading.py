"""Reproduce all145 source cells; do not infer canonical dialects or variants."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TSV = HERE / 'named-groups-manual-first-reading.tsv'
OUT = HERE / 'named-groups-first-reading.jsonl'
REPORT = HERE / 'named-groups-first-reading-validation.json'
CENSUS = HERE / 'independent-structure-pages60-74-20260927.json'
PDF = HERE.parents[5] / 'tmp/pdfs/chakma-discovery/niranjan-2010.pdf'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def build():
    census=json.loads(CENSUS.read_text())
    assert sha(PDF)==census['source']['sha256']
    records=list(csv.DictReader(TSV.open(newline=''),delimiter='\t'))
    assert len(records)==145
    rows=[]; counts=Counter(); groupcounts=Counter(); groups=defaultdict(list)
    for r in records:
        p,g,c=map(int,(r['page'],r['group'],r['cell']))
        form=r['form']; scope=r['group_label']; note=r['note']
        if g==64:
            form=form.replace('; ','\n')
            note='One physical cell printed across two lines; six comma-listed members, including repeated forms, retained exactly. No invented semicolon in target.'
        assert form and scope and r['definition_bn']
        assert '\ufffd' not in ''.join(r.values())
        counts[p]+=1;groupcounts[p,g]+=1;groups[g].append(c)
        types=[]
        if any(s in scope for s in ('অন্যান্য','প্রায়','ইত্যাদি','কোন কোন')):types.append('qualified-collective-scope')
        if any(s in scope for s in ('রাজ্য','গ্রাম','অঞ্চল','রাঙ্গামাটি')):types.append('geographic-scope')
        if scope=='—':types.append('unresolved-printed-dash')
        if not types:types.append('named-group-label; identity-not-yet-registered')
        rows.append({
            'source_unit_key':f'niranjanchakma2010:p{p}:named-groups:g{g}:c{c}',
            'printed_page':p,'pdf_page':p+2,'printed_group':g,'group_cell':c,
            'target_form':form,'target_printed_lines':form.splitlines(),
            'source_group_label':scope,'scope_types':types,
            'definition_bn':r['definition_bn'],'definition_language':'Bengali',
            'definition_scope':'shared by all target cells in this printed numbered group',
            'reading_note':note or None,
            'canonical_language_id':None,'canonical_dialect_id':None,
            'relation_policy':'No inferred variant links, canonical dialects, or homograph merges.',
            'review_status':'owner original-page first reading; independent fine-reading review pending',
        })
    assert set(groups)==set(range(1,71))
    for g,cs in groups.items():assert cs==list(range(1,len(cs)+1)),g
    for p in census['named_group_pronunciation']['pages']:
        assert counts[p['printed_page']]==p['physical_cells']
        assert {str(g):n for (pp,g),n in groupcounts.items() if pp==p['printed_page']}==p['cells_by_group']
    assert len({r['source_unit_key'] for r in rows})==145
    for g in groups:
        assert len({r['definition_bn'] for r in rows if r['printed_group']==g})==1
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    report={
        'status':'complete source-only first-reading coverage, not canonical readiness',
        'physical_cells':145,'numbered_groups':70,'per_page_cells':{str(p):n for p,n in sorted(counts.items())},
        'input_sha256':{'pdf':sha(PDF),TSV.name:sha(TSV),CENSUS.name:sha(CENSUS)},
        'output_sha256':hashlib.sha256(payload.encode()).hexdigest(),
        'source_labels':sorted({r['source_group_label'] for r in rows}),
        'noted_cells':[r['source_unit_key'] for r in rows if r['reading_note']],
        'deferred':['Individual fine-glyph resolution','Group/social/regional identity reconciliation','Independent literal review','Whole-book integration; no source-stage installation'],
        'policy':'All145 physicalcells retained. Shared Bengali definitions untransformed. Six members of group64 stay in one cell with printed line break; no invented English or IPA. Qualified collective labels are not pseudolects.'}
    return payload,report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    payload,report=build()
    if args.check:
        assert OUT.read_text()==payload
        assert json.loads(REPORT.read_text())==report
    else:
        OUT.write_text(payload)
        REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'cells':145,'groups':70,'sha256':report['output_sha256']}))
