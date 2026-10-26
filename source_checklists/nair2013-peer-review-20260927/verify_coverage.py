"""Read-only whole-book peer inventory; writes only separate review evidence."""
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path('/Users/aryamanarora/Documents/Code/jambu-all')
DATA=ROOT/'data'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(DATA))
import source_meta
from pybtex.database import parse_file
def sha(path):
    if not path.exists():return None
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def strings(value):
    if isinstance(value,str):return {value}
    if isinstance(value,dict):return set().union(*(strings(v) for v in value.values())) if value else set()
    if isinstance(value,list):return set().union(*(strings(v) for v in value)) if value else set()
    return set()
def keys(record):
    for field in ('entry_keys','proposed_entry_keys'):
        if field in record:return record[field]
    return [record['proposed_entry_key']] if record.get('proposed_entry_key') else []
generator=DATA/'data/other/forms/raw_data/nair2013_chapter_coverage.py'
tree=ast.parse(generator.read_text())
chapters=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHAPTERS' for t in n.targets))
pages=[p for _,start,end in chapters for p in range(start,end+1)]
assert pages==list(range(4,59)) and len(chapters)==13
meta=source_meta.load()
bib=parse_file(str(DATA/'cldf/sources.bib')).entries
language={r['ID']:r for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
target_languages={'adiya':'Ravula','aranadan':'Aranadan','betta-kurumba':'BettaKurumba','cholanaickan':'Cholanaickan','irula':'Irula','kakkala':'Kakkala','kani':'Kanikkaran','kurichiya':'Kurichiya','malamuthan':'Malamuthan','malaveda':'Malaveda','mannan':'Mannan','muduga':'Muduga','mullu-kurumba':'Mullu Kurumba'}
handoffs={
 'adiya': ['source_checklists/20260927-nair-adiya-source-stage.md'],
 'aranadan':['data/other/forms/raw_data/nair_aranadan_2013/source-stage-handoff.json'],
 'betta-kurumba':['data/other/forms/raw_data/nair_betta_kurumba_2013/SOURCE_STAGE_HANDOFF.md','data/other/forms/raw_data/nair_betta_kurumba_2013/metadata-amendment/HANDOFF.md'],
 'cholanaickan':['data/other/forms/raw_data/nair_cholanaickan_2013/SOURCE_STAGE_HANDOFF.md'],
 'irula':['data/other/forms/raw_data/nair_irula_2013/SOURCE_STAGE_HANDOFF.md','source_checklists/20260927-nair-irula-source-stage.md'],
 'kakkala':['source_checklists/20260927-nair-kakkala-source-stage.md'],
 'kani':['source_checklists/20260927-nair-kani-source-stage.md','data/other/forms/raw_data/nair_kani_2013/source-stage-handoff.json'],
 'kurichiya':['data/other/forms/raw_data/nair_kurichiya_2013/SOURCE_STAGE_HANDOFF.md'],
 'malamuthan':['data/other/forms/raw_data/nair_malamuthan_2013/SOURCE_STAGE_HANDOFF.md'],
 'malaveda':['data/other/forms/raw_data/nair_malaveda_2013/SOURCE_STAGE_HANDOFF.md','source_checklists/20260927-nair-malaveda-source-stage.md'],
 'mannan':['data/other/forms/raw_data/nair_mannan_2013/SOURCE_STAGE_HANDOFF.md','source_checklists/20260927-nair-mannan-source-stage.md'],
 'muduga':['data/other/forms/raw_data/nair_muduga_2013/SOURCE_STAGE_HANDOFF.md','source_checklists/20260927-nair-muduga-source-stage.md'],
 'mullu-kurumba':['data/other/forms/raw_data/nair_mullu_kurumba_2013/SOURCE_STAGE_HANDOFF.md','data/other/forms/raw_data/nair_mullu_kurumba_2013/canonical-source-stage-validation.json']}
results=[];issues=[];aggregate=collections.Counter();allrefs=set()
for chapter,start,end in chapters:
    stem='20260927-nair-'+chapter
    package=DATA/'data/other/forms/raw_data'/('nair_'+chapter.replace('-','_')+'_2013')
    csvpath=DATA/'data/other/forms'/(stem+'.csv');yamlpath=csvpath.with_suffix('.yaml');auditpath=package/'installed-audit.jsonl'
    item={'chapter':chapter,'printed_pages':[start,end],'installed':csvpath.exists() and yamlpath.exists() and auditpath.exists(),'handoff_evidence':{p:sha(DATA/p) for p in handoffs[chapter]}}
    if chapter=='mullu-kurumba' and (package/'SOURCE_STAGE_HANDOFF.md').exists():
        item['handoff_evidence']['data/other/forms/raw_data/nair_mullu_kurumba_2013/SOURCE_STAGE_HANDOFF.md']=sha(package/'SOURCE_STAGE_HANDOFF.md')
    assert all(item['handoff_evidence'].values())
    if not item['installed']:
        candidate=json.loads((package/'candidate-status.json').read_text())
        item.update(candidate_rows=candidate['rows'],candidate_status=candidate['status'],candidate_sha256=sha(package/'proposed.csv'),not_claimed='Canonical installation and installed-only validation')
        results.append(item);continue
    rows=list(csv.reader(csvpath.open(newline='')));by={r[10]:r for r in rows}
    assert len(by)==len(rows) and all(len(r)==15 for r in rows)
    assert all(r[0] in language for r in rows)
    langs=collections.Counter(r[0] for r in rows);aggregate.update(langs)
    audit=list(map(json.loads,auditpath.read_text().splitlines()))
    ak=[k for a in audit for k in keys(a)]
    assert set(ak)==set(by),(chapter,'audit key closure')
    assert all(a.get('accepted_for_installation',a.get('accepted_for_source_stage')) is True for a in audit)
    assert all(not r[11] or r[11] in by for r in rows)
    assert all(not r[13] or all(k in by for k in r[13].split('|')) for r in rows)
    refs={k for k,owner in meta.source_owner.items() if owner==stem};allrefs.update(refs)
    assert refs and refs<=bib.keys()
    assert all(meta.transcription(r[7].split('[',1)[0],str(csvpath),r[0])==(None,False) for r in rows)
    assert meta.file_flag(str(csvpath),'identity','dedupe_by_entry_key',False) is True
    receipts=[p for p in [package/'installation-receipt.json',package/'installation/installation-receipt.json',package/'metadata-amendment/receipt.json'] if p.exists()]
    assert receipts,(chapter,'installation receipt missing')
    pinned=set().union(*(strings(json.loads(p.read_text())) for p in receipts))
    for role,path in [('csv',csvpath),('yaml',yamlpath),('audit',auditpath)]:
        if sha(path) not in pinned:issues.append({'chapter':chapter,'type':'own-asset hash absent from receipt evidence','role':role,'actual_sha256':sha(path)})
    item.update(rows=len(rows),target_rows=langs[target_languages[chapter]],comparison_rows=len(rows)-langs[target_languages[chapter]],by_language=dict(langs),csv_sha256=sha(csvpath),yaml_sha256=sha(yamlpath),audit_sha256=sha(auditpath),audit_records=len(audit),audit_key_appearances=len(ak),variant_edges=sum(bool(r[11]) for r in rows),derived_parent_edges=sum(len(r[13].split('|')) for r in rows if r[13]),references=sorted(refs),receipt_evidence={str(p.relative_to(DATA)):sha(p) for p in receipts})
    results.append(item)
oldpath=DATA/'source_checklists/20260927-nair2013-chapter-coverage.json'
old=json.loads(oldpath.read_text())
installed=[c for c in results if c['installed']]
current_rows=sum(c['rows'] for c in installed)
if old['chapters_with_canonical_source_files']!=len(installed) or old['source_rows']!=current_rows:
    issues.append({'type':'stored presence inventory predates later installations','stored_chapters':old['chapters_with_canonical_source_files'],'current_chapters':len(installed),'stored_rows':old['source_rows'],'current_rows':current_rows,'recommendation':'Refresh shared presence inventory after final Mullu installation; this review does not mutate it.'})
report={'status':'independent read-only chapter evidence review; discrepancies reported separately','source_pdf_sha256':sha(ROOT/'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'),'chapter_generator_sha256':sha(generator),'chapter_generator_review':'Correct 13 contiguous chapter ranges cover printed/PDF 4–58. Generator inventories CSV/YAML presence and raw counts only; it intentionally does not prove audit/acceptance/integration gates and writes a shared report, so this peer script independently reads the same ranges without invoking it.','stored_coverage_sha256':sha(oldpath),'chapters':results,'aggregate':{'installed_chapters':len(installed),'installed_rows':current_rows,'target_rows':sum(c['target_rows'] for c in installed),'emitted_comparison_rows':sum(c['comparison_rows'] for c in installed),'by_language':dict(aggregate),'variants':sum(c['variant_edges'] for c in installed),'derived_parent_edges':sum(c['derived_parent_edges'] for c in installed),'distinct_chapter_reference_keys':len(allrefs)},'discrepancies':issues,'accounting_qualification':'Per-chapter physical blocks, editorial units and witness slots have different definitions and are not summed as a single physical-token count. Deliberately excluded controls remain audited; included comparison rows are reported separately. Shared-attribution expansion, slash expansion and exact commentary reuse change row counts without inventing physical source tokens.','common_deferred_gates':['coordinated full CLDF/database build','compiled ID/source-key/graph/reference survival','full test suite','browser/app QA and representative entries','publication'],'review_limits':'Handoffs/checklists and installed audit/hash/config closure reviewed; this does not repeat all original-page literal audits, execute tests, parse compiled CLDF or certify browser results. Current shared bibliography may legitimately exceed historical receipt whole-file hashes.','shared_mutations':False}
(OUT/'chapter-evidence-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'aggregate':report['aggregate'],'discrepancies':issues},ensure_ascii=False,indent=2))
