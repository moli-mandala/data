"""Record completed fresh original-image review and exhaustive narrow metadata checks."""
import csv, hashlib, json
from pathlib import Path
H=Path(__file__).resolve().parent
I=H.parent/'integration-proposal'
DATA=next(p for p in H.parents if (p/'make_cldf.py').exists())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((H/'routed-selection.json').read_text())
for f,h in s['input_hashes'].items(): assert sha(I/f)==h,f
a=[json.loads(x) for x in (I/'comparison-routing-audit.jsonl').read_text().splitlines()]
rows=list(csv.reader((I/'comparison-rows.csv').open()))
bykey={r[10]:r for r in rows}; assert len(bykey)==len(rows)==1370
prior=json.loads((H/'selection.json').read_text())
old={r['source_unit_key'] for f in ('selected_rows','targeted_rows') for r in prior[f]}
assert not old & {r['source_unit_key'] for r in s['selected_rows']}
lexical={202:'I',203:'you',204:'you',205:'he',206:'she',207:'we',208:'we',209:'you',210:'they'}
required={202:{'pron','1sg','sg'},203:{'pron','2sg','sg','informal'},204:{'pron','2sg','sg','formal'},205:{'pron','3sg','sg','m'},206:{'pron','3sg','sg','f'},207:{'pron','1pl','pl','inclusive'},208:{'pron','1pl','pl','exclusive'},209:{'pron','2pl','pl'},210:{'pron','3pl','pl'}}
decisions=[]
for r in a:
    if r['item'] not in lexical: continue
    d=r['gloss_field_decision']; assert d['printed']==r['source_prompt']==r['gloss']
    assert d['lexical']==lexical[r['item']]
    assert set(d['grammar_retained_in_tags'])==required[r['item']]<=set(r['tags'])
    if r['source_unit_key'] in bykey:
        assert bykey[r['source_unit_key']][3]==d['lexical']
        assert required[r['item']]<=set(bykey[r['source_unit_key']][14].split())
    else: assert r['source_code']=='T'
    decisions.append(dict(source_unit_key=r['source_unit_key'],decision=d,status='pass'))
assert len(decisions)==68
langs={r['ID']:r for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
identity_path=H.parent/'identity-review/all-source-routing-proposal.json'
identity=json.loads(identity_path.read_text())
new_language=identity['proposed_language_registry_row']
assert new_language['ID']=='Birjia' and new_language['ID'] not in langs
langs[new_language['ID']]=new_language
dialects=list(csv.DictReader((I/'comparison-dialects.csv').open()))
assert len(dialects)==6
for d in dialects:
    assert d['Clade']==langs[d['Language_ID']]['Clade']
    assert not any(d[k] for k in ('Latitude','Longitude','Quality'))
    assert d['Source_Language_ID'] in ('A','B','J','U','V','k')
    for r in a:
        if r['source_code']==d['Source_Language_ID']:
            assert r['language_id']==d['Language_ID'] and r['dialect_tag']==d['Tag']
results=[]
for n,r in enumerate(s['selected_rows'],1):
    assert r['proposed_csv_row']==bykey[r['source_unit_key']]
    page=int(r['historical_audit_record']['PDF_Page'])
    img=DATA.parent/f'tmp/pdfs/korwa-kodaku-2022/wordlist-200dpi/page-{page:03d}.png'
    results.append(dict(sample=n,source_unit_key=r['source_unit_key'],actual_csv_row=bykey[r['source_unit_key']],original_pdf_page=page,printed_page=page-10,original_image=str(img.relative_to(DATA.parent)),image_sha256=sha(img),status='pass',checks=['literal phonetic form and marks','English prompt scope','source code','complete response boundary','registered scoped tags and dialect routing','citation','blank inferred phonemic/ancestry fields']))
held=[r for r in a if r['integration_disposition']=='held-language-identity-conflict']
assert len(held)==246 and {r['source_code'] for r in held}=={'T'}
report=dict(status='passed independent fresh original-output review',seed=s['seed'],sampled=20,material_errors=0,
    input_hashes=s['input_hashes'],selection_sha256=sha(H/'routed-selection.json'),
    method='20 fresh rows across six eligible source-code strata, disjoint from all first-pass sampled and targeted keys. Every selected original page image viewed after selection. All 68 parsed-label decisions and six dialect parent-clade/coordinate conditions checked separately.',
    results=results,grammar_decisions=decisions,dialect_drafts=dialects,
    registry_sha256=sha(DATA/'cldf/languages.csv'),
    proposed_birjia_registry_row=new_language,identity_proposal_sha256=sha(identity_path),
    layout_policy='Three relocated item208 rows retain uncertain tags and exact typed source-layout reasons in audit. Parent explicitly accepted audit-only reason policy; no assertion that conflicting source ranks are resolved.',
    grammatical_scope='Paired imperative/past prompts retain whole source responses without assigning an unsupported response-specific tense or splitting unpunctuated sequences.',
    routed_rows=1370,identity_held_rows=246,
    limitation='This passes the corrected comparison subset review, not whole-source installation or full build. Earlier failed report and owner inputs remain immutable; whole-report integration gates remain separate.')
(H/'independent-routed-review-pass2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],sampled=20,material_errors=0,grammar_decisions=len(decisions),dialects=len(dialects))))
