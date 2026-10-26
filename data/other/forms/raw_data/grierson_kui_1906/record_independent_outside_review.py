"""Save separate independent adjudication, preserving owner first readings."""
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=next(p for p in P.parents if (p/'data/SOURCE_INGESTION_CHECKLIST.md').exists())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
inputs=['outsidechapter-original-reviewed.jsonl','outside-comparands-first-reading.jsonl']
rows=[json.loads(x) for x in (P/inputs[0]).read_text().splitlines()]
assert len(rows)==34 and len({r['source_unit_key'] for r in rows})==34
page_images={
210:['kui-root-outside-210.png'],290:['kui-outside-290-detail.png'],
291:['kui-outside-291-detail.png','kui-outside-291-young-man-restored-v2.png'],
293:['kui-outside-293-comparands-pronouns.png','kui-outside-293-comparands-table.png'],
295:['kui-outside-295-comparands-clauses.png'],296:['kui-outside-296-comparands-tense.png'],
413:['kui-outside-413-comparands-ablative.png'],479:['kui-outside-479-comparands-plural.png'],
481:['kui-outside-481-comparands-page.png','kui-outside-481-comparands-you-he.png'],
484:['kui-outside-484-comparands-honorific.png'],566:['kui-root-outside-566.png'],
570:['kui-outside-570-comparands-pronouns.png'],623:['kui-outside-623-comparands-case-goat.png'],
624:['kui-outside-alternate645.jpg','kui-outside-alternate645-bottom.png']}
evidence={}
for page,images in page_images.items():
    evidence[str(page)]=[{'path':'tmp/'+n,'sha256':sha(ROOT/'tmp'/n)} for n in images]
decisions=[]
for r in rows:
    d=dict(source_unit_key=r['source_unit_key'],form=r['form'],gloss=r['gloss'],
           literal_result='agrees_with_original',analysis_result='source_scope_supported',
           evidence=evidence[str(r['printed_page'])],uncertainty=r['uncertainty'])
    if r['source_unit_key'].endswith('p291:outside:young-man'):
        d.update(uncertainty_resolution='The separate upper horizontal e stroke is visible in the original. Confirm lāvēñju over lāveñju; no paradigm-based harmonization used.')
    if r['source_unit_key'].endswith('p210:outside:father'):
        d['attribution_result']='Context supports Kui but does not repeat the language label; retain the existing explicit attribution uncertainty.'
    if r['printed_page']==570:
        d['note']='These three Kui literals are directly printed and glossed. The separate Kolami cross-reference adjudication does not alter them.'
    if r['source_unit_key'].endswith('p481:outside:thou'):
        d['note']='Retain printed īmu, including m, and source thou gloss; no replacement by īnu from other pages.'
    if r['source_unit_key'].endswith('p566:outside:he'):
        d['note']='Plain initial e is retained. He remains contextual rather than a repeated direct gloss; surrounding Basim attribution applies to its own comparisons, not this explicitly named Kui token.'
    if r['printed_page']==624 and r['source_unit_key'].rsplit(':',1)[-1] in {'beat','do'}:
        d['note']='Distinct final curved mark versus the preceding dotted i is visible; source grapheme encoding remains provisional, without phonemic assertion.'
    decisions.append(d)
report={
 'reviewer':'/root/asur_finish','date':'2026-09-27',
 'status':'all 34 target Kui readings reviewed; Kolami reference expansion not independently licensed',
 'scope':'All 34 records in outsidechapter-original-reviewed.jsonl, spanning 14 printed pages; source-local comparison review, not an entire-volume claim.',
 'input_sha256':{n:sha(P/n) for n in inputs},
 'method':'Independently viewed every corresponding original image/crop, compared literal forms, glosses and explicit analysis/context; source punctuation and marks were not normalized. Enlarged p295 token crop as a fine-mark control. Original p566 full-page view was also performed during the immediately preceding identity review.',
 'counts':{'target_records':34,'printed_pages':14,'material_literal_corrections':0,'retained_source_uncertainties':sum(bool(r['uncertainty']) for r in rows)-1,'resolved_quantity_uncertainties':1},
 'decisions':decisions,
 'kolami_reference':{
   'printed_page':570,'evidence':evidence['570'],
   'verbatim_reference':'The same forms are usual in Kōlāmī.',
   'nearest_context':'A full Naiki personal-pronoun table is followed by Compare Kui ānu, I; āmu, we; īnu, thou, etc. The same-forms sentence then follows.',
   'decision':'documentary_unresolved_reference_no_literal_expansion',
   'reason':'The immediately preceding Kui examples are a plausible antecedent, but the main paragraph discusses regular Naiki pronoun forms and its complete table. The comparison clause ends etc., so the reference does not unambiguously specify precisely these three literals. Do not manufacture three certain independently attested Kolami rows from that ambiguity.',
   'affected_source_unit_keys':['grierson1906kui:p570:outside-comparand:12','grierson1906kui:p570:outside-comparand:13','grierson1906kui:p570:outside-comparand:14'],
   'candidate_expansions_preserved_as_audit_only':[{'form':'ānu','gloss':'I'},{'form':'āmu','gloss':'we'},{'form':'īnu','gloss':'thou'}],
   'recommended_representation':'Retain one physical documentary reference, exact wording and locus, linking both plausible antecedent groups. Keep all candidate expansion keys in audit as not emitted, not deleted. Explicit Kui literals and exact Naiki canonical reuses are unaffected.'},
 'limits':'No independent phonetic interpretation of the source notation key; existing attribution and curved-mark uncertainties remain explicit. No canonical output or frozen owner inputs changed.'}
comparands={r['source_unit_key']:r for r in [json.loads(x) for x in (P/inputs[1]).read_text().splitlines()]}
for key,candidate in zip(report['kolami_reference']['affected_source_unit_keys'],report['kolami_reference']['candidate_expansions_preserved_as_audit_only']):
    assert comparands[key]['source_language_label']=='Kolami' and comparands[key]['form']==candidate['form']
out=P/'independent-outside-target-and-kolami-review-20260927.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'file':str(out),'sha256':sha(out),'counts':report['counts']}))
