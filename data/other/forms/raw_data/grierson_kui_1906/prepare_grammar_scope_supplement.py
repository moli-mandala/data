"""Independent scoped addendum and guarded 34-record review derivative."""
import copy, hashlib, json
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=next(p for p in P.parents if (p/'data/SOURCE_INGESTION_CHECKLIST.md').exists())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n): return [json.loads(x) for x in (P/n).read_text().splitlines()]
g={r['source_unit_key']:r for r in read('grammar-reviewed.jsonl')}
report=json.loads((P/'independent-full-grammar-review-20260927.json').read_text())
assert sha(P/'grammar-reviewed.jsonl')==report['input_sha256']
units=[]
for n,(form,label) in enumerate([('Kandhs','Oriya'),('Gōnds','Telugu'),('Kōds','Telugu')],1):
    units.append(dict(source_unit_key=f'grierson1906kui:p457:ethnonym-documentary:{n}',printed_page=457,
      form=form,gloss='',source_language_label=label,disposition='documentary_attributed_ethnonym',
      lexical_emission=False,source_note=f'The source names {form} as a designation used by {label} people for the group. This is the printed English-running-text ethnonym, not an independently elicited Kui lexical item. Preserve the exact spelling and do not invent a singular, phonemic form or donor relation.',graph_edges=[]))
for n in range(1,5):
    original=g[f'grierson1906kui:p469:chinna-accusative:{n}']
    assert original['form']==['nanna','mamma','ninna','mimma'][n-1]
    units.append(dict(source_unit_key=f'grierson1906kui:p469:kanarese-accusative-comparison:{n}',
      printed_page=469,form=original['form'],gloss=original['gloss'],language_id='Kannada',
      source_language_label='Kanarese',tags=original['tags'],
      disposition='explicit_source_identity_comparison',lexical_emission=True,
      literal_source_key=original['source_unit_key'],physical_span_shared=True,
      source_note='The four Chinna Kimedi accusative pronouns are explicitly said to be identical with those used in Kanarese. This comparator preserves that source claim and its exact spelling, including mamma and mimma; it is not an independently reprinted token or a modern spelling correction. The paragraph is attributed to Lingum Letchmajee’s grammar.',
      graph_edges=[]))
updates=[dict(source_unit_keys=[f'grierson1906kui:p459:v-b-comparison:{n}' for n in [3,4,5,6]],
  printed_page=459,kind='distribution_context',
  source_note='The source contrasts Khondmals ēbāñju and ēbāru with Standard ēañju and ēāru, then states that the Standard form is also used in Kalahandi. Retain this distribution statement at the comparison block; its singular wording does not supply additional separately printed Kalahandi tokens.',
  new_attestations=False,graph_edges=[])]
for n in [1,2]:
    k=f'grierson1906kui:p460:plural-comparison:{n}'
    assert g[k]['form']==['ru','gaḷ'][n-1]
    updates.append(dict(source_unit_keys=[k],printed_page=460,kind='joint_language_attribution',
      form=g[k]['form'],source_languages=['Kanarese','Tamil','etc.'],language_id=None,
      disposition='joint_source_comparison_unresolved_individual_routing',
      uncertainty={'type':'language_attribution_scope','reason':'Both suffixes precede the joint phrase in Kanarese, Tamil, etc.; the source does not pair each suffix uniquely to one language or assert each one in every listed language.'},
      source_note='The source jointly compares ru and gaḷ in Kanarese, Tamil, etc. Preserve both literal suffix records and this collective attribution without guessing a suffix-to-language pairing.',
      new_attestations=False,graph_edges=[]))
sources=['tmp/kui-457.png','tmp/kui-grammar-original-leaf480.jpg','tmp/kui-grammar-original-leaf481.jpg','tmp/kui-grammar-original-leaf490.jpg']
supplement=dict(reviewer='/root/asur_finish',date='2026-09-27',
 status='originals independently inspected; source-local reconciliation proposal, canonical unchanged',
 input_hashes={n:sha(P/n) for n in ['grammar-reviewed.jsonl','independent-full-grammar-review-20260927.json']},
 evidence=[dict(path=s,sha256=sha(ROOT/s)) for s in sources],
 counts={'additional_units':7,'documentary_ethnonyms':3,'explicit_Kanarese_comparator_interpretations':4,'metadata_scope_updates':3},
 units=units,existing_unit_scope_updates=updates,
 interpretation='The four Kanarese interpretations share existing physical tokens; do not inflate the count of printed tokens. The explicit identical-with-those-used-in-Kanarese wording differs from the ambiguous same-forms cross-reference on p570.',
 protected_files='grammar-metadata-candidate and all owner/frozen inputs unchanged; no donor or derivation edges inferred.')
(P/'grammar-scope-supplement-20260927.json').write_text(json.dumps(supplement,ensure_ascii=False,indent=2)+'\n')

source='outsidechapter-original-reviewed.jsonl'
independent=json.loads((P/'independent-outside-target-and-kolami-review-20260927.json').read_text())
assert sha(P/source)==independent['input_sha256'][source]
old=read(source); new=copy.deepcopy(old)
for r in new:
    r['review_status']='independently reviewed against original; source uncertainties retained where stated'
    r['independent_review']='independent-outside-target-and-kolami-review-20260927.json'
    if r['source_unit_key']=='grierson1906kui:p291:outside:young-man':
        assert r['form']=='lāvēñju' and r['uncertainty']['type']=='transcription'
        r['uncertainty']=None
        r['resolved_uncertainty']='Independent original review confirms separate macron over e; literal form unchanged.'
assert [r['source_unit_key'] for r in old]==[r['source_unit_key'] for r in new]
assert len(new)==34
for a,b in zip(old,new):
    for k,v in a.items():
        if k not in {'review_status','uncertainty'}: assert b[k]==v
    if a['source_unit_key']!='grierson1906kui:p291:outside:young-man': assert a['uncertainty']==b['uncertainty']
target='outsidechapter-independent-reviewed.jsonl'
(P/target).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new))
receipt=dict(source=source,source_sha256=sha(P/source),derivative=target,derivative_sha256=sha(P/target),
 review_sha256=sha(P/'independent-outside-target-and-kolami-review-20260927.json'),
 records=34,keys_preserved=34,literal_field_changes=0,gloss_changes=0,analysis_changes=0,
 uncertainty_changes=1,remaining_uncertainties=sum(bool(r['uncertainty']) for r in new),
 guard='Exact source hash matched independent review; all pre-existing fields except review status and the one adjudicated uncertainty are byte-value identical.',canonical_changed=False)
(P/'outsidechapter-independent-reviewed-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'supplement':supplement['counts'],'derivative':receipt},ensure_ascii=False))
