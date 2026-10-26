"""Reproduce outside-source census evidence without canonical writes or new renders."""
from pathlib import Path
import hashlib, json
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'data/source_meta.py').exists())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def read(p): return json.loads(p.read_text())
old=HERE/'scope-decisions.json'
prior=HERE/'scope-decisions-before-original-closure.json'
if not prior.exists(): prior.write_bytes(old.read_bytes())
decisions=read(prior)
closure=read(HERE/'sharper-original-closure-review.json') if (HERE/'sharper-original-closure-review.json').exists() else None
if closure:
 for witness in decisions['physical_lexical_witnesses_outside_chapter']:
  for resolved in closure['p182']['resolved_forms']:
   if witness['key']==resolved['key']:
    witness['form_candidate']=resolved['form']
    witness.pop('alternatives',None)
    witness['literal_resolution']=resolved['resolution']
    witness['disposition']='literal_original_reviewed_candidate' if witness['source_label']=='Kolami' else 'literal_resolved_identity_review_pending'
    if witness['source_label']!='Kolami' and closure.get('accepted_Berar_route'):
     witness['disposition']='root_accepted_qualified_historical_attribution'
     witness['accepted_route']=closure['accepted_Berar_route']
 decisions['addenda_original_review']['pages']=[16,17,18]
 decisions['addenda_original_review']['originals'].insert(0,str((HERE/'later-addendum-leaf704-unverified.jpg').relative_to(ROOT)))
 decisions['addenda_original_review']['p16']='Complete sharper original n704 visibly printed16 reviewed: corrections30–267; p168 adds Korku authority only, no182 or selected chapter correction.'
documentary=decisions.pop('retrieval_only_documentary_pages')
decisions['original_documentary_pages_checked']=documentary
for p,note in documentary.items(): decisions['original_pages_checked'][p]='Entire original viewed; '+note
for p,note in {'501':'Source explicitly labels the surrounding section Gondi and Basim specimens Gondi, despite geographic coincidence. Not historical Dravidian Bhili of Basim; no Naikri attribution.', '622':'Entire original Brahui page viewed. Kolami is named in a dative/accusative-class comparison, with no literal Kolami exponent printed. Documentary source claim only.'}.items(): decisions['original_pages_checked'][p]=note
decisions['pending']=['p182 negative-form fine glyph and broader Berar identity adjudication; retained as typed holds', 'Original addendum16 is not retained locally: OCR range evidence only, no new original absence claim']
if closure: decisions['pending']=[] if closure.get('accepted_Berar_route') else ['p182 broader Berar identity awaits root adjudication; literal tōlen/todī resolved by sharper original, without phonological inference']
decisions['status']='Outside-source census reviewed; two p182 source uncertainties preserved; canonical integration deferred'
if closure:
 decisions['original_pages_checked']['182']='Complete sharper primary n202 original verified printed182 and narrow native crop viewed: tōlen and todī resolved, independently confirmed by root. Qualified Naikri base only for broader Dravidian Bhili of Berar, uncertain historical-attribution caveat; no Basim lect/coordinates or borrowing edge.'
 decisions['status']='Outside census and root qualified source decisions passed; canonical integration deferred'
old.write_text(json.dumps(decisions,ensure_ascii=False,indent=2)+'\n')
leads=read(HERE/'normalized-volume-leads.json')
chapter={'561','562','565','566','567'}
classifications=[]
for page in leads['lead_pages']:
 p=page['printed_page']
 if p in chapter: disposition='chapter_owner_scope'
 elif p in {'182','623','626','627'}: disposition='explicit_lexical_witness_separately_keyed'
 else: disposition='documentary_or_non_target_control'
 assert p in chapter or p in decisions['original_pages_checked'],p
 for hit in page['hits']:
  classifications.append({'printed_page':int(p),'sentence_index':hit['sentence_index'],'raw':hit['raw'],'disposition':disposition,'physical_literal_accounting_separate':True})
images=[]
for manifest in sorted(HERE.glob('original-printed-*-1600.json')):
 m=read(manifest); img=manifest.with_suffix('.jpg')
 assert sha(img)==m['sha256']
 p=m['requested_printed_page']; physical=m['djvu_physical_page']
 assert physical==p+(19 if p<277 else 20),(p,physical)
 images.append({'printed_page':p,'physical_page':physical,'original_folio_visually_verified':True,'image':str(img.relative_to(ROOT)),'sha256':m['sha256']})
front=read(HERE/'frontmatter-original-manifest.json')
for im in front: assert sha(ROOT/im['image'])==im['sha256']
overlap=read(HERE/'exact-overlap-and-table-review.json')
evidence=[HERE/'normalized-volume-leads.json',HERE/'exact-overlap-and-table-review.json',old,HERE/'frontmatter-original-manifest.json',ROOT/'data/data/other/forms/raw_data/grierson_holiya_1906/independent-scope-identity-review-20260926.json',ROOT/'data/data/other/forms/raw_data/grierson_holiya_1906/independent-errata-applicability-review-20260926.json',ROOT/'tmp/holiya-scope-review/addenda-leaf706.jpg',ROOT/'tmp/korvi-full-scope/alternate-leaf708.jpg',ROOT/'tmp/korvi-full-scope/alternate-fulltext.txt',ROOT/'tmp/naiki-scope/dsal/570.jpg',ROOT/'tmp/naiki-scope/dsal/571.jpg',ROOT/'tmp/kui-outside-623-comparands-page.png']
if closure: evidence.append(HERE/'sharper-original-closure-review.json')
report={
 'reviewer':'/root/asur_finish','date':'2026-09-27','status':'outside_scope_reviewed_with_explicit_source_holds',
 'source_sha256':'33e9aaa220db22fcde712705581a69f7edcfc7e20b3e7a09347dc969768020f1',
 'retrieval':{'keyed_pages':len(leads['all_volume_keyed_pages']),'lead_pages':len(leads['lead_pages']),'lead_sentences':len(classifications),'normalization':'NFD accent stripping and casefold used only for retrieval labels; raw source text retained unchanged','classified_leads':classifications},
 'physical_lexical_witnesses_outside_chapter':decisions['physical_lexical_witnesses_outside_chapter'],
 'counts':{'outside_physical_tokens':5,'outside_language_attributions':6,'new_outside_physical_tokens_not_existing_exact_rows':4,'existing_canonical_rows_reused_including_chapter':3,'physical_printings_represented_by_existing_rows':2,'existing_outside_physical_printings':1,'p566_historical_held_rows':5},
 'originals':images,
 'contents_review':{'physical_pages':list(range(8,16)),'printed_pages':['v','vi','vii','viii','ix','x','xi','xii'],'all_eight_originals_visually_checked':True,'main_entries':{'Kolami':561,'Bhili of Basim':566,'Naiki':570,'Gondi of Basim':501,'standard_Dravidian_list':646},'index_result':'Contents list no separate alphabetical volume index. Eight contents pages are documentary headings/page locators, not lexical attestations. No claim of full 701-page visual absence proof.','Gondi_Basim_control':'Contents and original p501 explicitly distinguish Gondi of Basim from historical Bhili of Basim; location alone is not language identity.'},
 'tables':{'cached_population_hits':overlap['table_label_matches'],'standard_list_header_evidence':'Complete original646–649 header review independently retained in grierson_holiya_1906/independent-scope-identity-review-20260926.json; all11columns, no Kolami/Basim/Naiki. Do not invent a standard241-cell column.','cached_id_policy':'Cached table IDs are retrieval identifiers here; original folios not independently certified by this table parser.'},
 'errata':decisions['addenda_original_review'],
 'controls':[{'printed_page':186,'form':'n','source_label':'Bhili and Khandeshi','disposition':'non_target_unqualified_comparison','reason':'No Dravidian or Basim qualification, so not mapped to Naikri.'},{'printed_page':233,'form':'kōlār','source_label':'Gadaba','disposition':'false_label_retrieval_lead','reason':'Example word, not Kolami language label.'},{'printed_page':501,'source_label':'Gondi of Basim','disposition':'outside_target_language_scope_control','reason':'Explicit distinct source language; no geographic reassignment.'}],
 'source_reference_policy':'p570 same-forms and p571/p480/p622 class cross-references remain documentary; do not reconstruct unprinted Kolami paradigms. Qualified comparison or conjectured loan is not an accepted edge.',
 'exact_overlap_report':'exact-overlap-and-table-review.json','evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in evidence},
 'mapping_history':{'failed_first_mapping_image':'first-mapping-attempt-physical-202-printed-183.jpg','decision':'Physical202 is printed183, not182. Historical failed request/manifest retained; excluded from accepted182 evidence. Verified182 is physical201.'},
 'limits':['Keyed Nordic prose has305page units and omits tables/images/pages; retrieval alone is not exhaustive original absence proof.','Original DjVu OCR is6.6KiB corrupted sparse headings, never used for accepted literals or absence.','Originals of every substantive outside lead and all original contents were checked; whole original volume was not reread page by page.','Later addendum16 has localOCR range evidence through267 only; no retained original16 was independently viewed. Original17–18 checked completely and no561–575 corrections found.','p182 glyph and Berar identity remain explicit holds; no phonology, donor edge, or Basim lect guessed.','Chapter561–569 extraction belongs to other owners; adjacent Naiki570–575 remains existing independent package. No canonical writes/build/DB/browser/full-suite claims.'],
 'canonical_mutations':False}
assert len(report['physical_lexical_witnesses_outside_chapter'])==5
report['canonical_overlap_qualification']=overlap['exact_emitted_occurrences']
report['counts']['existing_canonical_rows_reused_including_chapter']=2
report['counts']['canonical_rows_with_physical_overlap_including_chapter']=3
report['counts']['p566_literal_equality_pending']=1
report['limits'].append('p566 current canonical and vs newly read amd is a physical-key overlap only; literal reuse remains pending sharp-original adjudication. Preserve one occurrence plus correction history, not duplicate and/amd attestations.')
if closure:
 report['sharper_original_closure']=closure
 report['limits']=[s for s in report['limits'] if not s.startswith('Later addendum16') and not s.startswith('p182 glyph')]
 report['limits'].append('p182 literal tōlen/todī resolved; root accepted qualified Naikri base only, with uncertain historical-attribution caveat and no Basim lect, coordinates or donor edge.')
 report['status']='passed_outside_census_and_root_qualified_decisions'
(HERE/'outside-census-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(HERE/'OUTSIDE_REVIEW.md').write_text('''# Kolami / Basim outside-chapter review

Five physical lexical tokens occur outside pp. 561–569: two comparisons on p. 182, one shared Naiki/Kolami `nē` on p. 623, and separate Kolami `tōten` occurrences on pp. 626 and 627. The p. 182 glyph and broader “Dravidian Bhili of Berar” identity remain typed holds. That label does not identify the Basim lect.

Three exact installed rows represent two printings when the p. 566 Kolami `and` overlap is included. Two of those rows share the one p. 623 printing. Five historical p. 566 Kui holds remain immutable evidence; later source reconciliation must preserve that history and must not fabricate Form ID aliases.

All retrieved documentary leads, eight original contents pages, and substantive lexical pages have been checked. The original contents and p. 501 explicitly distinguish Gondi of Basim from Bhili of Basim. Cached table label hits are population/distribution records; previously independently reviewed complete standard-list headers have no Kolami/Basim/Naiki column. No unprinted paradigm was reconstructed from class cross-references.

Complete original addenda 17–18 contain no relevant chapter correction. Addendum 16 has local OCR range evidence only, retained as a review limitation. Nordic keyed prose and the corrupted original OCR are discovery aids, not a full-volume original absence proof.

`outside-census-review.json` records physical keys, exact reuse, every retrieval hit, original folios, controls, hashes, and limits. Reproduce report accounting with `finalize_census.py`; exact current canonical closure with `verify_overlap_and_tables.py`. No canonical source, registry, bibliography or compiled files were modified. Integration, full builds, compiled validation and browser QA remain deferred.
''')
print(json.dumps({'status':report['status'],'counts':report['counts'],'lead_pages':report['retrieval']['lead_pages'],'lead_sentences':len(classifications)}))
if closure:
 doc=HERE/'OUTSIDE_REVIEW.md'
 text=doc.read_text().replace('The p. 182 glyph and broader “Dravidian Bhili of Berar” identity remain typed holds. That label does not identify the Basim lect.','Sharper primary original resolves p. 182 as `tōlen` and `todī`. Root independently confirmed both. Broader “Dravidian Bhili of Berar” routes to Naikri with an uncertain historical-attribution caveat; it does not identify the Basim lect or establish a borrowing edge.')
 text=text.replace('Complete original addenda 17–18 contain no relevant chapter correction. Addendum 16 has local OCR range evidence only, retained as a review limitation.','Complete later original addenda 16–18 contain no relevant p. 182 or chapter correction. The newly retrieved n704 original visibly printed16 corrects earlier pages30–267; p168 adds an authority only. These later supplement witnesses remain separate from the original1906 edition.')
 text=text.replace('Three exact installed rows represent two printings when the p. 566 Kolami `and` overlap is included. Two of those rows share the one p. 623 printing.','Three installed rows overlap two physical printings, including the p. 566 Kolami key. The p. 566 `and` versus newly read `amd` spelling awaits sharp-original adjudication, so this is physical-key overlap only, not closed literal reuse. The two p. 623 `nē` rows remain exact accepted reuse of one printing.')
 doc.write_text(text)
