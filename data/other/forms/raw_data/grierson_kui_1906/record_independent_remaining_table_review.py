"""Record completed original-image review; never mutate the owner's transcription."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
def pin(path):
 p=ROOT/path
 return {'path':str(path),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
source=HERE/'table-first-reading.json'
readings=json.loads(source.read_text())['readings']
prompts={r['prompt_id']:r for r in map(json.loads,(HERE/'table-prompt-inventory.jsonl').read_text().splitlines())}
# These are keyed reviewer proposals, not accepted owner changes.
changes={
35:('Kānnu',['Kānnu'],'Visible horizontal bar over first a; narrower original strip distinguishes it from adjacent plain letter tops.'),
47:('Ābā, tāñji',['Ābā','tāñji'],'Initial capital A has a separate macron. Preserve the independently unresolved stacked mark on second alternative.'),
49:('Dādā (elder), tāmēsā (younger).',['Dādā','tāmēsā'],'Younger form has horizontal upper stroke over e; ordinary s retained. Elder/younger qualifiers retain separate alternative scope.'),
53:('Kḍuā',['Kḍuā'],'Separate u body follows dotted d before final ā. Initial K directly adjoins ḍ; no o is inserted before ḍ. Reviewer proposes recovering this formerly headless cell; parent adjudication still required.'),
92:('Umbāē',['Umbāē'],'Final rounded glyph has e body and thick horizontal upper component, unlike ordinary s. Prefer ē, retain compressed e/ē typography uncertainty rather than excluding the whole form.'),
103:('Ro ābārki',['Ro ābārki'],'Visible interword gap after Ro. Restore printed space; no paradigm-based reconstruction.'),
160:('Iru māñjēru',['Iru māñjēru'],'Printed predicate shows upper horizontal components on a and e; compare narrow strip against the same row, not other persons. Quantity remains compressed; request parent adjudication.'),
172:('Ānu ābā-muĩ',['Ānu ābā-muĩ'],'Rounded open-top u after m, not a. Retain curved i mark as provisional source encoding.'),
174:('Ānu ājā duĩ',['Ānu ājā duĩ'],'Rounded open-top u after d, not a. Retain curved i mark as provisional source encoding.'),
210:('Ēbāru sānēru',['Ēbāru sānēru'],'Predicate medial vowel has e body and upper bar; it is not ā. Read independently from adjacent alternative209.'),
226:('Sukāli goṛākāri jini iḍulāiṭi mānē.',['Sukāli goṛākāri jini iḍulāiṭi mānē.'],'House expression includes kā before ri, final i in jini, and ā+i after l in iḍulāiṭi. See native-resolution strip; full sentence retained. Parent adjudication requested for dense middle sequence.'),
231:('Tānā dādā tānā āṅgi-bāhā-ṭākā ikē ḍeṅgā-gāṭāñju.',['Tānā dādā tānā āṅgi-bāhā-ṭākā ikē ḍeṅgā-gāṭāñju.'],'Comparative context word visibly ends ē, not a. Other clause spelling retained.'),
234:('Irā ṭākā ēbāniki simu',['Irā ṭākā ēbāniki simu'],'First word final ā clearly printed; not i.'),
}
notes={
26:'Parenthesized alternative retained as printed, without creating an unattested pronoun combination.',
27:'Curved i upper mark has the same source-shape ambiguity as earlier pronouns; retain literal provisional encoding, no phonological conclusion.',
28:'Final e quantity compressed; retain current ē with typography caveat.',
34:'Printed medial e looks plain; Muṅgeli preferred over Muṅgēli.',
39:'No secure separate upper bar on e; retain Leñjā with existing Lēñjā alternative uncertainty.',
41:'Medial e upper component compressed; retain existing literal and quantity uncertainty.',
51:'Stacked medial marks remain compressed. Retain maḍhēñju provisionally with existing explicit candidates; no nasal phoneme inference.',
52:'Heavy medial ink obscures underdot shape; retain Āḍā with source typography uncertainty.',
55:'Printed i has compact ordinary dot. Medial e/nasal stack retained as existing Mriēñju, not normalized from other attestations.',
59:'Original supports u before ḍ, not ṇ: retain Gāuḍēñju.',
63:'Existing Ḍāñjupēnu preferred; compressed stacked mark after ā remains uncertain, do not add a separately inferred e.',
71:'Two o bodies present, first upper bar and second nasal mark; retain two vowels and provisional quantity/nasal encoding.',
72:'Upper mark over final u is compact and unusually dot-like. Retain existing ū only provisionally with explicit uncertainty between a compressed length mark and another source mark; do not silently drop it.',
90:'Both medial e quantity/nasal upper components compressed; keep source-specific uncertainty.',
97:'Explicit (a suffix) confirmed. Final upper nasal mark is displaced rightward; retain source spelling without inventing a boundary hyphen.',
99:'First alternate e upper stroke compressed; retain current length uncertainty.',
106:'Final e has compressed thick upper component. Existing plain-e reading remains uncertain against ē; no forced harmonization across gule/gulē occurrences.',
118:'Printed bāhāṭākā has no hyphen at the internal boundary; retain this occurrence.',
124:'Final e of Gule has compressed upper component; retain plain e provisionally with length uncertainty.',
127:'Existing Gulē preferred but upper stroke compressed; retain recorded quantity alternative.',
128:'Existing medial cluster and curved final i retained with explicit typed uncertainty.',
129:'Keep both compound boundary and source male-child qualifier; medial consonant/length remains compressed.',
130:'Same typography caveat as128 but independently attested plural form; no spelling harmonization.',
132:'Three semicolon-separated source alternatives retained; no unattested grammatical gender labels inferred from their endings.',
135:'Nasal upper mark crowded; retain existing Ḍeṅgā with typography uncertainty.',
153:'Visible gap before li confirmed. Do not silently join to adjacent sitāli attestations.',
157:'Ink genuinely missing across initial pronoun and predicate. Existing Inu mañji is a usable provisional reading only, with explicit missing-ink uncertainty; no confident macron reconstruction from other persons.',
169:'Dense internal marks retained; final i compact dot, not a confident extra macron.',
175:'Third alternate middle m/w still compressed. Retain kommu with explicit kowmu alternative uncertainty.',
185:'Parenthesized mā is source notation, not authority to fabricate unprinted alternatives.',
188:'Parenthesized mā retained independently in this occurrence.',
209:'Explicit alternative predicate shares printed Iru only; source-scoped full-expression expansion is allowed with raw wording retained.',
221:'Final t/ṭ remains compressed; existing t preferred provisionally. Preserve exact question and uncertainty.',
223:'Plural mrikā independently visible; do not substitute singular elsewhere.',
224:'Full clause and joined trejā-māĩ retained; existing dense pāhēri marks remain typography-qualified.',
225:'Printed first pronoun i has compact dot, distinct from explicit curved final i in238. Retain Nāi independently.',
227:'kuiṭi supported, without n; printed jinu retained.',
228:'mriēni retained; first i compact dot and medial upper e stroke compressed. Preserve full sentence.',
229:'Second word underdot and e upper component support existing sēṇḍo.',
230:'Printed tree expression and kuiṭi retained. No word-level invented glosses extracted from whole translation.',
233:'Two alternatives scope only over final predicate; shared clause prefix may be inherited with explicit expansion provenance.',
236:'Comma separates coordinated clauses, not alternatives. Retain whole expression.',
239:'Existing forms retained including compact initial Ni and source nasal stack; no harmonization with other sons forms.',
240:'Final i upper mark in koḍiti is wider than nearby ordinary dots; current i versus ī remains typed quantity uncertainty. Do not normalize from another verb.',
}
items=[]
for n in range(26,242):
 row=next((r for r in readings if r['prompt_id']==n),None)
 if row is None:
  assert n in (133,136,195,196,197,198,199,200)
  items.append({'prompt_id':n,'source_key':prompts[n]['key'],'status':'physical blank confirmed' if n in (133,136) else 'shared explicit future cross-reference confirmed','evidence':'Printed dotted leader, no lexical response.' if n in (133,136) else 'One bracket says As the present tense, spanning195–200. Resolve corresponding179–184 only after their readings are accepted, preserve both source locators; not six printed tokens.'})
  continue
 lit,forms,evidence=changes.get(n,(row['literal'],row['forms'],'Independent complete original-image review agrees with the literal, alternative boundaries and corresponding prompt.'))
 if n in notes:evidence+=' '+notes[n]
 status='correction proposed' if n in changes else ('retained with typed caveat' if n in notes or row.get('uncertainty') else 'agrees')
 items.append({'prompt_id':n,'source_key':prompts[n]['key'],'printed_page':prompts[n]['printed_page'],'archive_leaf':row['source_leaf'],'english_prompt':prompts[n]['english_prompt'],'old_literal':row['literal'],'proposed_literal':lit,'old_forms':row['forms'],'proposed_forms':forms,'status':status,'evidence':evidence,'prior_uncertainty':row.get('uncertainty'),'old_tags':row['tags']})
metadata=[
 {'prompts':[103],'proposal':'Add multiword-expression after restoring printed space.'},
 {'prompts':[134,137],'proposal':'Add registered degree tag; retain exact superlative label in Notes. No unsupported comparative form for blank133/136.'},
 {'prompts':[170,171,177,178,218,219],'proposal':'Review explicit nonfinite prompt series as participial forms, retaining exact being/having been/beating/having beaten/going/gone labels. Add participle where confirmed by chapter analysis; do not infer a specific target morphological suffix solely from English.'},
 {'prompts':list(range(138,156)),'proposal':'Preserve explicit male/female versus general animal prompt distinctions in Gloss/Notes. Do not automatically equate lexical sex terms with a grammatical gender class without source support. Number contrast already explicit in paradigm.'},
 {'prompts':list(range(220,242)),'proposal':'All22 are complete printed sentence prompts, including imperatives; add sentential to multiword-expression. Preserve complete English translations, question scope and explicit two-clause punctuation; no invented token alignment.'},
 {'prompts':[92,93,94],'proposal':'Explicit who/what/why prompt functions support interr; do not infer an extra pronoun POS for why merely from adjacency.'},
 {'prompts':list(range(156,168))+list(range(179,191))+list(range(195,201))+list(range(205,217)),'proposal':'Retain source person/number/tense labels; normalize draft prs/imp only through existing registered aliases to pres/impv, not new vocabulary.'},
]
report={'status':'Complete independent26–241 original-image reading; proposals require owner reconciliation, not installation approval','reviewer':'norton_finish','date':'2026-09-27','owner_inputs_edited':False,'input':pin(source.relative_to(ROOT)),'prompt_input':pin((HERE/'table-prompt-inventory.jsonl').relative_to(ROOT)),'coverage':{'prompts':216,'literal_cells':208,'no_response_cells':2,'future_cross_reference_prompts':6,'first25_review_separate':True},'sources':[pin(Path(f'tmp/kui-alternate-leaf{x}.jpg')) for x in range(673,702,4)],'detail_images':[pin(p.relative_to(ROOT)) for p in sorted((ROOT/'tmp').glob('kui-independent-detail-*.png'))],'correction_count':len(changes),'corrections':[r for r in items if r['status']=='correction proposed'],'metadata_proposals':metadata,'items':items,'residual_policy':'Dense ink and missing ink are local typed transcription uncertainty, not grounds for omitting an attested cell. Curved i marks remain provisional literal encoding, not asserted nasal phonology. Existing owner uncertainty is not silently cleared unless explicitly resolved above. All original source files preserved.'}
(HERE/'independent-table-26-241-review-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'coverage':report['coverage'],'corrections':len(changes)},ensure_ascii=False))
