"""Independent source-scoped metadata recommendations; never edits source readings."""
from pathlib import Path
import json, hashlib
H=Path(__file__).resolve().parent;P=H.parent
src=P/'first-reading/complete-first-reading.jsonl'
rows=[json.loads(l) for l in src.read_text().splitlines()]
assert hashlib.sha256(src.read_bytes()).hexdigest()=='22fe3f77dd063609585a70cbe032392e143c955dc237df471a4c16dedf316136'
units=[]; edges=[]
claims={
'nasalization':'The source identifies nasalization as a characteristic of Aranadan and compares these Malayalam forms.',
'medial-length':'The source describes lengthening of medial vowels in Aranadan, illustrated by the Malayalam comparison.',
'final-vowel':'The source attributes the change from Malayalam final -a to Aranadan -e to Kannada influence; it does not identify these words as Kannada loans.',
'initial-consonant':'The source attributes the illustrated v-to-b correspondence to Kannada influence; it does not identify individual Kannada donor words.',
'medial-consonant':'The printed prose says Malayalam i becomes y or j in Aranadan; the comparison forms instead contain a visibly underlined l. Both the printed claim and forms are retained.',
'geminate-consonant':'The source says the geminated form of dotted r becomes cc in Aranadan.'}
for r in rows:
 s=r['section']; pos=r['position']; ws=[]
 for w in r['witnesses']:
  tags=[{'pst':'pret','1':'first-person','2':'second-person','3':'third-person'}.get(t,t) for t in w['tags']]
  reason=[]; notes=[]; kind='lexical-attestation'
  if s in claims:notes.append(claims[s])
  if s=='ethnonyms':
   kind='documentary-community-name';notes.append('The source lists these as forms of the community name and tentatively connects them with Eranadu. The geographical derivation is conjectural.')
  if s=='nasalization' and pos==2:tags=['pron','1sg'];reason.append('Shared explicit I gloss scopes both target and comparator.')
  if s=='medial-length' and pos==3:tags=['pron','3sg'];reason.append('Shared he gloss supports third-person singular; no grammatical gender is inferred from English alone.')
  if s=='pronoun-prose':
   tags=['pron','1pl' if pos in (1,2) else '2sg'];reason.append('Explicit first plural / second singular prose, including the parenthetical Kannada comparison.')
   if pos in (1,2):notes.append('The source denies an inclusive/exclusive distinction; this form denotes a '+('large' if pos==1 else 'small')+' group.')
  if s=='plural-prose':
   kind='bound-morphology' if pos in (1,2) else 'lexical-attestation'
   if pos==3:tags=['adj','quantifier'];reason.append('Explicitly called a quantitative adjectival form, used with inanimate plural expressions.')
   else:notes.append('The source allows the suffix with '+('animate and inanimate nouns in Malayalam.' if pos==1 else 'animate nouns only in Aranadan.'))
  if s in ('dative-prose','tense-suffixes'):kind='bound-morphology';reason.append('Explicit source suffix role, not a lexical verb.')
  if s=='dative-prose':notes.append('The source says this dative suffix is not used after pronouns.')
  if s=='tense-suffixes':
   notes.append('The source lists this as a '+('future' if pos==1 else 'past' if pos<=4 else 'present')+' tense suffix.')
   if pos==1:notes.append('The source explicitly compares the same future suffix with Malayalam; this is not a borrowing claim.')
  if s=='verb-paradigm':
   col=str(pos).split(':')[1];tags=['verb']+({'stem':['stem'],'past':['pret'],'present':['pres'],'future':['fut']}[col]);kind='verb-stem' if col=='stem' else 'inflected-verb'
   reason.append('Source heading Tense paradigm of some verbs and the exact column label; no person, number, gender, or infinitive inferred.')
  if s in ('pronoun-examples','dative-examples'):
   kind='whole-clause';reason.append('Whole source translation; no unaligned word-level lexical glosses or person/agreement tags propagated to the construction.')
   if s=='pronoun-examples' and pos==3:tags.append('neg');reason.append('Explicit negative whole translation do not speak.')
   if s=='dative-examples' and pos==1:tags.append('interr');reason.append('Explicit whole English translation do you want is interrogative; no question mark inserted into Original.')
   if s=='dative-examples':notes.append('The source gives this expression to illustrate absence of the dative suffix after pronouns.')
  if s in ('inanimate-plurals','genitive-examples'):
   kind='whole-phrase';reason.append('A whole printed phrase, not a clause or a compound analysis.')
   if s=='inanimate-plurals':tags=['multiword-expression','pl'];notes.append('The source illustrates inanimate plurality through the quantitative adjectival form kuṛe.')
   else:notes.append('The source presents the phrase as an example of genitive deletion; no overt genitive inflection is inferred.')
  if s=='vocabulary' and pos==3:notes.append('The source specifies the disease sense of cold.')
  if not reason:reason.append('No additional explicit grammatical heading or label; do not guess POS from an English noun-like gloss.')
  ws.append(dict(witness_key=w['witness_key'],form=w['form'],language=w['language'] if s!='ethnonyms' else None,raw_proposed_language=w['language'],export_disposition='documentary only; lexical language attribution not established' if s=='ethnonyms' else 'emit reviewed witness',kind=kind,proposed_tags=list(dict.fromkeys(tags)),existing_tags=w['tags'],tag_basis=reason,source_notes_to_retain=notes,graph_disposition='No supported edge for this occurrence unless separately listed.'))
 units.append(dict(source_unit_key=r['source_unit_key'],section=s,witnesses=ws,disposition='emit reviewed witnesses' if ws and s!='ethnonyms' else 'documentary/context only',context_basis=r['description'] if not ws else None))
def key(s):return 'nair2013aranadan:'+s
for child,parent,basis in [
 ('p9:medial-consonant:2:w2','p9:medial-consonant:2:w1','Explicit j/y alternation claim and the paired naaji/naayi forms.'),
 ('p9:medial-consonant:3:w2','p9:medial-consonant:3:w1','Explicit j/y alternation claim and the paired puje/puye forms.'),
]:
 edges.append(dict(child=key(child),parent=key(parent),type='variant',basis=basis,status='proposed for independent owner approval'))
keys={w['witness_key'] for r in rows for w in r['witnesses']}
assert len(units)==97 and sum(len(u['witnesses']) for u in units)==105
assert all(e['child'] in keys and e['parent'] in keys for e in edges)
x=dict(status='independent proposal; no canonical writes',reviewer='asur_finish',input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),coverage=dict(units=97,witnesses=105,context_units=sum(not u['witnesses'] for u in units),documentary_name_units=4,emission_witnesses=101,additional_shared_form_witnesses=1),source_evidence='Original pp9–11 visually reviewed; review/notation-render-manifest.json and scope/render-manifest.json',units=units,proposed_edges=edges,
 additional_source_witnesses=[dict(proposed_key=key('p10:tense-suffixes:1:malayalam'),form='-um',language='Malayalam',tags=['suffix','fut'],gloss='',basis='The future tense suffix of Aranadan is -um, as in Malayalam. Explicit shared form/function, not a reconstructed form.',physical_locator=key('p10:tense-suffixes:1'),physical_new_printed_form=False)],
 graph_class_decisions=[
 dict(scope='32 verb paradigm cells',decision='Ordinary tense inflection, not derivation. No automatic stem or suffix component edges; source does not segment each allomorphic cell.'),
 dict(scope='kuṛe maram / kuṛe kallu',decision='Quantifier construction explicitly described, but the independent lexical endpoints maram/kallu are not separately glossed in this inventory. Do not invent base rows or incomplete whole-derivation edges; preserve construction claim in Notes.'),
 dict(scope='Malayalam comparisons and Kannada influence claims',decision='Source phonological comparison/contact influence is not an explicit lexical borrowing statement. No borrowed edges or loanword tags.'),
 dict(scope='ethnonym Eranadu connection',decision='Source says may be connected; preserve attributed conjecture, no asserted geographic derivation edge.'),
 dict(scope='aranaaṭan / Aranaaṭan',decision='All four community-name occurrences are documentary only: the source does not establish lexical-language attribution. Preserve name-variation claims in audit, no lexical variant edges.'),
 dict(scope='case deletion examples',decision='Zero/absent morphology is a source claim, not an unattested suffix-bearing base. No generated words or donor/component edges.')],
 required_export_checks=['All proposed tags must pass canonical registry normalization; raw pst/1/2/3 should not leak into output.','Account for all105 physical witnesses:101 lexical/morphology witnesses and4 documentary community-name occurrences; add the explicitly shared Malayalam future-suffix witness on the existing physical unit.','No inclusive/exclusive tags; no verb tag on bare tense suffixes; no gen tag falsely describing an overt suffix in deletion examples.','Do not let English gloss spaces determine multiword status of single source words.','Preserve all source claims in scholarly Notes and exact raw evidence in audit; drop only workflow prose at export.'])
original={w['witness_key']:w for r in rows for w in r['witnesses']}
x['keyed_field_proposals']=[]
for u in units:
 for w in u['witnesses']:
  old=original[w['witness_key']]
  if w['kind']=='documentary-community-name':continue
  if old['tags']!=w['proposed_tags']:
   x['keyed_field_proposals'].append(dict(witness_key=w['witness_key'],field='tags',old=old['tags'],new=w['proposed_tags'],basis=w['tag_basis']))
  if w['source_notes_to_retain']:
   note=' '.join(w['source_notes_to_retain'])
   x['keyed_field_proposals'].append(dict(witness_key=w['witness_key'],field='note',old=old['note'],new=note,basis='Source-scoped scholarly claim; raw first-reading note remains in audit.'))
x['root_disposition_authority']='../root-review/ethnonym-disposition.json'
x['edge_direction_policy']='Second listed alternate points to first for display only; no chronological priority is inferred.'
import ast
W=next(p for p in H.parents if (p/'data/tags.py').exists())
registry=next(ast.literal_eval(n.value) for n in ast.parse((W/'data/tags.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GRAMMATICAL_TAGS' for t in n.targets))
assert all(t in registry for u in units for w in u['witnesses'] for t in w['proposed_tags'])
x['validation']={'exact_unit_keys':len({u['source_unit_key'] for u in units})==97,'exact_witness_keys':len(keys)==105,'all_proposed_tags_registered':True,'all_graph_endpoints_resolve':True,'source_reading_unchanged':True}
(H/'metadata-graph-proposal.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(x['coverage'],len(edges),'edges proposed')
