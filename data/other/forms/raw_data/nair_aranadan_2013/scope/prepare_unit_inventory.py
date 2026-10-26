"""Allocate complete chapter positions, not accepted lexical transcriptions."""
from pathlib import Path
import collections,hashlib,json
H=Path(__file__).resolve().parent
source=json.loads((H/'whole-pdf-census.json').read_text())
positioned={r['pdf_page']:r['spans'] for r in json.loads((H/'chapter-positioned-text-unaccepted.json').read_text())}
units=[]
def add(page,section,number,kind,description,witnesses=1):
    units.append(dict(source_unit_key=f'nair2013aranadan:p{page}:{section}:{number}',pdf_page=page,printed_page=page,
       section=section,position=number,kind=kind,description=description,expected_physical_witness_slots=witnesses,
       raw_evidence=f'page-{page:02d}-raw-text.txt',positioned_evidence='chapter-positioned-text-unaccepted.json',
       original_image=f'original-page-{page:02d}-150dpi.png',
       status='inventoried; glyph transcription and final export disposition pending',accepted_transcription=False))
for n in range(1,5):add(9,'ethnonyms',n,'named-form-occurrence','Introductory italic name occurrence; preserve repeated/capitalized witnesses and tentative Eranadu connection.')
for section in ('nasalization','medial-length','final-vowel','initial-consonant','medial-consonant'):
    for n in range(1,4):
        alt=section=='medial-consonant' and n in (2,3)
        add(9,section,n,'comparison-example','Aranadan versus explicit Malayalam comparator; shared gloss; alternate target responses in last two medial-consonant rows.' if alt else 'Aranadan versus explicit Malayalam comparator with source gloss.',3 if alt else 2)
for n in range(1,4):add(10,'geminate-consonant',n,'comparison-example','Continuation of page9 explanation: Aranadan/Malayalam pair with shared gloss.',2)
for n,description in enumerate(('Final-vowel source symbols and Kannada-influence claim.','Initial-consonant source symbols and Kannada-influence claim.','Medial-consonant source symbols and alternatives.','Geminated-consonant source symbols; examples continue on next page.'),1):add(9,'phonological-notation',n,'context-symbols',description,0)
for n in range(1,4):add(10,'pronoun-prose',n,'target-pronoun','Two first-person plural forms distinguished by group size, then second-person singular; no invented inclusive/exclusive distinction.')
add(10,'pronoun-prose',4,'comparison-control','Explicit Ka. comparator on second-person singular, not an Aranadan witness.')
for section,count in [('pronoun-examples',3),('inanimate-plurals',2),('dative-examples',2),('genitive-examples',3)]:
    for n in range(1,count+1):add(10,section,n,'whole-translated-expression','Entire printed expression and translation; do not synthesize word-level glosses.')
add(10,'plural-prose',1,'comparison-morphology','First plural-suffix occurrence is explicitly Malayalam.')
add(10,'plural-prose',2,'target-morphology','Second plural-suffix occurrence explicitly Aranadan, limited to animate nouns.')
add(10,'plural-prose',3,'target-quantifier','Standalone quantified adjectival form; no independent lexical English gloss supplied.')
add(10,'dative-prose',1,'morphology-with-negated-distribution','Dative suffix explicitly absent after pronouns; preserve this restriction, no invented suffixed forms.')
for n in range(1,8):add(10,'tense-suffixes',n,'target-morphology','One future, three past, three present markers explicitly listed; retain original hyphen presence and no fabricated conjugation.')
for row in range(1,9):
    for col in ('stem','past','present','future'):add(10,'verb-paradigm',f'{row}:{col}','paradigm-cell',f'Complete 8×4 table, row{row}, {col}; printed stem gloss scopes row; preserve occurrence-specific spellings.')
for n in range(1,7):add(11,'vocabulary',n,'target-lexical-example','Final prose vocabulary item with printed gloss; third cold is explicitly disease.')
for page,section,description in [(9,'introduction','Names, tentative homeland etymology, distribution/population and broad Malayalam/Kannada feature comparison.'),(9,'nasalization-comment','Nasalization claim and Paniya/Adiya comparison without additional quoted forms.'),(9,'length-comment','Aranadan and regional medial-vowel lengthening claim.'),(10,'pronoun-comment','No inclusive/exclusive distinction; large/small group distinction.'),(10,'plural-comment','Animate-only suffix and inanimate quantification; comparator scope.'),(10,'case-comment','Dative and genitive deletion restrictions.'),(10,'tense-comment','Future compared with Malayalam, different past/present; no inference of individual form ancestry.'),(11,'bibliography','Usha1992 attribution; not independently collated here.')]:add(page,section,1,'documentary-context',description,0)
keys=[r['source_unit_key'] for r in units];assert len(keys)==len(set(keys))
(H/'source-unit-inventory.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in units))
hits=[]
for r in source['search_hits']:
    page=r['pdf_page']
    disposition='chapter target/context' if page in (9,10,11) else 'contents' if page==3 else 'unrelated substring/control'
    reason={1:'Somasekharan name in preface; source method context.',2:'Somasekharan dedication.',8:'Sankarankutty in Adiya bibliography.',13:'aran is explicitly a Betta Kurumba masculine noun.',19:'Muralidharan bibliographic attribution, Cholanaikka.',20:'Muralidharan attribution, Cholanaikka.',21:'Muralidharan bibliography.',33:'Somasekharan reference, Kani.',36:'Kurichiya lexical substrings and Kumaran bibliography.',37:'Malamuthan lexical substring neeran.',39:'Somasekharan reference, Malamuthan.'}.get(page,'Aranadan source chapter or contents.')
    hits.append(dict(pdf_page=page,disposition=disposition,reason=reason,original_hits=r['matches']))
report=dict(status='complete bounded scope census; no accepted transcription',source_sha256=source['sha256'],
    chapter_pages=[9,10,11],whole_book_pages_examined_as_native_text=58,inventory_units=len(units),
    by_kind=dict(collections.Counter(r['kind'] for r in units)),physical_witness_slots=sum(r['expected_physical_witness_slots'] for r in units),
    count_note='Slots include explicit comparator witnesses and repeated names; context symbols have no forced lexical count. Final emitted count requires independent reading and disposition.',
    outside_chapter_search_adjudication=hits,
    primary_url_evidence={'in_original':'http://www.languageinindia.com/','exact_document_url':'Not embedded in original; no network request in this task.'},
    license_evidence={'explicit_reuse_license_found':False,'method':'Full58-page text search for copyright/licence/license/Creative Commons; no matching licence statement. Do not infer CC licensing from public availability.'},
    pending=['Independent literal reading of every allocated target/control/morphology occurrence.','Native legacy-font glyph interpretation, especially private-use symbols and underlining.','Bibliographic attribution/license review by assigned independent reviewer.','Final language/control and source-claim/graph decisions; no automatic borrowing from influence prose.','Complete staged output, fresh audit, profiles/metadata/parser/tests before installation.'])
(H/'scope-status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('inventory_units','by_kind','physical_witness_slots')},indent=2))
