"""Bounded source-only Bagri comparison inventory, with explicit scope roles."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[6]
PACKAGE=HERE.parent
sys.path.insert(0,str(ROOT/'data'))
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value,jsonl=False):
    text=(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in value) if jsonl else json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    p=HERE/name
    if p.exists():assert p.read_text()==text, f'Preserve immutable first reading: {name}'
    else:p.write_text(text)
images={r['printed_page_expected']:r for r in json.loads((HERE/'native-originals-manifest.json').read_text())['pages']}
for r in images.values():assert sha(ROOT/r['lossless_png_path'])==r['lossless_png_sha256']
rows=[];parents=[]
def parent(key,page,scope,forms,glosses=None,tags=None,site='standard Bikaner',role='contextual-comparison',note='',uncertainty=None,raw_native=None):
    glosses=glosses or ['']*len(forms);tags=tags or [[] for _ in forms]
    assert len(forms)==len(glosses)==len(tags)
    keys=[]
    for rank,(form,gloss,tt) in enumerate(zip(forms,glosses,tags),1):
        keyw=f'{key}:w{rank}';keys.append(keyw)
        assert all(t in GRAMMATICAL_TAGS|GENDER_TAGS for t in tt),tt
        u=uncertainty.get(rank,[]) if uncertainty else []
        rows.append({'key':keyw,'parent_key':key,'printed_page':page,'source_form':form,
          'source_gloss':gloss,'proposed_tags':tt+(['uncertain'] if u else []),
          'source_language_label':'Bāgrī','source_site_label':site,'source_type_role':role,
          'source_note':note,'typed_uncertainties':u,'raw_native':raw_native.get(rank) if raw_native else None,
          'image':images[page]['lossless_png_path'],'image_sha256':images[page]['lossless_png_sha256'],
          'stage':'complete owner source first reading; independent original/identity review pending'})
    parents.append({'key':key,'printed_page':page,'scope':scope,'child_keys':keys,
      'physical_group_count':1,'source_type_role':role,'source_note':note})

# Initial three phonology paragraphs form the bounded Ahirwati-related argument.
parent('lsi9.2:p149:outside-lead-scope:genitive-spellings',149,'Written versus pronounced genitive suffix, before medial-h and b/w discussion',
 ['gō','kō','kā̃','gō'],tags=[['suffix','gen']]*4,
 note='Source says the written kō/kā̃ spellings are pronounced gō. These are quoted written forms versus pronunciation, not independent donor claims.',
 uncertainty={3:['Quantity/nasal glyph above final a is provisionally read ā̃; alternative ā retained for independent narrow reading.']},
 raw_native={1:'गो',2:'को',3:'काँ'})
parent('lsi9.2:p149:outside-lead-scope:medial-h-examples',149,'Three source-explicit medial-h comparisons in standard Bagri description',
 ['kasū̃','kahᵃsū̃','kayō','kahyō','chāyō','chāhyō'],
 ['I will say','I will say','said','said','he wished','he wished'],[['verb','fut','1sg']]*2+[['verb','pret']]*2+[['verb','pret','3sg']]*2,
 note='The paragraph says the medial-h loss is as in Standard Marwari. This analogy does not independently label the six printed forms as Marwari tokens.',
 uncertainty={2:['Small raised character after h is provisionally a; literal superscript retained rather than inferred schwa.']})
parent('lsi9.2:p149:outside-lead-scope:b-w-pronoun',149,'Bikaner b versus w comparison, also said to occur in Sirsa',
 ['bō','wō'],['he','he'],[['pron','3sg']]*2,
 note='Source: Bikaner often prefers initial b; Wilson observed the same peculiarity in Sirsa. In other Bagri areas under Panjabi, Bangaru or Ahirwati influence w/v is retained. One printed pair, not repeated printing for every area.')
parent('lsi9.2:p149:outside-lead-scope:letter-notation',149,'All nonlexical sound/letter symbols in the same three phonology paragraphs',
 ['k','g','ō','au','h','b','w','v','w','v'],role='documentary-sound-notation',
 note='Letter/sound comparisons and the pronunciation tendency of ō; no lexical head, IPA interpretation or morphological donor inferred.')

person=[('p1','sg',['1sg','sg','first-person']),('p1','pl',['1pl','pl','first-person']),
        ('p2','sg',['2sg','sg','second-person']),('p2','pl',['2pl','pl','second-person']),
        ('p3','sg',['3sg','sg','third-person']),('p3','pl',['3pl','pl','third-person'])]
standard=['hū̃','hā̃','hai','hō','hai','hai']
sirsa=[['sū̃'],['sā̃'],['sai','sē'],['sō'],['sai','sē'],['san']]
for (p,n,tt),form,forms in zip(person,standard,sirsa):
    parent(f'lsi9.2:p151:outside-standard-present:{p}:{n}',151,'Surrounding standard Bagri auxiliary present table',
      [form],tags=[['verb','copula','pres']+tt],note='Context control: the author says this follows Marwari and explicitly notes third plural is not nasalized.')
    parent(f'lsi9.2:p151:outside-Bagri-present:{p}:{n}',151,'Sirsa auxiliary present table; one physical cell with source alternatives',
      forms,tags=[['verb','copula','pres']+tt]*len(forms),site='Sirsa and other affected parts',role='eligible-attributed-comparison',
      note='Source says these occur in Sirsa and other parts affected by Bangaru OR Ahirwati. No unique target origin or borrowing edge.')
for n,g,form in [('sg','m','hō'),('pl','m','hā'),('sg','f','hī'),('pl','f','hī')]:
    parent(f'lsi9.2:p151:outside-standard-past:{g}:{n}',151,'Standard Bagri past auxiliary table',
      [form],tags=[['verb','copula','pret',n,g]],note='Context control; the following Hissar paragraph is a distinct source occurrence.')
parent('lsi9.2:p151:outside-Bagri-past-group',151,'Three Hissar auxiliary past forms in continuous comparison paragraph',
 ['thō','thā','thī'],tags=[['verb','copula','pret']]*3,site='Hissar and other affected parts',role='eligible-attributed-comparison',
 note='Source explicitly attributes the forms to Hissar and other parts affected by Bangaru AND Ahirwati. Number/gender assignments are not separately printed for this inline group; no automatic pairing to preceding table cells.')

# Account later same-page Hissar leads, but keep broader standard paradigms context-only.
finite=['mārū̃','mārā̃','mārē','mārō','mārē','mārē']
definite=['mārū̃-hū̃','mārā̃-hā̃','mārē-hai','mārō-hō','mārē-hai','mārē-hai']
for (p,n,tt),f,d in zip(person,finite,definite):
    parent(f'lsi9.2:p151:context-finite-present:{p}:{n}',151,'Standard Bagri present finite table',
      [f],['strike'],[['verb','pres','ind']+tt],note='Context control after the explicit auxiliary comparison; printed head I strike etc. supplies lexical gloss, grammatical labels remain Tags.')
    parent(f'lsi9.2:p151:context-definite-present:{p}:{n}',151,'Standard Bagri definite present table',
      [d],['strike'],[['verb','pres','definite']+tt],note='Source says formation uses the preceding finite tense, not the present participle; no inferred component edges.')
parent('lsi9.2:p151:context-Hissar-present-third-plural',151,'Explicit same-page Hissar third-person plural contrast',
 ['mārē̃'],['strike'],[['verb','pres','ind','3pl','pl','third-person']],site='Hissar',role='additional-source-scope-lead',
 note='Source-explicit Hissar form in later finite paragraph; retained for root scope adjudication rather than silently excluded or relabelled Ahirwati.')
for n,g,form in [('sg','m','mārē-hō'),('pl','m','mārē-hā'),('sg','f','mārē-hī'),('pl','f','mārē-hī')]:
    parent(f'lsi9.2:p151:context-imperfect:{g}:{n}',151,'Standard Bagri imperfect table',
      [form],['strike'],[['verb','pret','ipfv',n,g]],note='Source says imperfect does not change for person; no first-person tags inferred from illustrative heading I was striking etc.')
parent('lsi9.2:p151:context-imperfect-verbal-noun-marker',151,'Explicit verbal noun marker in imperfect formation statement',
 ['ē'],tags=[['suffix']],role='documentary-morphology-context',note='Source verbal noun in ē; no printed hyphen added or independent lexical verb tag.')
parent('lsi9.2:p151:context-Hissar-imperfect-example',151,'Whole Hissar imperfect source example',
 ['hū̃ mārᵃtō-thō'],['strike'],[['verb','pret','ipfv','1sg','sg','first-person','multiword-expression']],
 site='Hissar and the neighbourhood',role='additional-source-scope-lead',
 note='Source says Hissar uses the present participle as in Hindustani. Whole printed expression retained; no fabricated independent component or cross-language ancestry edges.',
 uncertainty={1:['Small raised a in the participle requires independent occurrence control; no phonemic schwa substitution.']})

assert len({r['key'] for r in rows})==len(rows)
assert sum(len(p['child_keys']) for p in parents)==len(rows)
core=[r for r in rows if r['source_type_role']=='eligible-attributed-comparison']
assert len(core)==11 # eight present alternatives plus three past forms
assert not any(r['source_form']=='saĩ' for r in rows)
save('child-witnesses-first-reading.jsonl',rows,True)
save('parent-coverage-first-reading.jsonl',parents,True)
routes={'registered_base':{'id':'bagri_fatehabad','name':'Bagri','glottocode':'bagr1243'},
 'proposed_historical_lects':[{'id':'lsi1908_bagri_sirsa','language':'bagri_fatehabad','name':'Sirsa (LSI1908 Bagri)','latitude':'','longitude':'','glottocode':''},
                            {'id':'lsi1908_bagri_hissar','language':'bagri_fatehabad','name':'Hissar (LSI1908 Bagri)','latitude':'','longitude':'','glottocode':''}],
 'guards':['Existing hissar is registered under kaithal/Haryanvi; geography alone cannot assign these Bagri witnesses to it.',
           'Bikaner and unlocalized standard Bagri controls require separately justified routing if accepted for emission; do not attach Sirsa/Hissar indiscriminately.',
           'p51 saĩ borrowing claim remains source analysis with no exact printed donor endpoint in this inventory.',
           'bō/wō printed pair has one shared physical scope; source says Sirsa shares the peculiarity without printing another pair.',
           'Neither alternative influence nor morphology formation authorizes ancestry/derivation edges.'],
 'stage':'proposed source-local routes; no registry writes or lexical acceptance'}
save('identity-and-provenance-proposal.json',routes)
save('first-reading-status.json',{'status':'complete bounded outside first reading; independent review and scope/route approval pending',
 'counts':{'parent_groups':len(parents),'child_interpretations':len(rows),'Sirsa_present_physical_cells':6,
           'Sirsa_present_literal_alternatives':8,'Hissar_past_physical_inline_group':1,'Hissar_past_forms':3,
           'additional_Hissar_scope_leads':2},
 'roles':{role:sum(r['source_type_role']==role for r in rows) for role in sorted({r['source_type_role'] for r in rows})},
 'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [HERE/'native-originals-manifest.json',HERE/'original-control-provenance.json',PACKAGE/'scope/ROOT_IDENTITY_DECISION.md',ROOT/'data/cldf/languages.csv',ROOT/'data/cldf/dialects.csv']},
 'outputs':{name:sha(HERE/name) for name in ['child-witnesses-first-reading.jsonl','parent-coverage-first-reading.jsonl','identity-and-provenance-proposal.json']},
 'script_sha256':sha(Path(__file__)),
 'boundary_review':{'149':'Full three opening phonology paragraphs through b/w/v contrast and source-geographic influence discussion; noun declension begins a separate unrelated block and is not expanded.',
                   '151':'Complete auxiliary comparison block plus explicit later Hissar source leads. All intervening standard tables recorded as context controls, not silently target-routed.'},
 'limits':['No source hypotheses converted into donor nodes; no p51 borrowing graph.',
           'Counts refer to source interpretations/groups, not a fabricated count of independent printed lexical tokens.',
           'Raw native/Latin genitive spellings retained together as paired script evidence.',
           'No canonical settings, registry, bibliography, data or installed audits changed.']})
print(json.dumps({'parents':len(parents),'children':len(rows),'status_sha256':sha(HERE/'first-reading-status.json')}))
