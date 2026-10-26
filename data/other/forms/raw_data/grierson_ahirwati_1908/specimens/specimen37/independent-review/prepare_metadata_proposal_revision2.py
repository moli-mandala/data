"""Complete reviewed specimen proposal; no canonical writes or inferred roots."""
from pathlib import Path
import json,csv,hashlib,collections,unicodedata as ud,ast
from urllib.parse import quote
ROOT=Path.cwd();P=Path(__file__).resolve().parent;S=P.parent;PKG=S.parents[1];OUT=P/'metadata-proposal-revision2'
OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def write(n,rr):(OUT/n).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
freeze=json.loads((S/'first-reading-freeze.json').read_text())
for f,h in freeze['files'].items():assert sha(S/f)==h,f
review=P/'independent-full-original-review.json'
assert sha(review)=='25edc7bb45e9240f1ff6a8789f1ab0c12b56c754ddfde923c52159c40932577f'
roman=rows(S/'roman-atoms-first-reading.jsonl');native=rows(S/'native-atoms-first-reading.jsonl')
rm={r['key']:r for r in roman};nm={r['key']:r for r in native}
align=rows(S/'script-alignment-proposal.jsonl');wholes=rows(S/'whole-utterances-first-reading.jsonl');joins=rows(S/'physical-break-joins-proposal.jsonl')
joins.append({'emission_key':'lsi9.2:p235:aligned-line:6:word:10','physical_keys':['lsi9.2:p235:aligned-line:6:word:10','lsi9.2:p235:aligned-line:7:word:1'],'source_form':'‘achchhā-taĩ-achchhā','source_analysis':rm['lsi9.2:p235:aligned-line:6:word:10']['raw_gloss']+rm['lsi9.2:p235:aligned-line:7:word:1']['raw_gloss'],'scope':'Fourth original-backed physical line-break continuation; both target and English have trailing hyphen before newline.'})
head={r['emission_key']:r for r in joins};tail={r['physical_keys'][1] for r in joins}
grammar_path=PKG/'grammar/child-witnesses-reviewed.jsonl';grammar=rows(grammar_path);assert len(grammar)==98
assert sha(grammar_path)=='5ac645b13ae1a3132ac094431fbce9376037705f93c0c59d9861e6e7ef999ee7'
GRAM={r['witness_key']:r for r in grammar}
identity_path=PKG/'scope/historical-identity-proposal.json'
identity=json.loads(identity_path.read_text());lect=identity['new_lects'][0]
assert identity['base']['Name']=='Ahirwadi' and lect['ID']=='lsi1908_ahirwati_gurgaon'
DIALECT='dialect:'+lect['Language_ID']+':'+lect['ID']+':'+quote(lect['Name'],safe='')
PERSON={'i':['pron','personal','1sg','first-person'],'me':['pron','personal','1sg','first-person'],
 'my':['pron','poss','1sg','first-person'],'he':['pron','personal','3sg','third-person'],
 'him':['pron','personal','3sg','third-person'],'his':['pron','poss','3sg','third-person'],
 'thou':['pron','personal','2sg','second-person'],'thee':['pron','personal','2sg','second-person'],
 'thy':['pron','poss','2sg','second-person'],'thine':['pron','poss','2sg','second-person'],
 'mine':['pron','poss','1sg','first-person'],'your':['pron','poss','second-person'],'we':['pron','personal','1pl','first-person'],
 'them':['pron','personal','third-person','pl']}
NOUN={'man','sons','son','father','property','share','way','days','fortune','expenditure','country','famine','landlord','fields','swine','husks','belly','senses','house','labourers','bread','god','clothes','hand','ring','feet','shoes','merriment','opinion','birth','field','music-etc.','brother','feast','years','service','sayings','she-goat','young-one','companions','feasting','harlots','pity','caress'}
PLURAL={'sons','days','fields','husks','labourers','clothes','feet','shoes','years','sayings','companions','harlots'}
PAST={'said','was-given','went-away','wasted','made-completely','fell','became','he-engaged-himself','was-sent','it-was-wished','a-giving-was','it-was-said','remained','he-started','came','attached','was-made','was-lost','began','he-came','he-spoke','went','he-was-appeased','was-squandered-away','was-done','lost-was','might-have-done'}
ADJ={'the-younger','a-few','a-great','a-rich','indigent','hungry','such','good','than-good','worthy','elder','safe-and-sound','angry','dear','great'}
ADV={'then','after','there','there-even','that-very','surely','now','again','like-even','yet','far','much','in-opinion','rejoicing','inside','outside','ever','near','ever-from','since','so-many'}
CONJ={'that','and','when','because','because-that','but'}
def clean(s):return s.strip(' .;,?!‘’“”"').lower()
def native_span(nn):
    out=''
    for i,q in enumerate(nn):
        continuation=i>0 and nn[i-1]['raw_native'].endswith('-') and q['physical_line']>nn[i-1]['physical_line']
        out+=('' if not out or continuation else ' ')+q['raw_native']
    return out
def metadata(form,gloss,source_rows):
    g=clean(gloss);f=clean(form);tags=[];notes=[];holds=[];evidence=[]
    base=g
    if g=='feet-on (in)':base='feet';tags+=['noun','pl','loc'];evidence.append('Explicit parenthetical in with target pāwā̃-mē̃ locative expression.')
    if g=='music-etc.(obj.)':base='music-etc.';tags+=['noun','obj'];evidence.append('Source explicitly prints obj. in the interlinear gloss; unlike English me/him this is an actual source function label.')
    # Source-aligned relational glosses; source chapter explicitly defines kō,
    # nē̃ and locative agreement. English obj, near/for/with do not imply cases.
    suffix=None
    for e in ['to','of','in','on','by','for','near','with','from','since']:
        if base.endswith('-'+e):base=base[:-len(e)-1];suffix=e;break
    for prefix in ['the-','a-','on-','in-']:
        if base.startswith(prefix) and base[len(prefix):] in NOUN|set(PERSON):base=base[len(prefix):]
    if base.split('-')[0] in PERSON and all(q in {'in','from','even','only'} for q in base.split('-')[1:]):base=base.split('-')[0]
    if base.startswith('by-') and base[3:] in PERSON:
        tags.extend(PERSON[base[3:]]);notes.append('The printed interlinear analysis explicitly identifies an agent/by role.');evidence.append('Source agent role retained without an inferred modern ergative/instrumental split.')
    if base in PERSON:tags.extend(PERSON[base])
    elif base in NOUN:
        tags.append('noun')
        if base in PLURAL:tags.append('pl')
        if base in {'father','son','sons','brother'}:tags.append('kinship')
    if suffix=='of':tags.append('gen');evidence.append('p51 explicitly defines genitive kō; occurrence interlinear of confirms relational function.')
    if suffix=='to' and any(x in f for x in ['nē̃','taĩ']):tags.append('dat');evidence.append('p51 defines dative/agent nē̃; this occurrence is independently glossed to.')
    if suffix in {'in','on'} and ('mē̃' in f or f=='ghari'):
        tags.append('loc');evidence.append('Source locative chapter p51 and this exact interlinear in/on scope support locative.')
    if suffix=='by':notes.append('The interlinear source gives an agent/by analysis here; the chapter says nē̃ also forms the agent case.');evidence.append('Agent is source terminology; no modern ergative-versus-instrumental distinction forced.')
    if suffix in {'for','near','with','from','since'}:
        holds.append({'type':'source-case-scope','detail':f'Printed {suffix} meaning retained; intrinsic case is not established for the complete form by English relation alone.'})
    if g in PAST:tags+=['verb','pret']
    if g in {'was','were'}:tags+=['verb','copula','pret']
    if g in {'is','art'}:tags+=['verb','copula','pres']
    if g in {'give','keep','put-on','put','see'}:tags+=['verb','impv'];evidence.append('Exact occurrence is a direct quoted instruction, not English ambiguous past put.')
    if g in {'not'}:tags.append('negator')
    if g=='a-certain':tags+=['determiner','indef']
    if g=='also':tags.append('adv')
    if g in {'those','by-this-one','in-that-very'}:tags+=['pron','demonstrative']
    if g=='those':tags.append('pl')
    if g=='in-that-very':tags.append('emph')
    if g=='which-with':tags+=['pron','relative'];holds.append({'type':'source-case-scope','detail':'Printed which-with meaning is retained; a specific target case is not asserted solely from English with.'})
    if g=='on-neck':tags.append('noun');holds.append({'type':'source-case-scope','detail':'The source gives on-neck; exact nominal declension/locative analysis of galē is not supplied here, so no case guessed from English on.'})
    if g=='merry-making':tags.append('noun')
    if g=='good-than-good':tags.append('adj')
    if g in {'going','on-seeing-even'}:tags.append('verb');holds.append({'type':'nonfinite-subtype','detail':'Printed nonfinite sense is retained; no participial subtype is imposed from English -ing alone.'})
    if g in {'spoke','said-it-was','he-was-attached','was-made-completely'}:tags+=['verb','pret']
    if g=='it-was-given':tags+=['verb','pret','pass']
    if g in {'he-was-attached','was-made-completely'}:tags.append('pass')
    if g in ADJ:tags.append('adj')
    if g in ADV:tags.append('adv')
    if g in CONJ:tags.append('conj')
    if g in {'all','how-many-even','so-many'}:tags.append('quantifier')
    if g in {'two','one'}:tags.append('num')
    if g in {'this','that'}:tags+=['pron','demonstrative']
    if g=='which':tags+=['pron','relative']
    if g=='what':tags+=['pron','interr']
    if g in {'anybody','anything','whatever'}:tags+=['pron','indef']
    if g in {'my-own','his-own','thy-own'} or f in {'apᵃṇū','apᵃṇō','apᵃṇā','apᵃṇē'}:
        tags+=['pron','poss','refl'];notes.append('Grierson gives apᵃṇū/apᵃṇō as own and apᵃṇā as the masculine oblique; this specimen interpretation is retained.');evidence.append('p51 own-pronoun-alternatives; no personal possessor inferred from source apᵃṇ- alone.')
    if g in {'to-make','to-be-called'}:tags+=['verb','inf']
    if g in {'will-go','i-will-say','shall-make'}:tags+=['verb','fut']
    if g in {'may-be','he-may-fill','i-may-be-called','may-eat'}:tags+=['verb','modal']
    if g in {'dying','feeding-for','an-eating-were','wandering-am','doing-am','to-be-given','to-be-done','taken-is','found-is','come-is','done-is','have-done','disobeyed-is','(he) found-is'}:tags.append('verb')
    if 'verb' in tags and (g.startswith('i-') or g in {'wandering-am','doing-am'}):tags+=['1sg','first-person']
    if 'verb' in tags and (g.startswith('he-') or g.startswith('(he)')):tags+=['3sg','third-person']
    if 'verb' in tags and g=='art':tags+=['2sg','second-person']
    # Source printed having/agent interpretations are preserved without
    # turning all English -ing or having strings into target participles.
    if 'having' in g:tags.append('verb');holds.append({'type':'nonfinite-subtype','detail':'Printed nonfinite English analysis is secure; no target participial subtype inferred from the English alone.'})
    if f=='ghari':
        tags+=['noun','loc'];evidence.append('p51 exact ghari is a locative noun.');notes.append('Grierson explicitly gives ghari as the locative of the consonant-final noun, with locative agreement in mhārē ghari.')
    if f=='mhārē':
        tags+=['pron','poss','loc'];evidence.append('p51 exact possessive locative agreement; not merely English on-my.');notes.append('The chapter contrasts locative mhārē with mhārō in mhārē ghari.')
    if f=='karᵃṇō':
        tags+=['verb','participle','pass','fut'];evidence.append('p51 agent-passive-whole-example exact karᵃṇō.');notes.append('Grierson calls karᵃṇō a future passive participle, identical in form to the infinitive, in tū-nē̃ karᵃṇō thō.')
    if f=='dīṇū̃':
        tags+=['verb','pass'];holds.append({'type':'related-grammar-quantity','detail':'Specimen dīṇū̃ differs in initial quantity from grammar diṇū̃; do not automatically transfer the occasional-neuter analysis.'})
    if f in {'thō','thā'}:
        tags+=['verb','copula','pret'];evidence.append('p51 past-copula source class; occurrence gloss controls tense.')
        if f=='thā':tags+=['m','pl']
    if not tags:holds.append({'type':'lexical-function-not-supplied','detail':'Literal and interlinear meaning remain; no confident lexical POS or target morphology forced.'})
    for q in source_rows:
        holds.extend({'type':'source-glyph','detail':h} for h in q.get('typed_uncertainties',[]))
    if holds:tags.append('uncertain')
    return list(dict.fromkeys(tags)),list(dict.fromkeys(notes)),holds,evidence
candidate=[];audit=[];meta=[];aliases=[]
def emit(key,form,gloss,nat,tags,notes,rr,nn,kind,holds,evidence,old_key=None):
    tags=list(dict.fromkeys(tags+[DIALECT]))
    pages=sorted({r['printed_page'] for r in rr});loc=' and '.join(f'p. {p}' for p in pages)
    row=['Ahirwadi',nat,form,gloss,form,'',' '.join(notes),f'grierson1908ahirwati[{loc}, specimen 37]','','',key,'','','',' '.join(tags)]
    assert len(row)==15;candidate.append(row)
    rec={'source_unit_key':key,'kind':kind,'row':row,'roman_physical_keys':[r['key'] for r in rr],
      'native_physical_keys':[r['key'] for r in nn],'source_record':{'Roman':rr,'Native':nn},
      'typed_uncertainties':holds,'grammar_evidence':evidence,'printed_interlinear_analysis':gloss,
      'translation_type':'source interlinear analysis, not author-supplied free translation','old_candidate_key':old_key,
      'phonemic_blank_reason':'Literal LSI source transcription retained; symbol-specific phonological values/profile not yet independently established.','accepted_for_installation':False}
    audit.append(rec);meta.append({'source_unit_key':key,'exact_form_guard':form,'exact_gloss_guard':gloss,'tags':tags,'notes':notes,'typed_uncertainties':holds,'grammar_evidence':evidence})
shared_groups=[]
for a in align:
    rr=[rm[k] for k in a['roman_keys']];nn=[nm[k] for k in a['native_keys']]
    effective=[q for q in rr if q['key'] not in tail]
    nat=native_span(nn);assert nat==a['native_form']
    shared=len(effective)>1
    group_key=rr[0]['key']+':script-span'
    if shared:
        shared_groups.append({'source_group_key':group_key,'historical_alignment_key':a['key'],
          'roman_physical_keys':a['roman_keys'],'native_physical_keys':a['native_keys'],
          'lexical_emission_keys':[q['key'] for q in effective],'full_native_form':nat,
          'raw_alignment_record':a,'policy':'Shared Native span retained as group evidence; no arbitrary per-Roman-word Native assignment.'})
    for q in effective:
        keys=head[q['key']]['physical_keys'] if q['key'] in head else [q['key']]
        qr=[rm[k] for k in keys]
        form=head[q['key']]['source_form'] if q['key'] in head else q['raw_form']
        gloss=head[q['key']]['source_analysis'] if q['key'] in head else q['raw_gloss']
        tags,notes,holds,evidence=metadata(form,gloss,qr)
        if shared:holds.append({'type':'cross-script-segmentation','detail':'Native is supplied for the shared group, not separately aligned to this Roman word.'});tags.append('uncertain')
        if ' ' in form:tags.append('multiword-expression')
        emit(q['key'],form,gloss,'' if shared else nat,list(dict.fromkeys(tags)),notes,qr,nn,'Roman-lexical-emission',holds,evidence)
        if shared:audit[-1]['shared_native_group_key']=group_key
        for r in qr:aliases.append({'physical_key':r['key'],'emission_key':q['key'],'role':'Roman physical witness; only explicitly reviewed printer continuations joined'})
    for q in nn:aliases.append({'physical_key':q['key'],'emission_key':None if shared else effective[0]['key'],'source_group_key':group_key if shared else None,'role':'Native physical witness; shared attribution group' if shared else 'Native physical witness; full exact lexical span'})
write('shared-native-groups.jsonl',shared_groups)
write('reviewed-printer-continuations.jsonl',joins)
for w in wholes:
    rr=[rm[k] for k in w['roman_word_keys']];nn=[nm[k] for k in w['native_word_keys']]
    key=rr[0]['key']+':expression';holds=[{'type':'source-glyph','detail':u} for u in w['typed_uncertainties']]
    tags=['sentential'];
    if holds:tags.append('uncertain')
    emit(key,w['proposed_whole_form'],w['printed_interlinear_analysis'],native_span(nn),tags,[],rr,nn,'whole-source-expression',holds,
         ['Original sentence/speech punctuation reviewed; nested quotation overlaps retained as independent source spans.'],w['key'])
assert len(candidate)==390 and len({r[10] for r in candidate})==390
assert collections.Counter(a['physical_key'] for a in aliases if ':aligned-line:' in a['physical_key'])==collections.Counter(rm.keys())
assert collections.Counter(a['physical_key'] for a in aliases if ':native-line:' in a['physical_key'])==collections.Counter(nm.keys())
registered=set()
for node in ast.walk(ast.parse((ROOT/'data/tags.py').read_text())):
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'GRAMMATICAL_TAGS','GENDER_TAGS'} for t in node.targets):registered|=ast.literal_eval(node.value)
assert all(set(r[14].split())-{DIALECT}<=registered for r in candidate)
with (OUT/'candidate.csv').open('w',newline='') as f:csv.writer(f).writerows(candidate)
write('candidate-audit.jsonl',audit);write('all-390-metadata-proposal.jsonl',meta);write('physical-key-aliases.jsonl',aliases)
unforced=[{'source_unit_key':r['source_unit_key'],'exact_form_guard':r['exact_form_guard'],'exact_gloss_guard':r['exact_gloss_guard']} for r in meta if any(h['type']=='lexical-function-not-supplied' for h in r['typed_uncertainties'])]
write('unforced-lexical-function-scope.jsonl',unforced)
(OUT/'routing-and-literal-policy-proposal.json').write_text(json.dumps({'base':identity['base'],'lect':lect,'dialect_tag':DIALECT,
 'transcription':{'convert':False,'preserve_hyphens':True},'split_alternates':False,'gloss_split_alternates':False,
 'phonemic':'Blank pending independently established symbol values; exact Original and Form preserve LSI marks, raised letters, quantities and punctuation.',
 'profile_status':'Literal symbol inventory covers every emitted Roman/Native codepoint; no automatic mapping to IPA or use of unrelated packages default profiles.',
 'source_key':'grierson1908ahirwati','stage':'proposal only; main source metadata/integration owned by root'},ensure_ascii=False,indent=2)+'\n')
inventory=collections.Counter(c for r in candidate for field in [r[1],r[2]] for c in field)
with (OUT/'literal-symbol-inventory.tsv').open('w') as f:
    f.write('symbol\tcodepoint\tname\tcount\tpolicy\n')
    for c,count in sorted(inventory.items()):f.write(f'{c}\tU+{ord(c):04X}\t{ud.name(c,"unnamed")}\t{count}\tliteral source preservation; no automatic IPA claim\n')
for r in candidate:
    for s in [r[1],r[2]]:assert not any(ud.category(c)=='Co' or c=='\ufffd' for c in s)
report={'status':'complete source-local emission/metadata proposal; independent metadata and final-output audit required','counts':{'candidate_rows':390,'Roman_lexical_emissions':363,'shared_native_groups':len(shared_groups),'whole_expressions':27,'Roman_physical_atoms':367,'Native_physical_atoms':365,'aliases':len(aliases),'joins':4,'grammar_interpretations_consulted':98},
 'generator_sha256':sha(Path(__file__)),
 'input_hashes':{str(p.relative_to(PKG)):sha(p) for p in [S/'first-reading-freeze.json',review,grammar_path,identity_path]},
 'outputs':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='status.json'},
 'policies':['Every Roman physical atom maps exactly once to a lexical emission; every Native physical atom maps exactly once to a lexical emission or explicit shared group; nested whole speech adds source expressions, not duplicated physical attestations.',
 'All27 expression keys anchor first immutable physical atom; prior ordinal proposal keys retained only as audit history.',
 'Full group Native preserved without guessed perword alignment; all four reviewed printer joins and segmentation exceptions retained.',
 'No graph edges, inferred roots, English object-case tags, or automatic -ing/participle extraction.',
 'Source identity follows proposed Ahirwadi/Gurgaon historical route; source lect registration and final source key remain integration decisions.',
 'No canonical writes or phonemic inference; literal-profile inventory is preliminary source policy evidence.']}
(OUT/'status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'counts':report['counts'],'status_sha256':sha(OUT/'status.json')}))
print('UNFORCED LEXICAL FUNCTIONS',sorted({r['exact_gloss_guard'] for r in unforced}))
