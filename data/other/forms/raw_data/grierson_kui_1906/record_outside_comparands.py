"""Source-local scoped comparison census, never canonical installation."""
import csv,hashlib,json,unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
rows=[]
def add(page,label,form,gloss='',tags=(),note='',status='attested comparison'):
 n=1+sum(r['printed_page']==page for r in rows)
 rows.append({'source_unit_key':f'grierson1906kui:p{page}:outside-comparand:{n:02d}','printed_page':page,'djvu_physical_page':229 if page==210 else page+20,'source_language_label':label,'form':form,'gloss':gloss,'tags':list(tags),'note':note,'status':status,'graph_edges':[]})
add(210,'Juang','ñge',tags=['suffix','uncertain'],note='Source says possibly originally a first-person suffixed pronoun, but generally used without first-person reference; do not tag confident first-person. Italic g/j typography requires independent confirmation.')
add(210,'Juang','konī-ñge','son',['noun','uncertain'],'Fine original supports a marked i before hyphen; source g/j and quantity are provisional.')
add(210,'Juang','kanu-niñge','son',['noun','uncertain'],'Retain exact second witness; italic g/j still requires independent confirmation.')
add(210,'Juang','bui-ñge','mother',['noun','uncertain'],'Italic g/j provisional. This belongs to the bounded suffix comparison, not all neighboring noun examples.')
add(290,'Tamil (contextual)','aḷ',tags=['suffix','f','uncertain'],note='Caldwell feminine-suffix comparison; language inferred from preceding explicit Tamil discussion, not a Kui ending. Preserve contextual attribution.')
add(290,'Telugu','kōḍalu','a daughter-in-law',['noun'])
add(290,'Telugu','āḍu','female')
add(290,'Kurukh','āli','a woman',['noun'])
for label,forms,extra in [('Tamil',['gaḷ'],'') ,('Tamil',['ga','āḷ','ā','āṅg'],'Source explicitly colloquial.'),('Kanarese',['gaḷu'],''),('Kanarese',['gāḷ','gāḍ','l','lā'],'Source explicitly colloquial; gāḷ/gāḍ upper quantities compressed.'),('Gondi',['k','ṅg','sk'],''),('Telugu',['lu'],''),('Brahui',['āk','k'],'')]:
 for form in forms:add(291,label,form,tags=['suffix','pl']+(['colloquial'] if 'colloquial' in extra else []),note='Full source plural comparison list. '+extra)
add(291,'Telugu','tammuḍu','a brother',['noun'])
add(291,'Telugu','*tammunu','a brother',['noun','uncertain'],'Printed starred hypothetical earlier form, not an independently attested lexical witness. Keep visible star and source attribution.','source hypothetical form')
add(291,'Telugu','tammuni',tags=['gen','obl'],note='Source labels this genitive and oblique base corresponding to brother. No invented lexical translation.')
for label,form,extra in [('Tamil','nīr',''),('Tamil','nīṅgaḷ',''),('Tuḷu','īr',''),('Telugu','mīru',''),('Telugu','īru','Explicitly old Telugu.')]:
 add(293,label,form,'you',['pron','2pl']+(['archaic'] if extra else []),extra)
for label,forms,scope in [('Tamil',['nāṅgaḷ'],'exclusive'),('Maḷyāḷam',['naṅṅaḷ'],'exclusive'),('Kurukh',['ēm'],'exclusive'),('Telugu',['ēmu','mēmu'],'exclusive'),('Tamil',['yām','nām'],'inclusive'),('Maḷyāḷam',['nām'],'inclusive'),('Kurukh',['nām'],'inclusive'),('Telugu',['manamu'],'inclusive')]:
 for form in forms:add(293,label,form,'we',['pron','1pl',scope],'Two-row comparison table. Fine reading compared with original and separately keyed Nordic table transcription; source variants remain separate alternatives, no modern spelling substituted.')
for label,form,gloss,tense in [('Tamil','avaṉ aḍeigiṟāṉ','he gets','pres'),('Kanarese','avanu māḍidanu','he did','pret'),('Gondi','ōr kītōr','he did','pret'),('Telugu','vāḍu koṭṭināḍu','he struck','pret')]:
 add(295,label,form,gloss,['sentential','verb','3sg',tense], 'Whole printed clause in the personal-suffix comparison; no component translations inferred. Tamil underbar consonants retained; dense e/i sequence merits independent fine review.')
for label,form,gloss,person in [('Kanarese','māḍ-uttā-ne','he does','3sg'),('Kurukh','es-d-an','I break','1sg'),('Gondi','nannā kīā-t-ōn-ā','I do','1sg'),('Telugu','nēnu koṭṭu-t-unnānu','I strike','1sg'),('Telugu','nēnu koṭṭu-d-unu','I would strike','1sg')]:
 add(296,label,form,gloss,['sentential','verb',person], 'Whole clause in explicit indefinite-present comparison containing the Kui example; no independent word glosses. Source says indefinite tense often also future, not a second fabricated attestation.')
for label,forms in [('Tamil',['ndu','ttu','chchu']),('Kanarese',['du']),('Kurukh',['k']),('Gondi',['t']),('Telugu',['t']),('Brahui',['k'])]:
 for form in forms:add(296,label,form,tags=['suffix','pret','conjunctive-participle'],note='Explicit conjunctive-participle suffix comparison used to form past; chchu explicitly described as vulgar pronunciation. Source tentative derivation from ku is not a graph edge.')
add(413,'Kurukh','tī',tags=['suffix','abl'],note='Ablative suffix explicitly compared with Kui ṭai; no asserted etymological edge.')
add(413,'Kurukh','n',tags=['infix'],note='Inserted before ablative tī; source distinguishes this from accusative n. Structural insertion, not a lexical word.')
for label,forms in [('Tamil',['iṉṟu','iṉḍe']),('Kanarese',['inda'])]:
 for form in forms:add(413,label,form,tags=['suffix','abl','uncertain'],note='Same ablative comparison paragraph; compressed Tamil underbar/dot distinctions need independent fine review.')
add(413,'Kurukh','gus','with',note='Explicit base to which tī may be added, within same comparison paragraph.')
for form in ['āl-tī','āl-gus-tī']:add(413,'Kurukh',form,'from a man',['noun','abl'],'Two printed alternatives; do not derive new unattested bare forms.')
for form in ['k','ṅg']:add(479,'Gondi',form,tags=['suffix','pl'],note='Direct plural comparison before Kui gā/skā and Korava ṅgu.')
for form,gloss in [('kāl-k','feet'),('mallā-ṅg','mountains')]:add(479,'Gondi',form,gloss,['noun','pl'])
add(479,'Korava (a dialect of Tamil)','ṅgu',tags=['suffix','pl'],note='Retain historical source label; no automatic modern Korvi/Tamil dialect assignment.')
for label,form,gloss in [('Gondi','Saiyuṅg','five'),('Gondi','sārūṅg','six'),('Kanarese','eidu','five'),('Kanarese','ei','five'),('Telugu','eidu','five'),('Kanarese','āru','six'),('Telugu','āru','six'),('Tamil','āru','six')]:
 add(481,label,form,gloss,['num'],'Exact bounded five/six comparison; Gondi source initial capital retained. Compressed vowel/ṅg typography subject to independent review.')
add(481,'Gondi','immā','thou',['pron','2sg','honorific'],'Source states originally plural employed as honorific singular; historical origin is not current plural tag.')
for label in ['Maḷyāḷam','Kanarese']:add(481,label,'nīm','you',['pron','2pl'],'Compared with honorific Gondi immā and Kui īmu; plural interpretation is source comparison context.')
add(481,'Gondi','ōr','he',['pron','3sg'],'Source says originally plural; current gloss he preserved.')
for label,form in [('Tamil','avar'),('Maḷyāḷam','avar'),('Kanarese','avaru')]:add(481,label,form,'they',['pron','3pl'])
add(481,'Gondi','ōn',tags=['pron','obl'],note='Explicit old singular now oblique/verbal base, not an invented standalone English translation.')
add(481,'Kōi (Gondi context)','ōḍu','he',['pron','3sg'],'Source explicitly Kōi witness; preserve attribution without silently assigning unlocalized Gondi.')
add(481,'Telugu','vāḍu','he',['pron','3sg'])
for label,form in [('Tamil','um'),('Telugu','mu')]:add(484,label,form,tags=['suffix','honorific'],note='Exact honorific comparison to Kui mu; surrounding Gondi imperative illustrations are separate from this quoted comparison span.')
for form,gloss,tags in [('avan','he',['pron','3sg']),('avand','he',['pron','3sg']),('avanuṅg','to him',['pron','3sg','dat']),('avandun','to him',['pron','3sg','dat']),('avadan','to him',['pron','3sg','dat'])]:
 add(566,'Parji',form,gloss,tags,'Complete n/nd/d comparison paragraph, followed by Kolami/Kui/Telugu. No phonological normalization.')
add(566,'Kolami','and','he',['pron','3sg'],'He supplied by immediately surrounding pronoun comparison; source labels the preceding-to-this derivation as must be derived, not independently established here.')
add(566,'Kolami','avnd','he',['pron','3sg','uncertain'],'Hypothetical source-derived earlier form; source does not print a star. Do not silently add one.','source hypothetical form')
add(566,'Telugu','vāḍu','he',['pron','3sg'],'Contextual pronoun comparison gloss, not repeated immediately after Telugu form.')
for form,gloss,tags in [('ān','I',['pron','1sg']),('ānun','me',['pron','1sg','acc']),('annē','my',['pron','1sg','gen']),('ām','we',['pron','1pl']),('īn','thou',['pron','2sg']),('inun','thee',['pron','2sg','acc']),('innē','thy',['pron','2sg','gen']),('īm','you',['pron','2pl']),('aun','he',['pron','3sg']),('aunun','him',['pron','3sg','acc']),('aunē','his',['pron','3sg','gen'])]:
 add(570,'Naiki',form,gloss,tags,'Complete immediately compared personal-pronoun table; existing Naiki package must be reused if exact occurrence is installed. Short initial i in inun confirmed independently in earlier Naiki review.')
for form,gloss,tags in [('ānu','I',['pron','1sg']),('āmu','we',['pron','1pl']),('īnu','thou',['pron','2sg'])]:
 add(570,'Kolami',form,gloss,tags,'Source says the same forms are usual in Kolami immediately after explicitly written Kui forms; reference expansion, not separately printed Kolami tokens.')
add(623,'Brahui','oṭ',tags=['suffix','ins'],note='Source says perhaps connected with following from-suffixes; do not replace instrumental analysis with ablative or infer an ancestry edge.')
for label,forms in [('Tamil',['iṉṟu']),('Korava',['iṇḍe','uṇḍ']),('Kurukh',['tī']),('Kōi',['aggāḍu'])]:
 for form in forms:add(623,label,form,'from',['suffix','abl']+(['uncertain'] if label=='Korava' else []),'Complete comparison list. Korava n-mark and Kōi medial quantities require independent detail adjudication; source perhaps-connection remains qualified.')
for label,form in [('Tamil','āḍ'),('Gondi','yēṭi'),('Brahui','hēṭ')]:add(623,label,form,'a goat',['noun'],'Explicit goat comparison in same paragraph; separate from instrumental suffix comparison.')
for label,forms in [('Brahui',['nā','ā']),('Burgandi',['e','ne']),('Naiki',['nē']),('Kolami',['nē']),('Gondi',['ā'])]:
 for form in forms:add(623,label,form,tags=['suffix','gen'],note='Explicit genitive comparison. Burgandi is a source lect label requiring registry identification; not silently promoted to a modern language.')
add(623,'Brahui','ṭi',tags=['suffix','loc'],note='Locative suffix compared by Dr Trumpp with Tamil place and Kui ta.')
add(623,'Tamil','iḍei','place',['noun'],'Source spelling retained; no modern idam substitution. The comparison is attributed to Dr Trumpp.')
for form,gloss in [('ō','that, he'),('ē','that, far off')]:add(624,'Brahui',form,gloss,['pron','dem'],'Remote demonstrative comparison; source says two forms as in Kurukh but supplies no specific Kurukh literals.')
add(624,'Tamil','avaṉ','that, he',['pron','dem'],'Contextual gloss in direct comparison to Brahui ō; final source n underbar retained.')
for form,gloss in [('khaniv','I may see'),('khanev','I may see'),('pāv','I may say')]:
 add(624,'Brahui',form,gloss,['verb','1sg'],'Full conjunctive-present examples immediately compared with Malto and Kui; source underlines kh as digraph, encoded k̲h̲ below.')
 rows[-1]['form']=rows[-1]['form'].replace('kh','k̲h̲')
for form,gloss in [('bandin','I draw'),('dargin','I catch')]:add(624,'Malto',form,gloss,['verb','1sg','pres'],'Whole source inflected examples, compared with Kui pāgiĩ and giĩ. Final in unmarked in this witness.')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Search only canonical raw CSV files, stream rows; exact source spelling is required.
index={}
forms={unicodedata.normalize('NFC',r['form']) for r in rows}
for p in sorted((ROOT/'data/data/other/forms').glob('*.csv')):
 with p.open(newline='') as f:
  for r in csv.reader(f):
   if len(r)>=14 and unicodedata.normalize('NFC',r[2]) in forms:
    index.setdefault(unicodedata.normalize('NFC',r[2]),[]).append({'file':str(p.relative_to(ROOT)),'row':r})
for r in rows:
 candidates=index.get(unicodedata.normalize('NFC',r['form']),[])
 exact=[]
 if r['source_language_label']=='Naiki':
  for c in candidates:
   q=c['row']
   if q[0]=='Naiki' and q[3]==r['gloss'] and ((r['printed_page']==570 and ':p570:pronoun-table:' in q[10]) or (r['printed_page']==623 and ':p623:genitive-comparison:naiki' in q[10])):
    exact.append(c)
 r['verified_existing_occurrence_reuse']=exact
 r['exact_spelling_candidate_count']=len(candidates)
 r['reuse_decision']='Reuse exact already-installed source occurrence, retaining full existing analysis and identity.' if exact else 'Keep a separate source attestation. Other identical spellings do not establish identical lect, grammatical analysis or source occurrence.'
 r['review_status']='Complete bounded original-span first reading; independent literal and attribution audit pending.'
 r['source_sha256']='33e9aaa220db22fcde712705581a69f7edcfc7e20b3e7a09347dc969768020f1'
 r['source_citation']=f'grierson1906lsi4[p. {r["printed_page"]}, comparative prose]'
 r['phonemic']=''
(HERE/'outside-comparands-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
assert len({r['source_unit_key'] for r in rows})==len(rows)
assert set(r['printed_page'] for r in rows)=={210,290,291,293,295,296,413,479,481,484,566,570,623,624}
assert sum(bool(r['verified_existing_occurrence_reuse']) for r in rows)==12
evidence={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tmp').glob('kui-outside-*-comparands-*.png'))}
evidence.update({str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'tmp/kui-outside-210-detail.png',ROOT/'tmp/kui-outside-290-detail.png',ROOT/'tmp/kui-outside-291-detail.png',ROOT/'tmp/kui-outside-291-young-man-restored-v2.png',ROOT/'tmp/kui-outside-566-detail.png',ROOT/'tmp/kui-outside-alternate645.jpg']})
(HERE/'outside-comparands-status.json').write_text(json.dumps({'status':'all fourteen bounded comparison spans first-read; independent acceptance pending, no installation','count':len(rows),'verified_exact_existing_occurrence_reuses':12,'source_sha256':sha(ROOT/'tmp/pdfs/LSI-V4.djvu'),'covered_pages':sorted({r['printed_page'] for r in rows}),'per_page_counts':{str(p):sum(r['printed_page']==p for r in rows) for p in sorted({r['printed_page'] for r in rows})},'evidence_sha256':evidence,'output_sha256':sha(HERE/'outside-comparands-first-reading.jsonl'),'generator_sha256':sha(Path(__file__)),'no_canonical_changes':True,'scope_boundary':'Complete explicit Kui comparison lists, clauses and immediately compared table; surrounding unrelated examples are not an entire-volume ingestion. Target Kui occurrences remain in outsidechapter-original-reviewed.jsonl.','residuals':['Four Juang suffix glyph/quantity decisions need independent narrow original review.','Tamil underbar versus underdot and several compressed quantities are provisional; source notation preserved without phonemic assertions.','Korava, Koi, Burgandi source identities require qualified registry routing, not automatic modern-language assignment.','Two source hypothetical forms remain marked as such, with no graph edges.','Kolami p570 forms are explicit same-forms cross-reference expansions; retain source scope.']},ensure_ascii=False,indent=2)+'\n')
print(len(rows))
