"""Complete chapters1–2 attestation proposal, with exact native-span provenance."""
import csv,hashlib,json,re,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent
pages={n:json.loads((P/'chapters12-native'/f'p{n:03}.json').read_text()) for n in range(1,42)}
units=[]
def text(n,line):
 return ''.join(s['decoded'] for s in pages[n]['lines'][line-1]['spans']).strip()
def add(n,lines,form,gloss,tags='',note='',lect='Chitwan',status='target',occurrence=1):
 if isinstance(lines,int):lines=[lines]
 form=unicodedata.normalize('NFC',form)
 key=f'dhakal2011:p{n}:ch12:l{lines[0]}:a{occurrence}'
 assert key not in {x['key'] for x in units},key
 evidence=[pages[n]['lines'][v-1] for v in lines]
 unresolved=re.findall(r'⟦U\+[0-9A-F]+⟧',form)
 units.append(dict(key=key,printed_page=n,pdf_page=n+22,line_ids=lines,form=form,gloss=gloss,tags=tags.split(),lect=lect,status=status,note=note,unresolved_glyphs=unresolved,native_evidence=evidence,review='Owner complete native-text read and original-image check of lexical pages; independent review pending'))
# Literature review: source qualifications, not silently assigned to Chitwan.
add(2,29,'-o','adjectival ending','suffix','Hodgson historical list ending; author contrasts contemporary -ə.',lect='unspecified')
add(2,29,'-ə','adjectival ending','suffix','Author says contemporary adjective ending corresponds to Hodgson -o.',lect='unspecified',occurrence=2)
for f,t,li in [('-i','erg',20),('-ke','acc',21),('-jə','loc',21)]:
 add(3,li,f,'','suffix '+t,'Reported analysis of Kotapish1973; not automatically attributed to Chitwan.',lect='unspecified',occurrence=1 if f!='-jə' else 2)
add(3,21,'maṭe','from','postp abl','Reported Kotapish1973 analysis.',lect='unspecified',occurrence=3)
add(3,27,'ɡoru-ke','ox','acc','Printed gloss ox-ACC; explicit Chitwan counterexample.')
add(3,27,'bʰotʰi-ke','Bhothi fish','acc','Printed gloss Bhothi fish-ACC; explicit Chitwan counterexample.',occurrence=2)
add(3,29,'-se','from','suffix abl')
add(3,29,'maṭe','from','postp abl',occurrence=2)
add(3,30,'maṭe','from','postp abl','Repeated report of Kotapish analysis.',lect='unspecified')
for k,f in enumerate(['lʰ','mʰ','nʰ','rʰ'],1):
 add(3,[16,17],f,'consonant sound','','Metalinguistic segment discussion; no lexical meaning asserted.',status='segment-evidence',occurrence=k)
add(5,9,'laɡ','touch/apply','verb','Verb cited in review of Paudyal2008.',lect='unspecified')
add(6,4,'nidzə','','neg','Printed gloss NEG; source discusses negative prefixes.',lect='unspecified')
add(6,22,'-inuhuk','','suffix pass','Author explicitly identifies passive suffix.',lect='unspecified')
# Name lists and speculative source etymology stay attributed documentary evidence.
for i,f in enumerate(['Darai','Darbai','Darhi','Daree','Daroe','Daraie','Dry'],1):
 add(10,[29,30,31,32,33],f,'name of ethnic group and language','proper-noun','Roman name variants attributed to previous publications; not a phonemic transcription.',lect='unspecified',status='name-evidence',occurrence=i)
add(11,[10,11,12,13],'daras','administrative regions','','Speculative origin cited from Gautam and Magar; language of etymon not established. No donor relation asserted.',lect='unattributed',status='comparison-control')
add(2,31,'khas kuraa','','','Cited label for another language; no target-language lexical analysis.',lect='other-language',status='comparison-control')
add(2,33,'lal mohar','','','Untranslated historical-document term; no target-language attribution.',lect='unattributed',status='comparison-control')
for i,li in enumerate([13,15,17],1):
 add(19,li,'khas kuraa','','','Repeated label in quoted Grierson discussion, not Darai lexical data.',lect='other-language',status='comparison-control')
# Cultural terminology; all repeated source attestations retained.
add(17,5,'puja','worship','noun uncertain','Unmarked cultural term in cited ethnographic prose; target-language attribution uncertain.',lect='unspecified')
add(17,21,'əsəra','porch','noun')
add(17,21,'əsəra','porch','noun',occurrence=2)
add(17,23,'əsəra','porch','noun','Repeated mention refers to same locally defined part of house.')
add(17,[23,24],'pənəici','place accommodating the water pot, attached to the front side of the house','noun','Gloss paraphrases explicit source description.',occurrence=2)
add(17,24,'əsəra','porch','noun','Repeated mention refers to same locally defined part of house.')
add(17,[24,25],'ə̃ɡna','yard','noun',occurrence=2)
add(17,[28,29],'kãdbari','area near oven','noun','F0E2 visibly a nasal mark over a; local definition on18 supplies meaning.')
add(17,30,'kẽwari','doors made of wood','noun','F0E2 visibly a nasal mark over e; source description, not inferred plural morphology.')
add(17,30,'pʰokra','doors made of twigs and mud','noun','Source description; not inferred plural morphology.',occurrence=2)
add(18,[6,7],'kãdbari','area near oven','noun','F0E2 visibly nasal over a.')
add(18,[7,8,9],'kãdbari','area near oven','noun','Repeated mention; F0E2 visibly nasal over a.')
add(18,[10,11,12],'mədzʰɡʰər','central part of the house; open area for feasting and lodging materials','noun','Gloss summarizes explicit prose definition.')
add(18,[18,19],'potka','shelf','noun')
add(18,22,'ɡənbaba','the ancestors','noun','Source describes ancestral shrine with this name; no inferred plural morphology.')
add(18,28,'ɡʰər ɡurəu','shaman','noun')
add(18,29,'suḍʰerni','midwife','noun')
add(18,31,'uṭʰəuni','naming ceremony','noun')
# Table pairs: retain literal semantic anomalies, not harmonized kinship glosses.
pairs={20:[(36,37),(38,39),(40,41)],21:[(3,4),(10,11),(12,13),(14,16),(17,18),(22,23),(24,25),(26,27),(28,29),(30,32),(33,34),(35,36),(37,38),(39,40),(41,42),(43,44),(45,46),(47,48),(49,50),(51,52),(53,54),(55,56),(57,58)],22:[(4,5),(6,8),(9,11),(12,13),(23,24),(25,26),(27,28),(29,30),(31,32),(33,34),(35,36),(37,38),(39,40),(41,42),(43,44),(45,46),(47,48),(49,50),(54,55),(56,57),(58,59),(60,61),(62,64)],23:[(5,6),(7,8)],24:[(4,5),(6,7),(8,9)]}
for n,pp in pairs.items():
 for fl,gl in pp:
  form=text(n,fl).replace('⟦U+F0E2⟧','\u0303')
  gloss=text(n,gl).strip('‘’\' ')
  tags='noun'+(' voc' if n==24 else '')
  note='Kinship list; printed relational gloss retained literally.'
  if '(M.)' in gloss:tags+=' m';gloss=gloss.replace('(M.)','').strip()
  if '(F.)' in gloss:tags+=' f';gloss=gloss.replace('(F.)','').strip()
  # Remove closing quote before the source parenthetical while retaining that scope.
  gloss=gloss.replace('’ (',' (').rstrip('’ ').strip()
  if n==21 and fl in [47,49]:tags+=' poss 1sg'
  if n==22 and fl==45:note+=' Printed father conflicts with mother on25; retain both attestations separately.'
  if n==21 and fl==57:note+=' Printed sister’s daughter is retained despite adjacent brother/sister contrast.'
  if n==24 and fl==6:note+=' F0E2 final nasal attachment visually checked.'
  for i,f in enumerate(form.split('/'),1):add(n,[fl,gl],f,gloss,tags,note,occurrence=i)
add(23,33,'hawa mante','addressing without maintaining proper manner','multiword-expression')
for li,f,g in [(18,'ete sun tə','Listen hither'),(19,'ete her tə','Look hither'),(20,'ijə dze','this one, I said')]:
 add(24,li,f,g,'sentential','Whole indirect-address expression; source does not segment or label internal grammar here.')
add(25,4,'səmdʰi',"son/daughter's wife's father",'noun')
add(25,[5,6],'pau','high honorific address to son/daughter’s wife’s father','pron honorific high-honorific','Source explicitly describes this address usage; gloss is scoped contextual description.')
add(25,[5,6],'səmdʰinin',"son/daughter's wife's mother",'noun','Printed mother differs from father on22.',occurrence=2)
add(25,[6,7],'ɡosaĩ','address to son/daughter’s wife’s mother','voc','Source describes address usage; gloss is contextual description.')
add(25,[7,8],'bʰosre','husband’s elder brother','noun')
for li,f,g,t in [(15,'morike','to die','verb'),(21,'dzudzi','penis','noun'),(21,'ɡenṭʰa','penis','noun'),(22,'cʰucʰi','vegina','noun'),(24,'dzʰãṭ','hairs around the sexual organs','noun'),(26,'cudʰike','have sexual intercourse','verb'),(28,'ɡuləi','testicles','noun'),(30,'ṭiṭno','a part of vagina','noun'),(32,'mudzi','anus','noun'),(34,'lalahi','menstruation','noun'),(34,'bərsat','menstruation','noun')]:
 add(25,[li,li+1] if li not in[21,34] else li,f,g,t,'Source explicitly lists as taboo topic; no vulgar/pejorative synonym inferred. Printed English spelling retained.',occurrence=2 if f in ['ɡenṭʰa','bərsat'] else 1)
for li in [31,32]:
 add(26,[li,32] if li==31 else [32],'dərai','Darai: ethnic and language name','proper-noun','Explicit Chitwan pronunciation; sentence continues on27.',lect='Chitwan')
 add(26,[li,32] if li==31 else [32],'dərəi','Darai: ethnic and language name','proper-noun','Explicit Tanahun pronunciation; sentence continues on27.',lect='Tanahun',occurrence=2)
# Comparative table: every cell and printed alternative, no cross-lect merger.
comparative=[(30,92,95,98,'we','pron 1pl'),(30,100,103,106,'new','adj'),(30,107,111,114,'this','pron prox'),(30,116,120,123,'what','pron interr'),(30,124,126,127,'not','neg'),(30,129,132,134,'throat','noun'),(30,135,137,139,'root','noun'),(31,3,5,7,'moon','noun'),(31,8,11,14,'star','noun')]
for n,ch,ta,gl,g,t in comparative:
 for li,lect in [(ch,'Chitwan'),(ta,'Tanahun')]:
  for i,f in enumerate(re.split(r'\s*[,/]\s*',text(n,li)),1):
   add(n,[li,gl],f,g,t,'Explicit dialect comparison in example12; all alternatives preserved.',lect=lect,occurrence=i)
add(37,8,'hamrə madzʰɡʰər','inner part of our home','multiword-expression','Radio programme title; broadcast in Nawalparasi does not prove lect origin.',lect='unspecified')
for i,f in enumerate(['Kokalsahi','Rajbanshi','Suryabanshi','Kulchapur','Banpale','Buda','Champaful','Khajuwar','Satanpur','Handikale','Narabangshi','Rana'],1):
 add(27,[50,51],f,'clan name','proper-noun','Cited ethnographic name spelling from Darai1996; retained as documentary name evidence, not asserted phonemic form.',lect='unspecified',status='name-evidence',occurrence=i)
# Complete page dispositions cover non-lexical material without promoting English survey prose.
notes={1:'Introduction and methodological context; no lexical examples.',2:'Historical adjective endings and cited language names.',3:'Segment discussion plus case morphemes and cited examples.',4:'Literature review; no printed target forms.',5:'Literature review includes laɡ.',6:'Literature review includes negative and passive morphemes.',7:'Methodological context; no target utterances.',8:'Corpus size, collection and dialect scope; no printed target utterances.',9:'Thesis organization.',10:'Ethnonym/autoglotonym alternative spellings; documentary evidence.',11:'Speculative ethnonym etymology and ethnographic English quotations; no assigned donor language.',12:'English ethnographic quotation and demography.',13:'Demographic table2.1; place labels are geographic context.',14:'Map2.1 geographic labels.',15:'Map2.2 is image-only geographic context; original visually checked, no lexical list.',16:'Map2.3 consultants and geographic context; personal names not lexical entries.',17:'House terminology; photograph and English description.',18:'House and naming-ceremony terms.',19:'Classification discussion and English historical quotation; language names are metalinguistic.',20:'Kinship introduction and example1.',21:'Kinship examples1–3.',22:'Kinship examples4,5,7; source numbering has no6.',23:'Kinship example8 and address discussion.',24:'Direct/indirect address examples9–10.',25:'Address terminology and taboo example11.',26:'Survey setup and local autonyms; respondent facts not lexical utterances.',27:'Autonym sentence continuation, demographic table2.2, cited clan names as documentary context.',28:'Occupation/education tables2.3–2.4, no Darai translations.',29:'Dialect survey English descriptions.',30:'Survey table2.5 and explicit lexical dialect comparison12.',31:'Continuation of example12 then survey table2.6.',32:'English survey question/table2.7; language-vitality discussion.',33:'Language-transmission table2.8 and English quotations.',34:'English language-vitality discussion/quotations.',35:'English survey questions/statistics table2.9.',36:'English survey questions/statistics table2.10.',37:'Radio title plus literacy discussion and English quoted criteria.',38:'English legal/UNESCO quotations and survey summary; no Darai translations.',39:'English questions/counts table2.11 and English quoted criteria.',40:'English questions/counts table2.12 and narrative responses; no Darai response text.',41:'English chapter summary; no target utterances.'}
rows=[]
for u in units:
 if u['printed_page']==26 and u['status']=='target':
  u['cross_page_evidence']={'printed_page':27,'pdf_page':49,'line':pages[27]['lines'][2]}
 if u['form']=='puja':
  u['phonemic_status']='unresolved conventional Roman spelling; not IPA93'
  u['note']+=' Printed in Times Roman italic, not IPA93; phonemic interpretation and house conversion remain unapproved.'
 if u['status']!='target':continue
 tags=list(u['tags'])
 if u['lect']=='Chitwan':tags.append('dialect:Darai:darai_chitwan:Chitwan')
 if u['lect']=='Tanahun':tags.append('dialect:Darai:darai_tanahun:Tanahun')
 rows.append(['Darai','',u['form'],u['gloss'],'','' if u['form']=='puja' else u['form'],u['note'],f"dhakal2011darai[{'pp. 26–27' if u['printed_page']==26 else 'p. '+str(u['printed_page'])}, native lines {','.join(map(str,u['line_ids']))}]",'','',u['key'],'','','',' '.join(tags)])
(P/'chapters12-proposal-audit.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in units))
with (P/'chapters12-proposal.csv').open('w',newline='') as s:csv.writer(s).writerows(rows)
for stem,selected in [('chapters12-ipa-proposal',[r for r in rows if r[2]!='puja']),('chapters12-roman-preservation',[r for r in rows if r[2]=='puja'])]:
 with (P/(stem+'.csv')).open('w',newline='') as s:csv.writer(s).writerows(selected)
census=[]
for n in range(1,42):
 uu=[u for u in units if u['printed_page']==n]
 census.append(dict(printed_page=n,pdf_page=n+22,native_line_count=len(pages[n]['lines']),disposition=notes[n],source_units=[u['key'] for u in uu],target_rows=sum(u['status']=='target' for u in uu),audit_only=sum(u['status']!='target' for u in uu),native_span_sha256=hashlib.sha256((P/'chapters12-native'/f'p{n:03}.json').read_bytes()).hexdigest(),status='Complete native-text read; lexical examples and image-only map checked against original; independent review pending'))
(P/'chapters12-page-census.json').write_text(json.dumps(census,ensure_ascii=False,indent=2)+'\n')
print(f'{len(units)} attestation/evidence units; {len(rows)} target proposal rows; all41pages accounted; no installation')
