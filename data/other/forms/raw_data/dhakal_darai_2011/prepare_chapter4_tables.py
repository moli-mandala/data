"""Complete printed lexical/paradigm table cells; first reading only."""
import json,re,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent
pages={n:json.loads((P/'chapter4-native'/f'p{n:03}.json').read_text()) for n in range(77,125)}
U=[]
def txt(n,l):return ''.join(s['decoded'] for s in pages[n]['lines'][l-1]['spans']).strip()
def add(n,li,gloss,tags='',form=None,extra=(),note='',status='first-reading',ordinal=1):
 f=txt(n,li) if form is None else form
 f=unicodedata.normalize('NFC',f.strip().lstrip(':').strip())
 key=f'dhakal2011:p{n}:ch4:table:l{li}:a{ordinal}'
 assert key not in {x['source_unit_key'] for x in U},key
 U.append({'source_unit_key':key,'printed_page':n,'pdf_page':n+22,'form':f,'gloss':gloss,'tags':tags.split(),'source_lines':[li,*extra],'raw_evidence':[pages[n]['lines'][v-1] for v in [li,*extra]],'note':note,'status':status,'unresolved_glyphs':re.findall(r'⟦U\+[0-9A-F]+⟧',f),'review':'Native-cell alignment first reading; original-image review pending.'})
def quoted(n,li):return txt(n,li).strip('‘’\' : ')
for n,fl,gl,t in [(80,55,56,'noun'),(80,58,58,'noun derived'),(80,60,61,'noun'),(80,63,64,'noun derived'),(80,66,67,'noun'),(80,68,69,'noun derived'),(80,71,72,'noun'),(80,74,75,'noun derived')]:
 if fl==gl:add(n,fl,'a person involved in marriage ceremony',t,form='dzənteherə')
 else:add(n,fl,quoted(n,gl),t,extra=[gl])
add(81,46,'to clean with a broom','verb',form='bahaḍike',extra=[47]);add(81,46,'broom','noun derived',form='bahaḍni',extra=[47],ordinal=2)
add(81,48,'to stand','verb',extra=[49]);add(81,51,'naming ceremony','noun derived',extra=[52])
# Compounds retain all printed component forms plus full compounds.
for n,triples in {82:[[(5,'my elder brother'),(7,'my younger brother'),(9,'(my) brothers')],[(15,'boy'),(18,'girl'),(21,'children')],[(33,'husband'),(35,'wife'),(37,'husband and wife')],[(43,'hand'),(46,'leg'),(48,'limbs/body')],[(58,'grandfather'),(60,'grandmother'),(62,'grandparents')],[(69,'pains'),(71,'happiness'),(73,'pains and happiness')],[(104,'line'),(106,'line'),(107,'line')],[(116,'forest'),(119,'jungle'),(120,'forest')],[(129,'land'),(131,'land'),(132,'land')]],83:[[(27,'meat'),(29,'spinach'),(31,'cooked meat (meat curry)')],[(38,'fish'),(40,'spinach'),(42,'cooked fish (fish curry)')],[(57,'basket'),(60,'litter'),(62,'marriage relation')],[(70,'colour'),(73,'juice'),(75,'enjoyment')]]}.items():
 for triple in triples:
  for i,(li,g) in enumerate(triple):
   f=txt(n,li).strip(': ')
   tags='noun'+(' compound' if i==2 else '')
   note='Printed component/compound table; English plural does not assert source plural morphology.'
   if n==82 and li==73:note+=' Literal dukʰədukʰə retained despite happiness gloss and dukʰəsukʰə in following example.'
   if n==82 and li in [5,7,9]:tags+=' poss 1sg'
   add(n,li,g,tags,form=f,note=note)
# Gender pairs.
for ml,mg,fl,fg in [(29,30,31,32),(34,35,37,38),(41,42,43,44),(45,46,47,48),(49,50,51,52)]:
 for li,gi,gender in [(ml,mg,'m'),(fl,fg,'f')]:
  tags='noun '+gender+(' poss 1sg' if li in[49,51] else '')
  add(85,li,quoted(85,gi),tags,extra=[gi],note='Explicit Male/Female columns in20.')
for base,gl,mas,fem in [(28,29,31,33),(34,35,37,39),(40,41,42,44),(45,46,47,49),(50,52,53,56),(57,58,59,61),(62,63,64,66),(67,69,71,74),(75,76,77,79),(80,81,82,84),(86,87,89,91),(92,93,94,96)]:
 for li,t in [(base,'adj'),(mas,'adj m'),(fem,'adj f')]:add(87,li,quoted(87,gl),t,extra=[gl],note='Adjective/base, masculine and feminine columns in24; repeated forms retained as separate cells.')
# Possessive suffix table has two separately scoped dialect columns.
for start,person,honor in [(17,'1sg',''),(24,'1pl',''),(29,'2sg',''),(34,'2sg','honorific'),(39,'2pl',''),(44,'3sg',''),(49,'3sg','honorific'),(56,'3pl','')]:
 case_lines={17:(17,18,19,21),24:(24,25,26,27),29:(29,30,31,32),34:(34,35,36,37),39:(39,40,41,42),44:(44,45,46,47),49:(49,50,52,54),56:(56,57,58,60)}[start]
 nom,gen,ch,ta=case_lines
 add(90,nom,'','pron '+person+' '+honor,note='Table4.1 pronoun column; person/number explicit, no separate English translation printed.')
 add(90,gen,'','pron poss '+person+' '+honor,note='Table4.1 possessive pronoun column.')
 for li,lect in[(ch,'Chitwan'),(ta,'Tanahun')]:
  f=txt(90,li);add(90,li,'','suffix poss '+person+' '+honor,note='Explicit '+lect+' possessive-suffix column.',status='zero-morphology-evidence' if f=='-ø' else 'first-reading')
  U[-1]['lect']=lect
# Numeral/classifier table.
for n,(nl,hu,non) in enumerate([(41,42,43),(45,46,47),(49,50,51),(53,54,55),(57,58,59)],1):
 for li,t in[(nl,'num'),(hu,'num classifier'),(non,'num classifier')]:add(92,li,str(n),t,note='Table4.2 numeral and human/nonhuman classifier columns; no inferred noun supplied.')
# Personal pronouns: every case cell, with source blank ERG cells represented.
personal=[('1sg','',31,[32,33,34,35],'I'),('1pl','',37,[38,39,40,41],'we'),('2sg','non-honorific',43,[44,45,46,47],'you'),('2pl','non-honorific',49,[49,50,51,52],'you'),('2sg','honorific',54,[55,56,57,58],'you'),('2pl','honorific',60,[61,62,63,64],'you'),('2sg','honorific high-honorific',66,[67,68,69,70],'you'),('2pl','honorific high-honorific',72,[73,74,75,76],'you'),('3sg','non-honorific',78,[79,80,81,82],'he/she'),('3pl','non-honorific',84,[85,86,87,88],'they'),('3sg','honorific',90,[91,92,93,94],'he/she'),('3pl','honorific high-honorific',96,[97,98,99,100],'they')]
for person,honor,nl,cls,g in personal:
 base=txt(112,nl).split("'")[0].strip()
 for i,f in enumerate(base.split('/'),1):add(112,nl,g,'pron nom '+person+' '+honor,form=f,ordinal=i,note='Table4.3 literal alternatives; do not extend a printed suffix to an unmarked alternative.')
 for ci,(li,case) in enumerate(zip(cls,['gen','acc dat','abl com','erg']),1):
  f=txt(112,li)
  if nl==49 and ci==1:f=f.split("'")[-1].strip()
  add(112,li,g,'pron '+case+' '+person+' '+honor,form=f,ordinal=3 if nl==49 and li==49 else 1,status='source-blank' if f=='-' else 'first-reading',note='Explicit case column; original SPL heading on third-person plural retained in raw evidence.')
# Demonstratives, source column spellings retained despite surprising syncretisms.
for number,honor,lis in [('sg','',[23,24,25,27,28]),('pl','',[31,33,34,35,36]),('sg','honorific',[38,39,40,41,42])]:
 for li,case in zip(lis,['nom','gen','acc dat','abl com','erg']):add(115,li,'this' if number=='sg' else 'these','pron prox '+number+' '+honor+' '+case,note='Table4.4 proximal case column; literal source spelling.')
add(115,45,'these','pron prox pl honorific nom')
for i,(f,case) in enumerate(zip(txt(115,46).split(),['gen','acc dat','abl com','erg']),1):add(115,46,'these','pron prox pl honorific '+case,form=f,ordinal=i,note='Four printed columns on shared native line.')
# Relative and interrogative tables.
for n,items in [(116,[(32,'who','nom'),(33,'who','MULTI'),(34,'who','abl com'),(36,'what','nom'),(37,'what','erg'),(38,'what','gen'),(39,'what','dat acc'),(40,'what','abl com')]),(117,[(23,'who','nom'),(24,'who','erg'),(25,'who','gen'),(26,'who','acc dat'),(27,'who','abl com'),(29,'what','nom'),(30,'what','erg'),(31,'what','gen'),(32,'what','acc dat'),(33,'what','abl com')])]:
 for li,g,case in items:
  forms=txt(n,li).split() if case=='MULTI' else txt(n,li).split('/')
  for i,f in enumerate(forms,1):
   c=['erg','gen','dat acc'][i-1] if case=='MULTI' else case
   add(n,li,g,'pron '+('relative' if n==116 else 'interr')+' '+c,form=f,ordinal=i,note='Literal table case form, independent of differing prose spellings.')
for li,gl,t in [(7,8,'pron refl'),(9,10,'pron refl erg'),(11,12,'pron refl abl com')]:add(119,li,'self',t,extra=[gl])
# Indefinite contrast pairs, every printed repeated base form retained.
for li,forms,g,t in [(24,['kiti'],'what','pron interr'),(26,['kjahau','kəlhəu'],'something','pron indef'),(28,['kiti'],'what','pron interr'),(30,['kiti-hun'],'something','pron indef'),(34,['kono'],'who','pron interr'),(35,['kolhəu'],'somebody','pron indef'),(38,['kono'],'who','pron interr'),(39,['kono-hun'],'somebody','pron indef')]:
 for i,f in enumerate(forms,1):add(121,li,g,t,form=f,ordinal=i)
(P/'chapter4-tables-first-reading.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in U))
print(len(U),'complete table cells/alternatives including blanks and zero morphology; original-image review pending')
