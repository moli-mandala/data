"""Complete literal extraction of the pinned, unfinished MLA Ho A–C archive."""
from __future__ import annotations
import argparse,csv,hashlib,json,re,unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source-wayback.txt'
PROPOSAL=HERE/'proposal.csv'
AUDIT=HERE/'audit.jsonl'
INSTALLED=HERE.parents[1]/'20260925-donegan-stampe-ho.csv'
SOURCE_SHA256='dede86913fdad97ba71bda4c778774f90331fb8b947cee83feceff579824f77f'
SOURCE_KEY='DHED'
POS={'n':['noun'],'v':['verb'],'vr':['verb','refl'],'vc':['verb','caus'],'adj':['adj'],'adv':['adv'],'num':['num'],'pron':['pron'],'sx':['suffix'],'pp':['postp'],'interj':['interj'],'nk':['noun','kinship'],'vax':['auxiliary'],'':[],'c':[],'to':[],'.':[]}

def records():
 data=SOURCE.read_bytes()
 if hashlib.sha256(data).hexdigest()!=SOURCE_SHA256:raise ValueError('Source snapshot changed')
 parts=data.decode('ascii').split('\f');assert len(parts)==3
 out=[]
 for line,raw in enumerate(parts[2].splitlines(),1):
  if not raw.strip():continue
  if raw.startswith('\x10'):
   out.append({'source_id':f'heading:{line}','body_line':line,'raw':raw,'kind':'heading'});continue
  m=re.search(r'\$(\d{5})\*?\$',raw)
  if not raw.lstrip().startswith('<'):
   assert m and out[-1]['source_id'].startswith('unnumbered:')
   out[-1].update(source_id=m.group(1),raw=out[-1]['raw']+'\n'+raw,line_end=line)
   continue
  out.append({'source_id':m.group(1) if m else f'unnumbered:{line}','body_line':line,'raw':raw,'kind':'lexical'})
 assert len(out)==1524
 assert [x['source_id'] for x in out if x['source_id'].isdigit()]==[f'{n:05d}' for n in range(1,1517)]
 return out

def quote_end(s,start):
 """Balance source opening backticks, skipping ordinary apostrophes in words."""
 double=s.startswith('``',start)
 if double:
  end=s.find("''",start+2)
  return (s[start+2:end],end+2) if end>=0 else (s[start+2:],len(s))
 depth=1;i=start+1
 while i<len(s):
  if s[i]=='`':depth+=1
  elif s[i]=="'":
   if i and i+1<len(s) and s[i-1].isalpha() and s[i+1].isalpha():i+=1;continue
   if i and s[i-1]=='s' and re.match(r'\s+[a-z]',s[i+1:]):i+=1;continue
   depth-=1
   if depth==0:return s[start+1:i],i+1
  i+=1
 return s[start+1:],len(s)

def clean(text):
 text=re.sub(r'>(?=[A-Za-z])','> ',text)
 text=re.sub(r'\s+',' ',text.replace('^','').replace('_',' ').replace('%','').replace('<','').replace('>','')).strip()
 return re.sub(r',\s*,',',',text)

def parse(record):
 sid=record['source_id'];key='ho-mla2004:'+sid;raw=record['raw'];audit={**record,'entry_key':key,'rows':[],'repairs':[]}
 if record['kind']=='heading':audit.update(status='excluded_heading',reason='Alphabet heading or terminal control');return [],audit
 if sid in {'00870','01254'}:
  audit.update(status='excluded_control',reason='Archive explicitly states this story word is not Ho; probable Oriya attribution not forced into a new target row');return [],audit
 text=re.sub(r'\s*\$\d{5}\*?\$\s*$','',raw).strip()
 replacements={'00005':('VI\t``','{v} ``'),'00482':('<bA_ tasaD>>','<bA_ tasaD>'),'00689':('<b>anA_-jiyaG>','<banA_-jiyaG>'),'00955':('{adf}','{adj}'),'01460':('{adjj}','{adj}')}
 if sid in replacements:
  old,new=replacements[sid];assert old in text;text=text.replace(old,new);audit['repairs'].append({'old':old,'new':new,'reason':'unambiguous lexical/POS delimiter typo'})
 if sid=='00818':text=text.replace("{an ^inn, a ^lodging_house} `'","{n} `an ^inn, a ^lodging_house'");audit['repairs'].append({'reason':'definition was misplaced inside grammar braces; noun category explicit from inn/lodging-house'})
 if sid=='01222':text=text.replace("{.} `'vro become ^blurred","{.} `vro become ^blurred'");audit['repairs'].append({'reason':'malformed final quote repaired; literal damaged vro retained, no guessed POS or lexical emendation'})
 if sid=='01086':text=text.replace('<biTa gonyor (<gO~yor>) daru>','<biTa gonyor daru>,,<biTa gO~yor daru>');audit['repairs'].append({'reason':'explicit parenthetical alternate within head expanded as two source forms'})
 if sid=='00678':text=text.replace('; big ^brother',"'; {n} `big ^brother");audit['repairs'].append({'reason':'two locality-scoped nominal senses separated without spreading geographic attribution'})
 if sid=='01410':text=text.replace('; ^nest',"'; {n} `^nest");audit['repairs'].append({'reason':'South Singhbhum qualifies nest sense only; split from generic straw'})
 start=text.find('{');assert start>=0,(sid,text)
 header=text[:start];heads=re.findall(r'<([^<>]*)>',header)
 residual=re.sub(r'<[^<>]*>','',header).strip(' ,\t')
 if residual:raise ValueError(('unparsed head',sid,residual))
 if heads==['']:
  audit.update(status='excluded_empty',reason='Source record has empty head and empty definition');return [],audit
 if not heads:raise ValueError(('missing head',sid))
 audit['headwords']=heads
 # Entry16 contains an explicitly second headword sense interrupted by a quoted
 # example. Separate this malformed wrapper without assigning sentence heads.
 if sid=='00016':
  text=text.replace("', e.g. <mi a_reyo_ kari_m>  {`say at least something'; (when followed by a noun)} `", "'; {n} `")
  audit['repairs'].append({'reason':'second noun sense mouthful separated from intervening sentence example; exact source example retained in raw'})
 senses=[];pos=start;tail=''
 while pos<len(text):
  m=re.match(r'\{([^}]*)\}\s*',text[pos:])
  if not m:tail=text[pos:];break
  label=m.group(1);pos+=m.end()
  if label not in POS:raise ValueError(('unknown label',sid,label))
  if pos>=len(text) or text[pos]!='`':raise ValueError(('missing opening quote',sid,text[pos:]))
  gloss,end=quote_end(text,pos);senses.append((label,gloss));pos=end
  nxt=re.match(r'[.;\s]*(?=\{)',text[pos:])
  if nxt:pos+=nxt.end()
  else:tail=text[pos:];break
 scope_plan=json.loads((HERE/'reviewed-sense-scopes.json').read_text()).get(sid)
 if scope_plan:
  audit['unsplit_source_senses']=[{'label':x,'gloss':y} for x,y in senses]
  senses=[(x['label'],x['gloss']) for x in scope_plan]
  audit['repairs'].append({'reason':'explicit ordinary/register/reflexive clauses scoped as separate senses; full original clauses retained'})
 audit['source_senses']=[{'label':x,'gloss':y} for x,y in senses];audit['source_commentary']=tail
 rows=[]
 for si,(label,source_gloss) in enumerate(senses,1):
  inline_refs=re.findall(r'@?\b(?:B|H|Les)\.\s*\d+(?:/\d+)*(?:-\d+)?(?:,\s*\d+)*',source_gloss)
  scoped_gloss=source_gloss
  for ref in inline_refs:scoped_gloss=scoped_gloss.replace(ref,'')
  scoped_gloss=re.sub(r',\s*,',', ',scoped_gloss).replace('()','')
  aliases=[]
  alias_source=scoped_gloss
  def alias_definition(match):
   begin=alias_source.rfind(';',0,match.start())+1
   end=alias_source.find(';',match.end())
   return clean(alias_source[begin:end if end>=0 else len(alias_source)].replace(match.group(0),'')).strip(' .,;')
  for am in re.finditer(r'\((also(?: called| spelled)?|(?:an )?alternate form of|a variant form of)\s+([^()]*)\)',scoped_gloss):
   kind,body=am.groups()
   if body.startswith('cf.') or '<' not in body:continue
   for alt in re.findall(r'<([^<>]+)>',body):aliases.append((alt,kind.removeprefix('an ').removeprefix('a '),alias_definition(am)))
   if body.count('<')>body.count('>'):
    alt=body.rsplit('<',1)[1].strip(' ,')
    if alt:aliases.append((alt,kind.removeprefix('an ').removeprefix('a '),alias_definition(am)));audit['repairs'].append({'reason':'missing closing delimiter in explicit parenthetical alternate recovered','form':alt})
   scoped_gloss=scoped_gloss.replace(am.group(0),'')
  grammar_notes=[]
  reflexive_head=bool(re.search(r'\((?:a )?reflexive form\b',scoped_gloss))
  if reflexive_head:
   for gm in re.findall(r'\((?:a )?reflexive form[^)]*\)',scoped_gloss):grammar_notes.append(clean(gm));scoped_gloss=scoped_gloss.replace(gm,'')
  alias_source=scoped_gloss
  for am in re.finditer(r'\balso called\s+((?:<[^>]+>(?:[,\s]*(?:or|and)?[,\s]*)?)+)',alias_source,re.I):
   for alt in re.findall(r'<([^>]+)>',am.group(1)):aliases.append((alt,'also called',alias_definition(am)))
   scoped_gloss=scoped_gloss.replace(am.group(0),'')
  gloss=clean(scoped_gloss).strip(' ,;');tags=list(POS[label]);flags=[]
  if reflexive_head:tags.append('refl')
  if re.search(r'\([^)]*(?:refl\. form|reflexive form)[^)]*\)', scoped_gloss):
   tags.append('refl')
   for gm in re.findall(r'\([^)]*(?:refl\. form|reflexive form)[^)]*\)',gloss):grammar_notes.append(gm);gloss=gloss.replace(gm,'').strip()
  if sid in {'00288','00419','00528','01274','01301','01303'}:tags.append('voc')
  if sid=='00280':
   tags.append('voc');grammar_notes.append(clean(source_gloss));gloss='father'
  poetic_glosses={('00913',1):'',('01054',1):'a nilgai',('00667',1):'the Sirkeer Cuckoo',('00706',1):'a pointed stick used for beating down paddy when putting it into a bandi',('01181',1):'the bamboo staff used by one who goes into a trance for divining',('01246',2):'to fast',('00028', 1):'',('00144', 1):'to store things under the roof of a house or in a tree',('00530', 1):'the acacia tree, probably Acacia arabica, Willd., Mimosaceae',('00573', 1):'',('00603', 1):'',('00606', 2):'to fence in; a threshing floor',('00629', 1):'',('00743', 1):'not having (something inanimate)',('00868', 1):'dung beetle',('00879', 1):'to be mildly crazy as a transitory consequence of a serious illness; to shake in a trance',('00903', 1):'to spit all over; to show disrespect',('00942', 1):'',('00970', 1):'a contagious disease',('01087', 1):'to start to eat or drink again after sickness; morally unclean',('01123', 1):'a word, a spoken assurance or promise',('01224', 1):'a snake',('01228', 1):'a hooded snake with erect poison fangs and a dilatable neck smaller than the cobra’s, yellow and dark brown with yellow belly and black near the tail',('01240', 1):'a straw rope',('01264', 1):'a bird which attacks silkworms; probably the Tree Pie',('01433', 1):'let',('01473', 1):'the Greyheaded Flycatcher',('00775',1):'a flower',('00303',1):'to release',('00314',1):'marriage',('00549',1):'to load upon',('00550',1):'to harass',('00418',1):'to hear',('00548',1):'the leather strips used to span drums; a mate or companion',('00551',1):'a field',('00602',2):'to make such sounds; to set a date; to invite',('00612',1):'garden',('00619',1):'to burn a hole into a flute; to burn',('00620',1):'to burn out; to drive out by burning; to touch someone with a heated object to neutralize the effects of an evil spirit',('00632',1):'the first pre-monsoon rains',('00752',1):'low lying fields',('00786',1):'to fast',('00799',1):'sky',('00871',1):'to feel pity',('00908',1):'pain, suffering',('01022',1):'',('01041',1):'',('01061',1):'',('01068',1):'',('01093',1):'numerous; make numerous',('01239',1):'head',('01245',1):'to sleep; a threshing floor',('01259',1):'Brahmin; those who bring buffaloes or other things from other places for sale',('01362',2):'to heap up; to sacrifice',('01362',3):'to become a spirit',('01403',1):'to lie down',('01493',1):'to feel pity; grieve',('01503',1):'to roam'}
  if (sid,si) in poetic_glosses and not scope_plan:
   grammar_notes.append(clean(source_gloss));gloss=poetic_glosses[(sid,si)]
   if re.search(r'poetic(?:al)?',source_gloss.replace('^','')):tags.append('poetic')
   if not gloss:flags.append('gloss: source comparison lacks a unique independent Ho definition; no donor-language meaning inherited')
  if '(recip. of <' in source_gloss.lower() or (si==1 and '|recip. of <' in tail.lower()):
   tags.append('reciprocal')
   for gm in re.findall(r'\(recip\. of [^)]*\)',gloss):grammar_notes.append(gm);gloss=gloss.replace(gm,'').strip()
   flags.append('derivation: explicit reciprocal parent retained in source prose; graph requires exact compatible sense (see record derivation decision)')
  if 'used for ^vocative' in source_gloss:tags.append('voc')
  if sid=='00005':tags.append('intr')
  if sid=='00746':tags.append('inanimate')
  if sid in {'00181','00815','01338','01441'}:tags.append('pejorative')
  if sid=='00815':grammar_notes.append(clean(source_gloss));gloss='term of abuse';flags.append('gloss: source gives abusive use and a sentence example, no independent literal definition')
  if sid in {'00231','00532','00675','00732','01009','01225'}:tags.append('proper-noun')
  if sid=='00742':tags.append('part')
  if sid=='01222' and si==3:tags.append('verb');flags.append('grammar: become is unambiguously verbal but damaged printed label remains uncertain; no reflexive tag guessed')
  if (sid=='00340' and si==2):tags.extend(['tr','animate'])
  if sid=='00449' or (sid=='00962' and si==1) or sid=='01232':tags.append('tr')
  if sid=='00585':tags.extend(['past','participle'])
  if sid=='00743':tags.append('participle')
  if sid in {'00715','00933'}:tags.append('reduplicated')
  if sid=='00166':tags.extend(['emph','neg'])
  if scope_plan:
   plan=scope_plan[si-1]
   if 'definition' in plan:grammar_notes.append(clean(source_gloss));gloss=plan['definition']
   tags.extend(plan.get('tags',[]))
  if re.search(r'collective (?:noun|word)',source_gloss.replace('^','')):
   tags.append('collective');grammar_notes.append(clean(source_gloss));gloss=re.sub(r'^(?:a )?collective (?:noun|word) for ','',gloss)
  for gm in re.findall(r'\(verbal noun of [^)]*\)',gloss):grammar_notes.append(gm);gloss=gloss.replace(gm,'').strip()
  if sid in {'00533','00595','00665','00791','00806','00811','00908','00912','00984','01056','01174','01258','01300','01313','01338'}:
   tags.append('loanword');grammar_notes.append(clean(source_gloss))
   if sid in {'00908','01056'}:flags.append('borrowing: source donor identification is explicitly tentative')
   for donor in re.findall(r'\([^)]*Hindi[^)]*\)',gloss):gloss=gloss.replace(donor,'').strip()
  if sid=='00533':gloss='after'
  if sid=='00665':tags.append('poetic')
  if sid in {'01228','01264'}:flags.append('source comparison is explicitly tentative; exact claim retained in Notes')
  if sid=='00577':grammar_notes.append(clean(source_gloss));gloss='to make; to inflict harm by witchcraft'
  for gm in re.findall(r'\([^)]*(?:better Ho|proper Ho|more common(?:ly| form)|more frequently used|poetic parallel|poetical parallel)[^)]*\)',gloss):
   grammar_notes.append(gm);gloss=gloss.replace(gm,'').strip()
  if audit['repairs']:flags.append('transcription: explicit malformed-markup repair')
  if label in {'c','to','.'}:flags.append('grammar: unexplained/damaged source label retained without guessing')
  if '??' in raw:flags.append('source editorial uncertainty')
  if not gloss:flags.append('gloss: source definition empty')
  if label=='vax':flags.append('grammar: source verbal adjunct category retained as auxiliary without inferred affix position')
  for marker,tag in [('(refl.)','refl'),('(fig.)','figurative'),('(poet.)','poetic'),('(tr.)','tr'),('(intr.)','intr')]:
   if marker in gloss:tags.append(tag);gloss=gloss.replace(marker,'').strip()
  if label=='pron':
   for phrase,tag in [('plural','pl'),('inclusive','inclusive'),('exclusive','exclusive')]:
    if phrase in gloss:tags.append(tag)
   if gloss=='you two':tags.extend(['du','second-person'])
  if re.search(r'\bpoetic(?:al)?\b',source_gloss.replace('^','')):tags.append('poetic')
  if re.search(r'#(?:Hi|Or|Skt|Engl)\.',tail):tags.append('loanword')
  commentary=re.sub(r'@[^!|#]*','',tail).strip(' .;')
  if grammar_notes:commentary=(commentary+'; '+'; '.join(grammar_notes)).strip('; ')
  etymology='; '.join(re.findall(r'#[^!|]+',commentary))
  commentary=re.sub(r'#[^!|]+','',commentary).strip(' .;')
  archive_refs=inline_refs+re.findall(r'@([^!|#$]+)',tail)
  base=key if si==1 else f'{key}:sense:{si}'
  scoped_heads=[(h,'head',gloss) for h in heads]+[(h,k,gloss if (sid,si) in poetic_glosses and not scope_plan else g) for h,k,g in aliases if h not in heads]
  for hi,(head,head_kind,head_gloss) in enumerate(scoped_heads,1):
   localflags=list(flags)
   if re.search(r'[^a-z -]',head) or 'ch' in head:localflags.append('transcription: source notation preserved without unsupported phonemic conversion')
   explicit_tags={'00044':['du','second-person'],'00045':['1pl','inclusive'],'00089':['3sg'],'00116':['1sg'],'00143':['du','third-person'],'00146':['3pl'],'00151':['du','first-person','inclusive'],'00155':['1pl','exclusive'],'00161':['du','first-person','exclusive'],'00171':['2sg'],'00192':['1sg'],'00260':['poss'],'00261':['poss'],'00262':['poss'],'00268':['2pl'],'00025':['poss'],'01479':['proper-noun']}
   tags+=explicit_tags.get(sid,[])
   locs={('00672',1):'noamundi',('00678',2):'noamundi',('00633',2):'chaibasa',('00934',2):'chaibasa',('01248',1):'goelkera',('01249',1):'noamundi',('01351',1):'south-singhbhum',('01410',2):'south-singhbhum'}
   if (sid,si) in locs:tags.append('dialect:ho:mla-'+locs[(sid,si)])
   if sid=='00678' and si==1:tags.append('voc');localflags.append('geography: source more toward the North lacks a named stable locality')
   rowtags=list(dict.fromkeys(tags+(['uncertain'] if localflags else [])))
   child=base if hi==1 else f'{base}:variant:{hi}' if head_kind=='head' else f'{base}:alias:{hi-len(heads)}'
   citation=f'DHED[archive entry {sid}]' if sid.isdigit() else f'DHED[archive body line {record["body_line"]}]'
   row_commentary=(commentary+' Archived reference prose: '+'; '.join(archive_refs)).strip() if archive_refs else commentary
   row=['ho','',unicodedata.normalize('NFC',head),head_gloss,'','',row_commentary,citation,'',etymology,child,base if hi>1 and (head_kind=='head' or head_kind in {'alternate form of','variant form of','also spelled'}) else '','','',' '.join(rowtags)]
   rows.append(row);audit['rows'].append({'entry_key':child,'form':head,'gloss':head_gloss,'source_label':label,'tags':row[14],'citation':citation,'review_reasons':localflags,'head_kind':head_kind})
 audit.update(status='ingested',reason='Every explicit head and grammatical sense retained')
 return rows,audit

def separate_archived_references(rows):
 for row in rows:
  if ', archive refs ' in row[7]:
   locator, prose = row[7].split(', archive refs ', 1)
   assert prose.endswith(']')
   prose = prose[:-1]
   row[7] = locator + ']'
   row[6] = (row[6] + ' Archived reference prose: ' + prose).strip()
 return rows

def prepare_legacy():
 rows=[];audit=[]
 for record in records():
  rs,a=parse(record)
  supplements=json.loads((HERE/'reviewed-supplements.json').read_text()).get(record['source_id'],[])
  for ni,sup in enumerate(supplements,1):
   key=a['entry_key']+f':supplement:{ni}'
   tags=' '.join(dict.fromkeys((sup['tags']+(' uncertain' if re.search(r'[^a-z -]',sup['form']) else '')).split()))
   row=['ho','',sup['form'],sup['gloss'],'','',sup['reason'],f'DHED[archive entry {record["source_id"]}, explicit subsidiary form]','','',key,'','','',tags]
   rs.append(row);a['rows'].append({'entry_key':key,'form':sup['form'],'gloss':sup['gloss'],'tags':tags,'source_label':'reviewed subsidiary form','review_reasons':[sup['reason']]})
  rows.extend(rs);audit.append(a)
 by_form={}
 by_key={r[10]:r for r in rows}
 for r in rows:by_form.setdefault(r[2],[]).append(r)
 pos_tags={'noun','verb','adj','adv','pron','num','postp','suffix'}
 pending=[]
 for a in audit:
  for si,sense in enumerate(a.get('source_senses',[]),1):
   if not (a['source_id'] in {'00288','00381','01046','01354'} or re.fullmatch(r'(?:syn\. for|same as|abbrev\. for|an alternate form of|an alternate spelling of|see|\^?another name for)\s*<[^>]+>[\s,.;_]*(?:q\.v\.)?[\s,.;_]*',sense['gloss'])):continue
   targets=re.findall(r'<([^>]+)>',sense['gloss'])
   base=a['entry_key'] if si==1 else a['entry_key']+f':sense:{si}'
   for x in a['rows']:
    if x['entry_key']==base or x['entry_key'].startswith(base+':variant:') or x['entry_key'].startswith(base+':alias:'):
     row=by_key[x['entry_key']];pos=set(row[14].split())&pos_tags
     candidates=[r for t in targets for r in by_form.get(t,[]) if r[10]!=row[10] and (not pos or bool(set(r[14].split())&pos))]
     x['definition_crossref']={'source_targets':targets,'candidate_keys':[r[10] for r in candidates]}
     row[9]=(row[9]+'; '+sense['gloss']).strip('; ')
     row[3]='';pending.append((row,x,candidates))
 for _ in range(4):
  for row,x,candidates in pending:
   if len(candidates)==1 and candidates[0][3]:
    row[3]=candidates[0][3];x['gloss']=row[3];x['definition_crossref']['status']='resolved exact source-head and POS'
 for row,x,candidates in pending:
  if not row[3]:
   x['definition_crossref']['status']='ambiguous' if len(candidates)>1 else 'unmatched or source target has no definition'
   x['gloss']='';x['review_reasons'].append('gloss: source cross-reference unresolved; no invented definition')
   if 'uncertain' not in row[14].split():row[14]+=' uncertain'
   x['tags']=row[14].strip();row[14]=x['tags']
 by_key['ho-mla2004:00750'][13]='ho-mla2004:00534:sense:2'
 for a in audit:
  if a['source_id']=='00750':a['derivation']={'source_parent':'baD','parent_key':'ho-mla2004:00534:sense:2','status':'exact verbal parent with compatible demand/exact meaning'}
 assert len({r[10] for r in rows})==len(rows)
 return rows,audit


def prepare():
 import importlib.util
 spec=importlib.util.spec_from_file_location("ho_full_expression_recovery",HERE/'prepare_expression_recovery.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 rows,census,recovered=module.prepare()
 audit=[json.loads(line) for line in (HERE/'legacy-before-expression-recovery-audit.jsonl').read_text().splitlines()]
 bykey={r[10]:r for r in rows};byid={a['source_id']:a for a in audit}
 for record in recovered:
  a=byid[record['source_id']]
  a.setdefault('whole_response_recovery',[]).append(record)
  keys=record['existing_form_keys'] or [record['entry_key']]
  for key in keys:
   row=bykey[key]
   existing=next((x for x in a['rows'] if x['entry_key']==key),None)
   item={'entry_key':key,'form':row[2],'gloss':row[3],'tags':row[14],'citation':row[7],'notes':row[6]}
   if existing is None:
    item.update(source_label='reviewed complete source response',review_reasons=record['uncertainty_types'],head_kind='source expression')
    a['rows'].append(item)
   elif record.get('existing_row_correction'):
    existing.update(item);existing['review_reasons']=record['uncertainty_types']
 for a in audit:
  for item in a.get('rows', []):
   row=bykey[item['entry_key']]
   if item.get('citation') != row[7]:
    item['archived_reference_citation_before_serialization_repair']=item.get('citation')
    item['citation']=row[7]; item['notes']=row[6]
 for a,c in zip(audit,census):
  assert a['source_id']==c['source_id']
  a['expression_census']=c
 return rows,audit

def main():
 p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');p.add_argument('--install',action='store_true');args=p.parse_args();rows,audit=prepare()
 if args.write or args.install:
  with (INSTALLED if args.install else PROPOSAL).open('w',newline='') as f:csv.writer(f).writerows(rows)
  (AUDIT if args.install else HERE/'proposal-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
 print(json.dumps({'source_units':len(audit),'rows':len(rows),'excluded':sum(a['status']!='ingested' for a in audit)}))
if __name__=='__main__':main()
