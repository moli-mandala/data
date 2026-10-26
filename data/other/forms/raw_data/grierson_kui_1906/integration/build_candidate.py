"""Integrate reviewed witnesses locally. Never writes outside integration/."""
import collections, copy, csv, hashlib, json, sys, unicodedata as ud
from pathlib import Path

HERE=Path(__file__).resolve().parent
Q=HERE.parent
DATA=next(p for p in Q.parents if (p/'tags.py').exists())
sys.path.insert(0,str(DATA))
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
from dialects import dialect_tag as make_dialect_tag
ALLOWED=GRAMMATICAL_TAGS|GENDER_TAGS
pins={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):
 p=Q/n;pins[n]=sha(p)
 return [json.loads(l) for l in p.read_text().splitlines()] if n.endswith('.jsonl') else json.loads(p.read_text())
def dump(n,x): (HERE/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def jsonl(n,x): (HERE/n).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
def text(x):return x if isinstance(x,str) else json.dumps(x,ensure_ascii=False)

def main():
 rows=[]; audit=[]; overlays=[]
 languages={r['ID'] for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
 mapping=read('source-lect-mapping-approved-20260927.json')
 routes={r['key']:r for r in mapping['mappings']}
 dialects={r['proposed_dialect_id']:r for r in mapping['mappings'] if r.get('proposed_dialect_id')}
 route_review=read('outside-comparands-routing-reuse-review-20260927.json')
 cmp_routes={r['source_unit_key']:r for r in route_review['decisions']}
 labels={r['input_label']:r for r in route_review['decisions']}
 ext={'external-Gondi':'Gondi','external-Kanarese':'Kannada','external-Tamil':'Tamil','external-Telugu':'Telugu'}
 # This is only a proposal for the new labels; independent final routing review is required.
 new_labels={'literary Telugu':'Telugu','Malayalam':'Malayalam','Tamil dialects (unspecified)':'Tamil','Kaikadi of Berar':'Kaikadi','Tulu':'Tulu'}
 def dialect_tag(d):
  if not d:return []
  r=dialects.get(d)
  if not r:raise ValueError(('unreviewed dialect',d))
  return [make_dialect_tag('Kui',d,r['proposed_name'])]
 def emit(key,lang,form,gloss,tags,note,citation,dialect=None,uncertainty=None):
  assert lang in languages,(key,lang)
  tags=list(dict.fromkeys(tags))
  assert not set(tags)-ALLOWED,(key,set(tags)-ALLOWED)
  if uncertainty:
   if 'uncertain' not in tags:tags.append('uncertain')
   note+=' Source uncertainty: '+text(uncertainty)
  if dialect=='grierson1906_kui_kalahandi':
   if 'uncertain' not in tags:tags.append('uncertain')
   note+=' Historical Kui/Kalahandi source route retained; modern Kuvi identification conflicts with this historical classification.'
  tags+=dialect_tag(dialect)
  row=[lang,'',form,gloss,'','',note.strip(),citation,'','',key,'','','',' '.join(tags)]
  assert len(row)==15 and form and key
  rows.append(row);return key
 def record(section,r,disposition,keys=None,**kw):
  key=r.get('source_unit_key',r.get('key'))
  audit.append(dict(source_unit_key=key,section=section,disposition=disposition,entry_keys=keys or [],source_record=r,**kw))
 def cite(page,locator):return f'grierson1906kui[p. {page}, {locator.replace(";", " and ")}]'
 # All physical table prompts survive, including one six-prompt reference bracket.
 table={r['prompt_id']:r for r in read('table-metadata-candidate.json')['readings']}
 prompts=read('table-prompt-inventory.jsonl'); byprompt={r['prompt_id']:r for r in prompts}
 for p in prompts:
  n=p['prompt_id'];key=p['key']
  if n in (133,136):record('table',p,'source-blank');continue
  source_n=n-16 if 195<=n<=200 else n
  r=table[source_n];tags=list(r['tags']);note=' '.join(r.get('source_notes_additions',[]))
  citation=cite(p['printed_page'],f'Khondmals table prompt {n}; English p. {p["english_printed_page"]}')
  if source_n!=n:
   tags=['fut' if t=='pres' else t for t in tags]
   note+=' Printed bracket for prompts 195–200 says As the present tense. This future use resolves the corresponding present prompt '+str(source_n)+'; no six independent printed forms are claimed.'
   citation+='; '+cite(byprompt[source_n]['printed_page'],f'Khondmals table prompt {source_n}')
  uncertainty=r.get('uncertainty') if r.get('current_uncertainty_tag') or 'uncertain' in tags else None
  keys=[]
  for i,f in enumerate(r['forms'],1):
   ek=key+(f':v{i}' if len(r['forms'])>1 else '')
   qualifier=r.get('form_qualifiers',[])
   gloss=p['english_prompt']
   if len(qualifier)==len(r['forms']):gloss=qualifier[i-1]+' '+gloss
   keys.append(emit(ek,'Kui',f,gloss,tags,note,citation,'grierson1906_kui_khondmals',uncertainty))
  record('table',dict(p,reviewed_reading=r),'emitted',keys,physical_token=source_n==n,cross_reference_source_key=byprompt[source_n]['key'] if source_n!=n else None)
 # Grammar and its explicit shared-token interpretations.
 supplement=read('grammar-scope-supplement-20260927.json')
 alternatives=read('root-grammar-alternative-emission-review-20260927.json')
 alternative_map={r['source_unit_key']:r for r in alternatives['decisions']}
 updates=collections.defaultdict(list)
 for u in supplement['existing_unit_scope_updates']:
  for key in u['source_unit_keys']:updates[key].append(u)
 for r in read('grammar-metadata-candidate.jsonl'):
  key=r['source_unit_key'];route=routes.get(r['route_key'],{});lang=route.get('proposed_base') or ext.get(r['route_key'])
  if not r['form']:record('grammar',r,'source-blank');continue
  if not lang:record('grammar',r,'identity-hold',reason=route.get('disposition','Unresolved joint language attribution'),scope_updates=updates[key]);continue
  note=r['source_note']
  if r.get('witness'):note+=' Source witness: '+r['witness']+'.'
  note+=' '+' '.join(u['source_note'] for u in updates[key])
  decision=alternative_map.get(key)
  if decision:
   assert r['form']==decision['old'],key
   note+=' '+decision.get('note','')
  forms=decision['forms'] if decision else [r['form']]
  eks=[]
  for i,form in enumerate(forms,1):
   eks.append(emit(key+(f':v{i}' if decision else ''),lang,form,r['gloss'],r['tags'],note,cite(r['printed_page'],r['block']+' '+key.rsplit(':',1)[1]),route.get('proposed_dialect_id'),r.get('uncertainty')))
  record('grammar',r,'emitted',eks,scope_updates=updates[key],alternative_decision=decision)
 for r in supplement['units']:
  if not r['lexical_emission']:record('grammar-supplement',r,'documentary');continue
  ek=emit(r['source_unit_key'],r['language_id'],r['form'],r['gloss'],r['tags'],r['source_note'],cite(r['printed_page'],'explicit Kanarese identity comparison'))
  record('grammar-supplement',r,'emitted',[ek],physical_token=False,literal_source_key=r['literal_source_key'])
 # Every physical atom maps to its approved joined emission. No phantom tail heads.
 atoms=read('specimens/atomic-emissions-reviewed.jsonl')
 physical=read('specimens/interlinear-reviewed.jsonl');atommap={}
 for r in atoms:
  key=r['emission_key'];citation='; '.join(cite(int(k.split(':')[1][1:]),':'.join(k.split(':')[2:])) for k in r['physical_source_keys'])
  note=r['source_note']+' '+r.get('boundary_decision','')
  if r['unit_type']=='title_atom':note+=' Printed title word with source interlinear meaning.'
  emit(key,'Kui',r['emission_form'],r['emission_gloss'],r['tags'],note,citation,r['source_dialect'],r.get('typed_uncertainties'))
  for k in r['physical_source_keys']:
   assert k not in atommap;atommap[k]=key
 assert len(atommap)==len(physical)==890
 for r in physical:record('specimen-physical-atom',r,'emitted' if atommap[r['source_unit_key']]==r['source_unit_key'] else 'joined-physical-fragment',[atommap[r['source_unit_key']]])
 for r in read('specimens/expression-emissions-reviewed.jsonl'):
  d={40:'grierson1906_kui_gumsur_bod',41:'grierson1906_kui_khondmals',42:'grierson1906_kui_khondmals',43:'grierson1906_kui_kalahandi'}[r['specimen']]
  note='Source-aligned interlinear translation.'
  key=emit(r['source_unit_key'],'Kui',r['emission_form'],r['emission_gloss'],r['tags'],note,cite(', '.join(map(str,r['printed_pages'])),f'specimen {r["specimen"]} {r["source_unit_key"].rsplit(":",1)[1]}'),d,r.get('typed_uncertainties'))
  record('specimen-expression',r,'emitted',[key],physical_token=False,aligned_atom_entry_keys=list(dict.fromkeys(atommap[k] for k in r['atom_keys'])))
 aliases={'ablative':'abl','dative':'dat','genitive':'gen','locative':'loc','past':'pret','plural':'pl','singular':'sg','present':'pres','pronoun':'pron','numeral':'num','whole-expression':'multiword-expression','demonstrative-base':'demonstrative'}
 for r in read('outsidechapter-independent-reviewed.jsonl'):
  tags=[aliases.get(t,t) for t in r['explicit_analysis']]
  key=emit(r['source_unit_key'],'Kui',r['form'],r['gloss'],tags,r['note']+' Source attribution: '+r['source_lect']+'.',cite(r['printed_page'],'outside-chapter '+r['source_unit_key'].rsplit(':',1)[1]),uncertainty=r.get('uncertainty'))
  record('outside-target',r,'emitted',[key])
 rootreview=read('root-comparand-supplement-review-20260927.json')
 for original in read('outside-comparands-reviewed-candidate-20260927.jsonl'):
  r=copy.deepcopy(original);key=r['source_unit_key'];status=r.get('status','');label=r['source_language_label'];route=cmp_routes.get(key) or labels.get(label) or {}
  if key=='grierson1906kui:p293:outside-comparand:16':
   r['tags']=[t for t in r['tags'] if t!='uncertain'];r['note']='Root independently confirms two-hump m in High Tamil plural discussion. Earlier n/m question retained only in historical audit.'
  if key=='grierson1906kui:p296:outside-comparand:16':r['note']+=' Visible lower mark under d also remains uncertain; provisional plain d must not be treated as confident normalization.'
  if r!=original:overlays.append({'source_unit_key':key,'old':original,'new':r,'authority':'root-comparand-supplement-review-20260927.json'})
  if 'hypothesis' in status or 'generalized' in label:record('outside-comparand',r,'documentary');continue
  if 'unresolved-reference' in status:record('outside-comparand',r,'reference-hold');continue
  reuse=route.get('reuse_key')
  if reuse:record('outside-comparand',r,'reused-existing-occurrence',[reuse],reuse_evidence=route);continue
  lang=route.get('language_id') or new_labels.get(label)
  if not lang:record('outside-comparand',r,'identity-hold',reason=route.get('note','No unique approved registry mapping'));continue
  if route.get('status')=='existing_historical_dialect_proposal':
   record('outside-comparand',r,'identity-hold',reason='Historical Koi/Gondi route proposed but final approval pending',route_proposal=route);continue
  note=r['note']+' Source language label: '+label+'.'
  tags=r['tags'][:]
  key=emit(key,lang,r['form'],r['gloss'],tags,note,cite(r['printed_page'],'outside comparative occurrence '+key.rsplit(':',1)[1]))
  if route.get('dialect_tag'):rows[-1][14]+=' '+route['dialect_tag']
  if label=='Kaikadi of Berar':rows[-1][14]+=' dialect:Kaikadi:lsi1906-kaikadi:Berar'
  record('outside-comparand',r,'emitted',[key],route=route or {'language_id':lang,'scope':'new label mapping pending independent final review'})
 graph_review=read('independent-integration-routing-graph-review-20260927.json')
 graph_approval=read('root-component-relations-review-20260927.json')
 assert pins['independent-integration-routing-graph-review-20260927.json']==graph_approval['independent_report_sha256']
 accepted_components={r['child_key']:r for r in graph_approval['accepted']}
 bykey={r[10]:r for r in rows}
 graph_accounting=[]
 scholarly_claims={
 'negative-conjunctive-example:1':'The source forms this negative conjunctive by adding ār-aṅgā to the past conjunctive participle.',
 'compound-example:4':'The source forms this compound perfect with the conjunctive participle and inflected copula.',
 'passive-example:1':'The source forms the passive with the verbal noun and become, and qualifies this construction as uncommon.',
 'compound-example:1':'The source describes a compound tense formed with a conjunctive participle and the copula; the separately printed participle uses gi(pk)i notation.',
 'compound-example:2':'The source describes a compound tense formed with a conjunctive participle and past copula; the separately printed participle uses gi(pk)i notation.',
 'adjective-example:1':'The source derives this adjective by adding gaṭṭa to a noun.',
 'verbal-noun-example:1':'The source derives this verbal noun from a relative participle with a third-person pronominal ending.',
 'verbal-noun-example:2':'The source derives this verbal noun from a relative participle with a third-person pronominal ending.'}
 for decision in graph_review['graph_decisions']:
  d=copy.deepcopy(decision);child=bykey[d['child_key']]
  assert child[2]==d['child_form']
  parents=d.get('approved_emission_parent_keys',d['parent_keys'])
  if d['disposition'] in ['accept_ordered_components','accept_after_approved_alternate_emission']:
   expected=['giwa','āti'] if d['disposition']=='accept_after_approved_alternate_emission' else d['parent_forms']
   assert [bykey[k][2] for k in parents]==expected,(d,parents)
   assert parents==accepted_components[d['child_key']]['ordered_parent_keys']
   assert expected==accepted_components[d['child_key']]['parent_forms']
   assert all(bykey[k][0]==child[0] for k in parents)
   child[13]='|'.join(parents)
   d['emitted_parent_keys']=parents
  else:d['emitted_parent_keys']=[]
  d['edge_type']='component' if d['emitted_parent_keys'] else None
  child[6]+=' '+scholarly_claims[':'.join(d['child_key'].split(':')[-2:])]
  graph_accounting.append(d)
 assert sum(len(d['emitted_parent_keys']) for d in graph_accounting)==graph_approval['component_edge_count']==6
 display_revision_path=HERE/'display-note-revisions.json'
 display_revisions=json.loads(display_revision_path.read_text())
 pins['integration/display-note-revisions.json']=sha(display_revision_path)
 for d in display_revisions['changes']:
  row=bykey[d['entry_key']]
  assert row[6]==d['old_notes'] and row[14]==d['old_tags'],d['entry_key']
  row[6]=d['new_notes'];row[14]=d['new_tags']
 for a in audit:
  a['display_revisions']=[d for d in display_revisions['changes'] if d['entry_key'] in a['entry_keys']]
 # Validate identity and physical coverage before writing anything.
 keys=[r[10] for r in rows];assert len(set(keys))==len(keys)
 assert len({a['source_unit_key'] for a in audit})==len(audit)
 for a in audit:
  if a['disposition'] not in ['reused-existing-occurrence']:assert set(a['entry_keys'])<=set(keys)
 assert sum(a['section']=='table' for a in audit)==241
 assert sum(a['section']=='grammar' for a in audit)==432
 assert sum(a['section']=='specimen-physical-atom' for a in audit)==890
 csvpath=HERE/'combined-candidate.csv'
 with csvpath.open('w',newline='') as f:csv.writer(f).writerows(rows)
 for a in audit:
  a['graph_claims']=[d for d in graph_accounting if d['child_key'] in a['entry_keys']]
 jsonl('combined-audit.jsonl',audit);dump('integration-overlays.json',overlays);dump('graph-accounting.json',graph_accounting)
 # Regenerate preservation inventory from actual output, not obsolete first readings.
 profile={' ':'#'};counts=collections.Counter()
 for row in rows:
  tokens=[]
  for c in ud.normalize('NFD',row[2]):
   if ud.combining(c) and tokens:tokens[-1]+=c
   else:tokens.append(c)
  counts.update(row[2])
  for t in tokens:
   if t!=' ':
    display=ud.normalize('NFC',t).lower().replace('ṅ','ŋ')
    profile[t]=display;profile[ud.normalize('NFC',t)]=display
 (HERE/'literal-preservation.txt').write_text('Grapheme\tIPA\n'+''.join(f'{k}\t{v}\n' for k,v in sorted(profile.items())))
 dump('unicode-inventory.json',[{'symbol':k,'codepoint':f'U+{ord(k):04X}','name':ud.name(k,'UNNAMED'),'count':v} for k,v in sorted(counts.items())])
 metadata={'file':'combined-candidate.csv','defaults':{'transcription':{'profile':'kui-integration-preservation','input':'form','preserve_hyphens':True},'identity':{'dedupe_by_entry_key':True},'forms':{'split_alternates':False},'gloss':{'grammar_tags':False}},'sources':{'grierson1906kui':{'reference':{'etymology_provenance':'source','ocr':True},'forms':{'split_alternates':False},'gloss':{'grammar_tags':False}}}}
 dump('metadata-proposal.json',metadata)
 dump('combined-candidate.yaml',metadata)
 dump('status.json',{'status':'combined review candidate, not installation-ready','input_hashes':pins,'rows':len(rows),'audit_units':len(audit),'sections':dict(collections.Counter(a['section'] for a in audit)),'dispositions':dict(collections.Counter(a['disposition'] for a in audit)),'languages':dict(collections.Counter(r[0] for r in rows)),'profile_rules':len(profile),'outputs':{p.name:sha(p) for p in [csvpath,HERE/'combined-audit.jsonl',HERE/'literal-preservation.txt',HERE/'metadata-proposal.json',HERE/'combined-candidate.yaml',HERE/'graph-accounting.json',HERE/'integration-overlays.json',Path(__file__)]},'scope':'Complete chapter and 241-prompt Khondmals column plus explicitly bounded outside comparisons; documentary/reference/identity dispositions retained.','pending':['Independent final routing/config review','Actual parser with proposed dialect overlay and complete profile','Fresh independent output audit','Canonical metadata/Bib/reference registration','Full build, database and browser deferred by instruction'],'canonical_changes':False})
 print(len(rows),len(audit),collections.Counter(a['disposition'] for a in audit))

if __name__=='__main__':main()
