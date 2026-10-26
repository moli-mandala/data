"""Derive two separate language proposals from cached primary Glottolog trees."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;DATA=ROOT.parents[4];WORKSPACE=DATA.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 registry=DATA/'cldf/languages.csv';frontend=WORKSPACE/'jambu-static/src/lib/clades.ts';existing=list(csv.DictReader(registry.open()))
 names=[('Tanchangya','tang1330','tnv','Eastern','indo1323'),('Chenchu','chen1255','cde','S. Dravidian II','sout3139')]
 proposals=[];routes=[];evidence=[]
 named=json.loads((ROOT/'named-group-metadata-proposal.json').read_text());numbered=json.loads((ROOT/'numbered-final-identity-review-proposal.json').read_text())
 for name,code,iso,clade,ancestor in names:
  path=ROOT/'identity-authorities'/f'{code}.html';text=path.read_text();start=text.index("GLOTTOLOG3.Tree.init('tree', ")+len("GLOTTOLOG3.Tree.init('tree', ");tree,_=json.JSONDecoder().raw_decode(text[start:])
  def find(node,chain):
   chain=chain+[{k:node[k] for k in ['id','iso','level','label']}]
   if node['id']==code:return chain
   for child in node['children']:
    result=find(child,chain)
    if result:return result
  chain=next(x for x in (find(n,[]) for n in tree) if x)
  assert chain[-1]['iso']==iso and chain[-1]['level']=='language'
  assert ancestor in {x['id'] for x in chain}
  assert clade in frontend.read_text()
  assert not any(r['ID']==name or r['Glottocode']==code for r in existing),'Already registered; reconcile instead of duplicate.'
  record={'ID':name,'Name':name,'Glottocode':code,'Latitude':'','Longitude':'','Clade':clade,'Location':'','Quality':''}
  proposals.append(record);evidence.append({'ID':name,'authority_url':f'https://glottolog.org/resource/languoid/id/{code}','retrieved':'2026-09-27','cached_html':str(path.relative_to(ROOT)),'sha256':sha(path),'classification_chain':chain,'ISO_639_3':iso,'ISO_direct_access':'SIL code page returned403; Glottolog primary record explicitly links and supplies ISOcode.','name_policy':'Tanchangya is listed as alternative of authority Tangchangya; source Bengali তঞ্চঙ্গ্যা matches this named language.' if name=='Tanchangya' else 'Source abbreviation চে expands চেঞ্চু; independently researched exact language, not a Telugu substitute.','coordinate_policy':'No source-site coordinates or location-quality rating supplied; all remain blank. Catalogue coordinates are not adopted as elicitation locations.'})
  if name=='Tanchangya':
   selected=[r for r in named['records'] if r['source_language_name']=='Tanchangya'];assert len(selected)==3
   for r in selected:routes.append({'source_unit_key':r['source_unit_key'],'proposed_base_id':name,'source_label':r['raw_source_group_label'],'required_source_note':'Explicit source Tanchangya attribution retained as separate language. '+('Shared attribution to Kuro Kutya also retained; one physical witness, no invented duplicate.' if 'ও' in r['raw_source_group_label'] else ''),'accepted_for_installation':False})
  else:
   selected=[r for r in numbered['comparators'] if r.get('new_registry_proposal',{} ) and r['new_registry_proposal']['ID']=='Chenchu'];assert len(selected)==4
   for r in selected:routes.append({'source_unit_key':r['comparison_key'],'proposed_base_id':name,'source_label':r['raw_label'],'required_source_note':'Source Chenchu comparison only; no inferred ancestry, donor link, or Telugu equivalence.','accepted_for_installation':False})
 out={'status':'primary-authority verified language registry proposal; root installation approval pending','records':proposals,'routes':routes,'primary_evidence':evidence,'inputs':{'cldf/languages.csv':sha(registry),'jambu-static/src/lib/clades.ts':sha(frontend),'named-group-metadata-proposal.json':sha(ROOT/'named-group-metadata-proposal.json'),'numbered-final-identity-review-proposal.json':sha(ROOT/'numbered-final-identity-review-proposal.json')},'dialect_records':[],'canonical_changes':False}
 (ROOT/'language-registry-proposal.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 with (ROOT/'languages-proposed.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(proposals[0]));w.writeheader();w.writerows(proposals)
 print(json.dumps({'records':2,'routes':len(routes),'clades':[r['Clade'] for r in proposals]}))
if __name__=='__main__':main()
