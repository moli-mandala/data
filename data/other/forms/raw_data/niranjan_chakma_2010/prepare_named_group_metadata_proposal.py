"""Source-qualified group provenance, without inventing dialects or geography."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parents[4]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 languages=DATA/'cldf/languages.csv';dialects=DATA/'cldf/dialects.csv'
 registry=list(csv.DictReader(languages.open()));dregistry=list(csv.DictReader(dialects.open()))
 matches=[r for r in registry if any(t in (' '.join(r.values())).lower() for t in ['tanchang','tangchang','tongchang'])]
 assert not matches,'Registry changed; review Tanchangya routing explicitly.'
 assert any(r['ID']=='Chakma' for r in registry)
 source=ROOT/'named-groups-reviewed-reading-proposal.jsonl';hom=ROOT/'homophones-reviewed-reading-proposal.jsonl'
 rows=[json.loads(x) for x in source.read_text().splitlines()];assert len(rows)==145
 extra=[r for r in map(json.loads,hom.read_text().splitlines()) if 'তঞ্চঙ্গ্যা' in ' '.join(r.get('source_qualifiers',[]))]
 assert len(extra)==1
 proposed=[]
 for r in rows+extra:
  label=r.get('source_group_label') or '; '.join(r.get('source_qualifiers',[]));explicit='তঞ্চঙ্গ্যা' in label
  proposed.append({'source_unit_key':r['source_unit_key'],'raw_source_group_label':label,'proposed_base_id':None if explicit else 'Chakma','source_language_name':'Tanchangya' if explicit else 'Chakma','dialect_tags':[],'new_dialect_registry_records':[],'identity_status':'explicit Tanchangya attribution; separate base absent from current registry' if explicit else 'Chakma chapter attribution; exact group label retained as provenance','required_source_note':('Source explicitly attributes this witness to '+label+'. ' + ('A shared co-attribution remains one physical witness; no automatic duplicate or Chakma reassignment. ' if 'ও' in label and explicit else '')+'The source records group-based pronunciation; no geographic dialect, coordinates or equivalence between similar group names is inferred.'),'label_scope':'explicit language/co-attribution' if explicit else ('missing label' if label=='—' else 'source group/region attribution'),'source_evidence':{'pdf_page':r['pdf_page'],'printed_page':r['printed_page'],'printed_group':r.get('printed_group'),'group_cell':r.get('group_cell'),'section_heading_pdf71':'গঝাভিত্তিক শব্দের উচ্চারণ ভিন্নতা','column_labels':['শব্দ','গঝা','শব্দার্থ']},'accepted_for_installation':False})
 assert len(proposed)==146
 result={'status':'complete keyed metadata proposal; no registry writes','inputs':{source.name:sha(source),hom.name:sha(hom),'cldf/languages.csv':sha(languages),'cldf/dialects.csv':sha(dialects)},'source_heading_review':{'pdf_page':71,'printed_page':69,'method':'Full original page atPDFium2x; heading and all three columns inspected. No separate explanatory prose before table. ContentsPDF6 confirms same scope.','decision':'A group-associated pronunciation table does not by itself establish geographic dialect identity. Preserve named gojha and regional labels in provenance; no unsupported dialect registry creation.'},'existing_Tanchangya_registry_matches':matches,'coverage':{'named_group_rows':145,'explicit_homophone_Tanchangya_rows':1,'all_keyed_records':146,'Tanchangya_explicit_records':sum(r['source_language_name']=='Tanchangya' for r in proposed),'dialect_tags_proposed':0,'raw_label_counts':dict(Counter(r['raw_source_group_label'] for r in proposed))},'records':proposed,'pending':['Separate Tanchangya language identity/base proposal needs primary identity evidence and registry approval; do not assign Chakma as a fallback.','Mixed Kuro Kutya/Tanchangya occurrence retains shared source attribution until mapping semantics reviewed.','তোন্যা/তন্যা labels are not silently equated with Tanchangya; কলিমা/মুলিমা and বোরবো/বরবো spelling differences remain literal.','Root approves provenance routing before final source output.'],'canonical_changes':False}
 (ROOT/'named-group-metadata-proposal.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result['coverage'],ensure_ascii=False))
if __name__=='__main__':main()
