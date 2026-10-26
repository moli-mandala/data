"""Bounded real-parser checks of frozen source preview; no compiled writes."""
import argparse,csv,hashlib,io,json,re,sys
from pathlib import Path
P=Path(__file__).resolve().parent
D=P.parents[4]
sys.path.insert(0,str(D))
import make_cldf,source_meta,tags
from pybtex.database import parse_file as parse_bib

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preview-dir',type=Path,default=P);ap.add_argument('--input-csv',type=Path);args=ap.parse_args()
 path=args.input_csv or args.preview_dir/'whole-source-export-preview.csv';rows=list(csv.reader(path.open()))
 pinned=sha(path);errors=io.StringIO()
 parsed,stats=make_cldf.parse_file(str(path),errors,name='20260921-niranjan-chakma')
 differences=[]
 fields={'lang':0,'param':1,'form':2,'old_form':2,'gloss':3,'native':4,'ipa':5,'notes':6,'source':7,'etymology':9,'entry_key':10,'variant_of_key':11,'borrowed_from_key':12,'derivation_parent_keys':13}
 for actual,expected in zip(parsed,rows):
  for attr,i in fields.items():
   if getattr(actual,attr)!=expected[i]:differences.append({'key':expected[10],'field':attr,'expected':expected[i],'actual':getattr(actual,attr)})
  if set(actual.tags.split())!=set(expected[14].split()):differences.append({'key':expected[10],'field':'tags','expected':expected[14],'actual':actual.tags})
 registry={r['ID'] for r in csv.DictReader((D/'cldf/languages.csv').open())}
 unknownlangs=sorted({r[0] for r in rows}-registry)
 knowntags=tags.GENDER_TAGS|tags.GRAMMATICAL_TAGS|tags.ERA_TAGS
 unknowntags=sorted({t for r in rows for t in r[14].split()}-knowntags)
 refs=parse_bib(str(D/'cldf/sources.bib')).entries
 citations={part.split('[',1)[0] for r in rows for part in r[7].split(';')}
 unknownrefs=sorted(citations-set(refs));meta=source_meta.load()
 auxiliary=['taittiriya_aranyaka_work','caryapada_work','srikrsnakirtana_work','ksemendra_brhatkathamanjari_work','dewan_chakma2003chakma_first_lesson']
 report={'status':'passed' if len(parsed)==len(rows) and not(differences or errors.getvalue() or unknownlangs or unknowntags or unknownrefs) else 'requires review','rows':len(rows),'parsed_rows':len(parsed),'stats':stats,'parser_errors':errors.getvalue(),'field_differences':differences,'unknown_languages':unknownlangs,'unknown_tags':unknowntags,'unknown_references':unknownrefs,'transcription_policy':meta.file_defaults('20260921-niranjan-chakma.csv')['transcription'],'auxiliary_references_registered':{k:k in refs and k in meta.source_owner for k in auxiliary},'source_metadata':{'ocr':meta.flag('niranjan2010chakma','reference','ocr'),'dedupe_by_entry_key':meta.flag('niranjan2010chakma','identity','dedupe_by_entry_key'),'split_alternates':meta.flag('niranjan2010chakma','forms','split_alternates')},'sha256':{'preview':pinned,'languages':sha(D/'cldf/languages.csv'),'yaml':sha(D/'data/other/forms/20260921-niranjan-chakma.yaml'),'canonical':sha(D/'data/other/forms/20260921-niranjan-chakma.csv'),'validator':sha(Path(__file__))},'scope':'Actual make_cldf.parse_file with installed source routing, no monkeypatch; no database/CLDF build or canonical source write.'}
 assert sha(path)==pinned
 (args.preview_dir/'whole-source-export-parser-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['status','rows','parsed_rows','unknown_languages','unknown_tags','unknown_references']}));print('field differences',len(differences))
if __name__=='__main__':main()
