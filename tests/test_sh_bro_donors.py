import csv,importlib.util,json,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/other/params/raw_data'
def audit():return json.loads((RAW/'20260910-sh-bro-donors-audit.json').read_text())
def test_reproducible_source_subset():
 s=importlib.util.spec_from_file_location('sh_bro_donors',RAW/'sh_bro_donors.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 assert m.OUTPUT.read_bytes()==m.render().encode()
 assert len(audit())==123
 assert sum(q['Status']=='install' for q in audit())==94
 assert sum(q['Status']=='reuse-existing-supplement' for q in audit())==29

def test_source_transcription_and_homonyms():
 d=audit();h={q['Form']:q for q in d}
 assert h['māl']['Gloss']=='property; wealth; goods'
 assert h['kam']['ResolvedParameter']=='loan-shinaic-phal-214'
 assert h['safar']['Gloss']=='journey' and h['ṣafar']['Gloss']=='Safar month'
 assert h['ā ʿīna']['Original']=='aa ʿiina' and h['ā ʿīna']['TranscriptionReview']
 assert h['ṣikárk']['Language_ID']=='Bur'
 assert h['γam']['Language_ID']=='Pers'
 langs={q['ID'] for q in csv.DictReader((ROOT/'cldf/languages.csv').open())}
 for q in d:
  assert q['Language_ID'] in langs
  assert unicodedata.normalize('NFC',q['Form'])==q['Form'] and '\ufffd' not in q['Form']
  assert q['Evidence'] and q['Source']
  for s in q['Source'].split(';'):assert s.count('[')==s.count(']')

def test_compiled_donor_heads():
 forms={q['ID']:q for q in csv.DictReader((ROOT/'cldf/forms.csv').open())}
 aliases={q['Legacy_ID']:q['Form_ID'] for q in csv.DictReader((ROOT/'cldf/form-id-aliases.csv').open())}
 refs={q['ID'] for q in csv.DictReader((ROOT/'cldf/references.csv').open())}
 for q in audit():
  key=q['ID'] if q['Status']=='install' else q['ResolvedParameter'];ident=aliases.get(key,key)
  assert ident in forms,key
  f=forms[ident];assert f['Language_ID']==q['Language_ID'] and f['Status']=='entry'
  if q['Status']=='install':assert f['Form']==q['Form'] and f['Gloss']==q['Gloss']
  for citation in f['Source'].split(';'):assert citation.split('[',1)[0] in refs
