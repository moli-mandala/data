"""Prove source-marker correction continuity against immutable revision2."""
import csv
import hashlib
import json
from pathlib import Path
import fitz

HERE=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p): return {r[10]:r for r in csv.reader(p.open(newline=''))}
old=rows(HERE/'revision2/proposed.csv');new=rows(HERE/'proposed.csv')
assert old.keys()==new.keys()
expected={
 'nair2013adiya:p5:plural-morphology:slot1':('-a','a'),
 'nair2013adiya:p5:plural-morphology:slot4':('-kaḷ','–kaḷ'),
 'nair2013adiya:p6:addressee-sex-markers:slot1':('-va','–va'),
 'nair2013adiya:p6:addressee-sex-markers:slot2':('-ma','–ma'),
 'nair2013adiya:p6:affinal-address-mi:slot1':('-mi','–mi'),
 'nair2013adiya:p7:affinal-address-i:slot1':('-i','–i'),
 'nair2013adiya:p7:negative-kaani:slot1':('-kaaṇi','–kaaṇi'),
 'nair2013adiya:p7:negative-a:slot1':('-a','–a'),
}
fields=['Language_ID','Parameter_ID','Form','Gloss','Native','Phonemic','Notes','Source','Cognateset_ID','Etymology','Entry_Key','Variant_Of_Key','Borrowed_From_Key','Derivation_Parent_Keys','Tags']
changes=[]
for key,a in old.items():
 b=new[key]; diff=[i for i in range(15) if a[i]!=b[i]]
 if 2 in diff: assert expected[key]==(a[2],b[2])
 assert set(diff)<={2,6},(key,diff)
 if diff: changes.append(dict(entry_key=key,changes={fields[i]:dict(old=a[i],new=b[i]) for i in diff}))
assert {r['entry_key'] for r in changes if 'Form' in r['changes']}==expected.keys()
pdf=HERE.parents[5]/'tmp/pdfs/tribal-languages-kerala-nair2013.pdf';doc=fitz.open(pdf)
source_lines=[]
for n in (5,6,7):
 for line in doc[n-1].get_text().splitlines():
  if any(token in line for token in ['a-suffix','–kal','–va','–mi','–i suffix','–kaan','–a negative','base en-','n\uf019a-']):
   source_lines.append(dict(printed_page=n,pdf_page=n,raw_pdf_text=line))
result=dict(status='source-wide bound-marker sweep complete; eight exact source-literal repairs, key/tag/other-field continuity passes',
 prior_csv_sha256=sha(HERE/'revision2/proposed.csv'),current_csv_sha256=sha(HERE/'proposed.csv'),
 prior_freeze_sha256=sha(HERE/'revision2/review-freeze.json'),source_pdf_sha256=sha(pdf),
 exact_form_corrections=8,changed_rows=len(changes),changes=changes,source_lines=source_lines,
 unchanged_bound_forms=['en- (table and p5 prose)','ña- (p5 paradigm)'],
 policy='A bound grammatical analysis does not license inventing a leading ASCII hyphen. Preserve printed en dash as U+2013, ASCII hyphen as U+002D, and bare a as bare a. Tags use keyed source categories rather than the shape of a leading mark.',
 metadata_changes=['Five Bib titles protected with nested braces; same words/authors/dates.',
                   'Source-lect Glottocode blank; existing Ravula parent ravu1237 retained in identity evidence.'],
 canonical_changes=False)
(HERE/'revision3-marker-continuity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('8 Form corrections; all tags, keys and non-Notes fields preserved')
