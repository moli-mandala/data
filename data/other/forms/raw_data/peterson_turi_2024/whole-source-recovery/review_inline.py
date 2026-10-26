"""Reconcile source-wide inline evidence after complete prose/span census."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
src=HERE/'inline-first-reading.jsonl';rows=[json.loads(x) for x in src.read_text().splitlines()]
fixes={
 'p277:inline:imperative-components-1':('middle voice marker',''),
 'p277:inline:imperative-components-2':('second-person singular marker',''),
 'p281:inline:grierson-numerals-1':('three',''),
 'p281:inline:grierson-numerals-2':('four',''),
 'p275:inline:sar-give':('give-appl.pst=3.obj-y-fin=3sg.subj','Exact analysis from cited example25; the contextual order is not added to lexical meaning.')}
changes=[]
for r in rows:
 r['review_status']='owner_original_page_reviewed' if r['printed_page'] in list(range(266,280))+[281,282,283] else 'native_context_reviewed_pending_page_image'
 suffix=r['source_unit_key'].split(':',1)[1]
 if suffix in fixes:
  g,n=fixes[suffix];changes.append(dict(key=r['source_unit_key'],old=r['gloss'],new=g));r['gloss']=g
  if n:r['note']=n
 # Bracket/parenthetical phonetic notation and surrounding context remain in exact native evidence.
 r['context_evidence']=f'native-pages/p{r["printed_page"]}.txt'
# Parenthetical reconstructed forms are explicit analyses, not attested speech.
for p,n in [(272,13),(274,21)]:
 rows.append(dict(source_unit_key=f'peterson2024turi:p{p}:reconstruction:example{n}',printed_page=p,pdf_page=p-260,form='*ɲɛl-tad=i=a=iŋ',gloss='see-pst.act=3sg.obj=fin=1sg.subj',kind='reconstructed_analysis',lect='Odisha',note='Printed parenthetical < *...: underlying analysis only. Does not replace the attested surface ɲɛl-tad=i=iŋ.',review_status='owner_original_page_reviewed'))
# Source sound inventory is evidence for profile interpretation, not vocabulary.
for p,k,f,n in [(266,'consonant-inventory','p pʰ b bʱ t tʰ d dʱ ʈ ɖ ʈʰ c ɟ k kʰ g gʱ ʔ ʧ ʤ ʧʰ ʤʱ m n ɳ ɲ ŋ l ɾ ɽ s h ʋ j','Table1 consonant chart; native layout visually reviewed, not a lexical list.'),(267,'vowel-inventory','i u (ə) ɔ ɛ ɑ','Table2; ə phonemic status explicitly uncertain, possible allophone ofɑ; no invented lexical vowel examples.'),(267,'diphthongs','ə̯̯ɛ, i̯̯a','Source duplicated nonsyllabic marks retained in raw evidence; chart sequences, not words.'),(266,'preglottalization','ʔb; ʔb˺m','General phonetic classes from comparative discussion, no lexeme.'),(283,'preglottalization','ʔb˺m','General class repeated in computational preprocessing discussion, no lexeme.')]:
 rows.append(dict(source_unit_key=f'peterson2024turi:p{p}:phonology:{k}',printed_page=p,pdf_page=p-260,form=f,gloss='',kind='phonological_inventory',lect='source scope',note=n,review_status='owner_original_page_reviewed'))
(HERE/'inline-reviewed.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
report=dict(status='owner_lexical_prose_reviewed_context_pages_pending',input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),units=len(rows),changes=changes,method='Read all42 native pages; compare every DejaVuSerifCondensed-Ita prose span against inventory, plus non-italic bracketed phonetic words. View original lexical pages266–279,281–283 individually. Nonlexical contextual/bibliography pages remain in page census.',controls='Foreign, unassigned comparison, contextual names, reconstruction and sound inventories remain documentary; no target attribution invented.',independent_review='pending')
(HERE/'owner-inline-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(f'{len(rows)} inline/documentary units saved')
