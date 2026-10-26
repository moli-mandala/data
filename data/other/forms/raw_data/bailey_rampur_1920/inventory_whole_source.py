"""Bounded paired Rampuri/Baghi chapter inventory, preserving existing inputs."""
import csv,hashlib,json
from pathlib import Path
import fitz

P=Path(__file__).resolve().parent
D=P.parents[4]
PDF=D.parent/'tmp/pdfs/bailey1920/bailey1920.pdf'
EXPECTED='7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39'

def main():
 h=hashlib.sha256()
 with PDF.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 assert h.hexdigest()==EXPECTED
 doc=fitz.open(PDF)
 pages=[*range(113,119),*range(130,149)]
 units=[dict(printed_page=n,pdf_page=n+26,role='boundary'if n in{130,148}else'shared-introduction'if n<119 else'paired-chapter',raw_text=doc[n+25].get_text(),method='Existing embedded text, locator aid only; literal visual transcription required.')for n in pages]
 (P/'whole-source-text-inventory-20260927.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n'for u in units))
 for lect in ('rampur','baghi'):
  package=P.parent/f'bailey_{lect}_1920'
  for name,source in [('legacy-before-whole-recovery.csv',D/f'data/other/forms/20260925-bailey-{lect}.csv'),('legacy-before-whole-recovery-audit.jsonl',package/'audit.jsonl')]:
   target=package/name
   if not target.exists():target.write_bytes(source.read_bytes())
 report=dict(status='whole-source census in progress, no canonical writes',owner='rohru_finish',pdf=str(PDF.relative_to(D.parent)),sha256=EXPECTED,pages=len(doc),printed_to_pdf_offset=26,inventory_pages=pages,packages=['bailey_rampur_1920','bailey_baghi_1920'],provisional_scope='Shared introduction113–118; paired chapter131–147, boundaries130/148. Original headings and lect-specific rules require full visual census.')
 (P/'whole-source-census-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 ledger=D/'source_checklists/20260925-source-completeness-ledger.tsv'
 lines=list(csv.reader(ledger.open(),delimiter='\t'))
 for row in lines:
  if row and row[0]in{'20260925-bailey-rampur','20260925-bailey-baghi'}:
   assert row[4]in{'queued','rohru_finish'},row
   row[3]='confirmed_partial_whole_source_recovery_in_progress';row[4]='rohru_finish'
 with ledger.open('w',newline='')as f:csv.writer(f,delimiter='\t',lineterminator='\n').writerows(lines)
 print(json.dumps(report))

if __name__=='__main__':main()
