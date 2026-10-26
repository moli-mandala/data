"""Serialize full-page independent drafts; never install or silently accept OCR."""
import csv,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
S=json.loads((P/'reverse-chapter-scope-review-20260927.json').read_text())
expect={r['printed_page']:r for r in S['pages']}
def main():
 out=[];coverage=[]
 for f in sorted((P/'reverse-first-readings').glob('p*.tsv')):
  page=int(f.stem[1:]);rows=list(csv.DictReader(f.open(),delimiter='\t'))
  assert len(rows)==expect[page]['left_cells']+expect[page]['right_cells'],page
  for r in rows:
   assert r['chakma_response'] and r['bengali_prompt']
   out.append(dict(source_unit_key=f"niranjan2010chakma:p{page}:reverse:{r['block']}:row{r['row']}",printed_page=page,pdf_page=page+2,block=r['block'],row=int(r['row']),raw_source_form=r['chakma_response'],raw_bengali_prompt=r['bengali_prompt'],source_note=r['note'],review_status='independent first reading only; full second glyph review required',scope='target Chakma response with Bengali source gloss',accepted_for_installation=False,source_pdf_sha256=S['pdf_sha256'],page_draft_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
  coverage.append(dict(printed_page=page,rows=len(rows),status='full page first reading; independent second review pending'))
 assert len({r['source_unit_key']for r in out})==len(out)
 for page in coverage:
  for block in ('left','right'):
   nums=[r['row'] for r in out if r['printed_page']==page['printed_page'] and r['block']==block]
   assert nums==list(range(1,expect[page['printed_page']][block+'_cells']+1))
 (P/'reverse-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
 (P/'reverse-first-reading-status.json').write_text(json.dumps(dict(stage=('complete chapter first reading; independent second review and integration pending' if len(coverage)==13 else 'partial working draft; not source-stage completion'),rows=len(out),pages=coverage,remaining_pages=[p for p in expect if p not in {r['printed_page']for r in coverage}],expected_physical_cells_provisional=S['physical_cell_count_provisional'],canonical_changed=False),indent=2)+'\n')
 print(len(out))
if __name__=='__main__':main()
