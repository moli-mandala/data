"""Freeze native spans for the complete noun/pronoun chapter, without installation."""
import hashlib,importlib.util,json
from pathlib import Path
import fitz
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('chapter_native',P/'snapshot_chapters12.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if __name__=='__main__':
 assert hashlib.sha256(module.PDF.read_bytes()).hexdigest()=='6ca12ee82a22a16b0394b08e319afd5a94f075111fb8a994ee10d59da234bebf'
 out=P/'chapter4-native';out.mkdir(exist_ok=True)
 with fitz.open(module.PDF) as d:
  for n in range(77,125):
   records=[]
   for block in d[n+21].get_text('dict')['blocks']:
    for line in block.get('lines',[]):
     records.append({'line':len(records)+1,'bbox':line['bbox'],'spans':[{'raw':s['text'],'decoded':module.decode(s['text'],s['font']),'font':s['font'],'bbox':s['bbox']} for s in line['spans']]})
   (out/f'p{n:03}.json').write_text(json.dumps({'printed_page':n,'pdf_page':n+22,'lines':records},ensure_ascii=False,separators=(',',':'))+'\n')
   (out/f'p{n:03}.txt').write_text('\n'.join(f"{r['line']:03}: "+''.join(s['decoded'] for s in r['spans']) for r in records)+'\n')
 print('48 complete chapter pages captured; unknown PUA explicit; no canonical writes')
