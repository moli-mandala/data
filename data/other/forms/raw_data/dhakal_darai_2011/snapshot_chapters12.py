"""Bounded native-span census of complete chapters1–2; no lexical installation."""
import hashlib
import json
import unicodedata
from pathlib import Path
import fitz
HERE=Path(__file__).resolve().parent
PDF=HERE.parents[5]/'tmp/pdfs/darai-survey/dhakal-2011.pdf'
KNOWN={'\uf0ab':'ə','\uf048':'ʰ','\uf067':'ɡ','\uf029':'\u0303','\uf04e':'ŋ'}
def decode(text,font):
    if 'SILDoulosIPA93' in font:
        text=''.join(KNOWN.get(c,c) for c in text)
    return unicodedata.normalize('NFC',''.join(f'⟦U+{ord(c):04X}⟧' if 0xe000<=ord(c)<=0xf8ff else c for c in text))
if __name__=='__main__':
    assert hashlib.sha256(PDF.read_bytes()).hexdigest()=='6ca12ee82a22a16b0394b08e319afd5a94f075111fb8a994ee10d59da234bebf'
    out=HERE/'chapters12-native';out.mkdir(exist_ok=True)
    d=fitz.open(PDF)
    for printed in range(1,42):
        page=d[printed+21];records=[]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                records.append({'line':len(records)+1,'bbox':line['bbox'],'spans':[{'raw':s['text'],'decoded':decode(s['text'],s['font']),'font':s['font'],'bbox':s['bbox']} for s in line['spans']]})
        obj={'printed_page':printed,'pdf_page':printed+22,'lines':records}
        (out/f'p{printed:03}.json').write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n')
        (out/f'p{printed:03}.txt').write_text('\n'.join(f"{r['line']:03}: "+''.join(s['decoded'] for s in r['spans']) for r in records)+'\n')
    print('41 pages preserved as native spans and explicit unresolved PUA markers')
