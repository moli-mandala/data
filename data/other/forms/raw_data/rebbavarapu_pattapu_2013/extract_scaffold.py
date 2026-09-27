"""Extract numbered PDF cells for visual transcription, never installation."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import pdfplumber

SHA256='bc16114164909100ddb2ccfd414fcb5e55aa3c1db1fca39883dd6d770acd03e3'

def extract(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==SHA256
    result=[]
    with pdfplumber.open(path) as pdf:
        assert len(pdf.pages)==9
        for number in (6,7,8):
            page=pdf.pages[number-1]
            footers=[w for w in page.extract_words() if w['text']=='Request']
            assert len(footers)==1 and footers[0]['top']>page.height*.9
            bottom=footers[0]['top']-1
            for column,(x0,x1) in enumerate(((0,310),(310,page.width)),1):
                text=page.crop((x0,0,x1,bottom)).extract_text() or ''
                matches=list(re.finditer(r'(?m)^\s*(\d{1,3})\.\s*(.*)$',text))
                for i,m in enumerate(matches):
                    item=int(m[1]);tail=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)]
                    tail=tail.split('Request for Change')[0].strip()
                    raw=m[2]+('\n'+tail if tail else '')
                    result.append({'item':item,'pdf_page':number,'printed_page':number,'column':column,'raw_text':raw,'status':'unreviewed-incomplete-text-layer','warning':'Rendered phonetic glyphs can be absent from PDF text. This is a locator scaffold, not an accepted transcription.'})
    assert len(result)==210 and {r['item'] for r in result}==set(range(1,211)),[(r['item'],r['pdf_page'],r['column']) for r in result]
    return sorted(result,key=lambda r:r['item'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pdf',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();rows=extract(a.pdf);a.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));print('Numbered source cells:',len(rows))
