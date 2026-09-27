"""Cache PDF page text with immutable page locators; no lexical emission or DB build."""
import argparse,hashlib,json
from pathlib import Path
import pdfplumber
ROOT=Path(__file__).resolve().parent

def extract(pdf,output):
    digest=hashlib.sha256(pdf.read_bytes()).hexdigest()
    expected=json.loads((ROOT/'source-manifest.json').read_text())
    assert digest==expected['pdf_sha256'], 'source edition changed'
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix(output.suffix+'.tmp')
    with pdfplumber.open(pdf) as document,temp.open('w') as f:
        assert len(document.pages)==387
        for number,page in enumerate(document.pages,1):
            text=page.extract_text() or ''
            row={'pdf_page':number,'text':text,'status':'raw text scaffold; not accepted lexical transcription','nul_count':text.count('\x00')}
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
            page.close()
    temp.replace(output)
    return digest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pdf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(extract(a.pdf,a.output))
