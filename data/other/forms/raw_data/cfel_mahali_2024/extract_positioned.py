"""Reattach combining glyphs geometrically; preserve original page text separately."""
import argparse,hashlib,json,unicodedata
from pathlib import Path
import pdfplumber
from pdfplumber.utils import extract_text
ROOT=Path(__file__).resolve().parent

def position_marks(chars):
    adjusted=[dict(c) for c in chars];audit=[]
    for i,c in enumerate(chars):
        if len(c['text'])!=1 or not ('\u0300'<=c['text']<='\u036f'):continue
        candidates=[(j,b) for j,b in enumerate(chars) if len(b['text'])==1 and not unicodedata.combining(b['text']) and not b['text'].isspace()
                    and b['fontname']==c['fontname'] and abs(b['top']-c['top'])<=5
                    and b['x0']-.4<=c['x0']<=b['x1']+.1]
        if len(candidates)>1:
            preceding=[(j,b) for j,b in candidates if j==i-1]
            if len(preceding)==1:candidates=preceding
        if len(candidates)!=1:
            audit.append({'index':i,'mark':c['text'],'status':'ambiguous-anchor','candidates':[j for j,b in candidates]});continue
        j,b=candidates[0];out=adjusted[i]
        out['x0']=out['x1']=b['x0']+(b['x1']-b['x0'])*.85
        out['top']=b['top'];out['bottom']=b['bottom'];out['doctop']=b['doctop']
        audit.append({'index':i,'mark':c['text'],'anchor_index':j,'anchor':b['text'],'status':'positioned','original_position':[c['x0'],c['top']]})
    return adjusted,audit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pdf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert hashlib.sha256(a.pdf.read_bytes()).hexdigest()==json.loads((ROOT/'source-manifest.json').read_text())['pdf_sha256']
    temp=a.output.with_suffix('.tmp');n=0;ambiguous=0
    with pdfplumber.open(a.pdf) as d,temp.open('w') as f:
        assert len(d.pages)==387
        for number,page in enumerate(d.pages,1):
            chars,audit=position_marks(page.chars);text=extract_text(chars) or ''
            f.write(json.dumps({'pdf_page':number,'text':text,'status':'positioned scaffold; visual audit pending','nul_count':text.count('\x00'),'position_audit':audit},ensure_ascii=False)+'\n')
            n+=len(audit);ambiguous+=sum(r['status']=='ambiguous-anchor' for r in audit);page.close()
    temp.replace(a.output);print(f'{n} combining glyphs; {ambiguous} ambiguous anchors')
