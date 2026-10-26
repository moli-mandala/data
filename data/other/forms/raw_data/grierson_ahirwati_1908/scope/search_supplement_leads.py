"""Discover supplement applicability in retained XML, no transcription or OCR."""
from pathlib import Path
import bz2,json,re,unicodedata,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[6]
stack=[];leads=[]
with bz2.open(ROOT/'tmp/pdfs/lsi-sprakbanken.xml.bz2','rb') as f:
    for event,e in ET.iterparse(f,events=('start','end')):
        if event=='start' and e.tag=='corpus': stack.append(dict(e.attrib))
        elif event=='end' and e.tag=='page':
            if stack[-1].get('vol')=='1':
                ss=[]
                for s in e.findall('sentence'):
                    raw=' '.join(w.text or '' for w in s.findall('w'))
                    folded=''.join(c for c in unicodedata.normalize('NFKD',raw).lower() if not unicodedata.combining(c))
                    if re.search(r'ahir|hirwat|hirwal|irwati|supplement|addenda',folded): ss.append({'id':s.get('id'),'raw':raw})
                if ss: leads.append({'corpus':stack[-1],'page':dict(e.attrib),'sentences':ss})
            e.clear()
        elif event=='end' and e.tag=='corpus': stack.pop()
data=(json.dumps(leads,ensure_ascii=False,indent=2)+'\n').encode();p=BASE/'supplement-xml-discovery.json'
if p.exists(): assert p.read_bytes()==data
else:p.write_bytes(data)
print(json.dumps(leads,ensure_ascii=False,indent=2))
