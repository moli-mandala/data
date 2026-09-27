"""Fresh source/draft sample, excluding retained historical selections."""
import argparse,bisect,collections,hashlib,json,random,re
from pathlib import Path
import csv,pdfplumber
ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[5]
CACHE=WORKSPACE/'tmp/pdfs/cfel-koda-mahali'

def descriptions():
    pages=list(map(json.loads,(CACHE/'mahali-positioned-pages.jsonl').read_text().splitlines()))
    starts=[];parts=[];offset=0
    for page in pages[9:]:
        text='\n'.join(page['text'].splitlines()[:-1])+'\n';starts.append(offset);parts.append(text);offset+=len(text)
    text=''.join(parts);matches=list(re.finditer(r'(?m)^(?P<native>[^\n()]*?)\((?P<grammar>[^()]+)\)\s*-',text));seen=collections.Counter();out={}
    for i,m in enumerate(matches):
        page=bisect.bisect_right(starts,m.start())+9;seen[page]+=1;key=f'cfelmahali2024:p{page}:entry:{seen[page]}'
        block=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)]
        parts=re.split(r'Description\s*:',block,maxsplit=1);out[key]=parts[1].strip() if len(parts)>1 else ''
    assert len(out)==2451
    return out

def sample(seed):
    raw={x['entry_key']:x for x in map(json.loads,(ROOT/'positioned-candidates.jsonl').read_text().splitlines())}
    rows={r[10]:r for r in csv.reader((ROOT/'review-draft.csv').open())};ds=descriptions();excluded=set()
    for f in [*ROOT.glob('output-sample-*.json'),*ROOT.glob('independent-audit-*.json')]:
        if not f.stat().st_size:continue  # stdout redirection may have created this selection
        report=json.loads(f.read_text())
        if report.get('seed')==seed and 'selection' in f.name:
            assert report['csv_sha256']==hashlib.sha256((ROOT/'review-draft.csv').read_bytes()).hexdigest(), 'Proposal changed; choose a fresh audit seed.'
            return report
        for field in ['entries','records','sample']:
            if isinstance(report.get(field),list):excluded.update(r['entry_key'] for r in report[field] if isinstance(r,dict) and 'entry_key' in r)
    keys=random.Random(seed).sample([k for k in raw if k not in excluded],20);entries=[]
    with pdfplumber.open(CACHE/'mahali-multilingual.pdf') as pdf:
        for k in keys:
            r=raw[k];page=pdf.pages[r['pdf_page']-1];hits=page.search(r'\(([^()]+)\)\s*-');hit=hits[r['page_item']-1]
            top=hit['chars'][0]['top']-10;bottom=hits[r['page_item']]['chars'][0]['top']-5 if r['page_item']<len(hits) else page.height-24
            entries.append(dict(r,source_description=ds[k],proposed_row=rows[k],crop_box_points=[90,max(0,top),page.width-35,min(page.height,bottom)],page_size_points=[page.width,page.height]));page.close()
    return {'seed':seed,'sample_size':20,'excluded_keys':sorted(excluded),'csv_sha256':hashlib.sha256((ROOT/'review-draft.csv').read_bytes()).hexdigest(),'audit_sha256':hashlib.sha256((ROOT/'draft-audit.jsonl').read_bytes()).hexdigest(),'entries':entries}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',required=True,type=int);a=p.parse_args();print(json.dumps(sample(a.seed),ensure_ascii=False,indent=2))
