"""Resolve TOC domain headings by exact page and text position, not whole-page scope."""
import argparse,bisect,collections,json,re
from pathlib import Path
from prepare_candidates import HEADER
ROOT=Path(__file__).resolve().parent

def map_domains(path):
    pages=[json.loads(x) for x in path.read_text().splitlines()]
    norm=lambda s:''.join(c.lower() for c in s if c.isalnum())
    sections=[]
    for p in pages[3:5]:
        for line in p['text'].splitlines():
            m=re.match(r'(\d+)\.\s*(.+?)[.…]+\s*(\d+)$',line)
            if not m or int(m[1])<3:continue
            page=int(m[3]);text=pages[page-1]['text'];offset=0;hits=[]
            for raw in text.splitlines(keepends=True):
                if norm(raw)==norm(m[2]):hits.append((offset,raw.strip()))
                offset+=len(raw)
            assert len(hits)==1,(m[2],hits)
            sections.append({'toc_number':int(m[1]),'printed_page':page,'offset':hits[0][0],'title':hits[0][1]})
    assert len(sections)==50
    positions=[(s['printed_page'],s['offset']) for s in sections];assignments=[]
    for p in pages[9:]:
        for i,m in enumerate(HEADER.finditer(p['text']),1):
            index=bisect.bisect_right(positions,(p['pdf_page'],m.start()))-1
            assert index>=0
            assignments.append({'entry_key':f"cfelmahali2024:p{p['pdf_page']}:entry:{i}",'toc_number':sections[index]['toc_number'],'domain':sections[index]['title']})
    assert len(assignments)==2451
    counts=collections.Counter(r['toc_number'] for r in assignments)
    assert len(counts)==50
    for s in sections:s['entries']=counts[s['toc_number']]
    return {'reported_domains':52,'observed_lexical_domains':50,'explanation':'TOC numbering1–52 includes Acknowledgement and Introduction; lexical domains are3–52. Preserve source claim separately.','sections':sections,'assignments':assignments}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pages',type=Path,required=True);a=p.parse_args()
    result=map_domains(a.pages);(ROOT/'domains.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(f"{len(result['sections'])} lexical domains; {len(result['assignments'])} assignments")
