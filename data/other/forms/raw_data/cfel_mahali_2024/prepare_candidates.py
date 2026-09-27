"""Prepare non-emitting source candidates; preserve damaged native text and raw locators."""
import argparse,bisect,collections,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
HEADER=re.compile(r'(?m)^(?P<native>[^\n()]*?)\((?P<grammar>[^()]+)\)\s*-')

def prepare(path):
    pages=[json.loads(x) for x in path.read_text().splitlines()]
    assert len(pages)==387
    parts=[];starts=[];offset=0
    for page in pages[9:]:
        lines=page['text'].splitlines()
        assert lines[-1]==str(page['pdf_page']),page['pdf_page']
        text='\n'.join(lines[:-1])+'\n'
        starts.append(offset);parts.append(text);offset+=len(text)
    text=''.join(parts);matches=list(HEADER.finditer(text))
    assert len(matches)==2451,len(matches)
    seen=collections.Counter();out=[]
    for i,m in enumerate(matches):
        page_index=bisect.bisect_right(starts,m.start())-1;page=page_index+10
        seen[page]+=1
        block=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)]
        lexical=re.split(r'Description\s*:',block,maxsplit=1)[0].strip()
        ipa=re.match(r'([\s\u0300-\u036f]*)/([^/]+)/',lexical)
        issues=[]
        if not ipa:
            assert page==380 and lexical.startswith('ɖũmbɔr/'),(page,lexical[:100])
            form='ɖũmbɔr';issues.append('source-punctuation:opening-IPA-slash-absent; visually verified p380')
        else:
            form=ipa.group(2)
            if ipa.group(1).strip():issues.append('source-punctuation:combining-mark-before-IPA-delimiter:'+ipa.group(1).strip())
        native=m['native'].strip()
        if '\x00' in native:issues.append('native-font:unmapped-glyphs; not accepted Native')
        # Last pair separator identifies English; earlier separators can contain damaged glyphs.
        assert '<>' in lexical,(page,lexical)
        english=lexical.rsplit('<>',1)[1].strip()
        out.append({'entry_key':f'cfelmahali2024:p{page}:entry:{seen[page]}','pdf_page':page,'printed_page':page,'page_item':seen[page],
                    'raw_native':native,'raw_grammar':m['grammar'],'raw_ipa':form,'candidate_english':english,
                    'raw_lexical_block':m.group()+lexical,'issues':issues,'has_description':bool(re.search(r'Description\s*:',block)),
                    'status':'candidate only; segmentation and glyph audit pending'})
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pages',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=prepare(a.pages);a.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps({'candidates':len(rows),'missing_descriptions':sum(not r['has_description'] for r in rows),'damaged_native':sum('\x00' in r['raw_native'] for r in rows)},indent=2))
