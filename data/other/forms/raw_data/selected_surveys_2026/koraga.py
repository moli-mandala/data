"""Parse the page-collated Koraga vocabulary without guessing ancestry or lects."""
import json,re,unicodedata
from pathlib import Path
R=Path(__file__).resolve().parent
HEAD=re.compile(r'(?:^|, )([^,]+?) ([omt](?:, [omt])*)(?=, )')
LECT={'o':'onti','m':'mudu','t':'tappu'}
def records():
    out=[]
    glyphs=json.loads((R/'koraga-glyph-review.json').read_text())
    decisions=json.loads((R/'koraga-glyph-decisions.json').read_text())
    assert not decisions['needs_larger_crop']
    edits={}
    for ident in decisions['barred']:
        g=glyphs[ident];edits.setdefault((g['page'],g['entry']),[]).append(g['char'])
    for page in range(88,119):
        for i,line in enumerate((R/'koraga-reviewed'/f'{page:03}.txt').read_text().splitlines(),1):
            original=line
            chars=list(line)
            for pos in edits.get((page,i),[]):
                assert chars[pos]=='i',(page,i,pos,line)
                chars[pos]='ɨ'
            line=''.join(chars)
            if (page,i)==(95,32):line=line.replace('ki:ri','ki:rɨ')
            if (page,i)==(101,19):line=line.replace('ta:li k','ta:lɨ k')
            if (page,i)==(97,40):line=line.replace('giḍle','giḍḷe')
            if page<=91:line=line.replace('ṅ','ŋ')
            if (page,i)==(91,4):line=line.replace('ulṇgu','uḷngu')
            out.append({'printed_page':page,'pdf_page':page+7,'item':i,'text':unicodedata.normalize('NFC',line),'initial_collation':original,'raw_ocr_page':f'koraga-ocr-lines.json#page={page+7}'})
    return out
def parse(r):
    raw=r['text'];parts=re.split(r': (?![omt](?:,|$))',raw,maxsplit=1)
    body=parts[0];comparison=parts[1] if len(parts)>1 else ''
    body=body.rstrip('.');matches=list(HEAD.finditer(body));out=[]
    # Published heads without a dialect siglum are retained under the canonical
    # base only. They do not inherit a neighbour's variety by alphabetic proximity.
    special={
        (93,36):[('kaḍdayi','kind of a drum')],
        (95,32):[('ki:rɨ','to strike matches')],
        (106,28):[('poḍavu','to explode')],
    }
    if (r['printed_page'],r['item']) in special:
        for f,g in special[r['printed_page'],r['item']]:out.append({'form':f,'gloss':g,'lect':'','tags':[],'notes':[],'comparison':comparison,'review':['dialect mapping: no dialect siglum printed']})
        return out
    assert matches,(r,body)
    prefix=body[:matches[0].start()].strip(', ')
    # Three printed coordinated heads share the following variety label.
    assert not prefix or prefix in {'be:','mansti','so:tu'},(r,'unparsed prefix',prefix)
    heads=[]
    for j,mt in enumerate(matches):
        end=matches[j+1].start() if j+1<len(matches) else len(body)
        gloss=body[mt.end():end].strip(', ')
        forms=[mt.group(1).strip()]
        if j==0 and prefix:forms.insert(0,prefix)
        heads.append({'forms':forms,'lects':mt.group(2).split(', '),'gloss':gloss})
    # Empty glosses take the next explicitly printed definition within this entry.
    following=''
    for h in reversed(heads):
        if h['gloss']:following=h['gloss']
        else:h['gloss']=following
    previous=''
    for h in heads:
        g=h['gloss']
        if g in {'id.','id'}:g=previous
        assert g,(r,'missing gloss')
        previous=g;tags=[];notes=[]
        for label,tag in [('pl.','pl'),('sg.','sg'),('F','f'),('M','m')]:
            if '('+label+')' in g:g=g.replace(' ('+label+')','');tags.append(tag)
        def see(mt):
            notes.append('Source cross-reference: '+mt.group(1));return ''
        g=re.sub(r'\s*\(see ([^)]+)\)',see,g).strip()
        if '(used with the negative suffix only)' in g:
            g=g.replace(' (used with the negative suffix only)','');notes.append('Used with the negative suffix only.')
        # The source prints the undefined siglum k on this one secondary head.
        unclear=[]
        if r['printed_page']==101 and r['item']==19:
            assert g=='ta:lɨ k, bolt',r
            g='bolt';unclear=[{'form':'ta:lɨ','gloss':'bolt','lect':'','tags':[],'notes':[],'comparison':comparison,'review':['dialect mapping: source prints undefined siglum k']}]
        for f in h['forms']:
            for lect in h['lects']:
                out.append({'form':f,'gloss':g,'lect':LECT[lect],'tags':list(tags),'notes':list(notes),'comparison':comparison,'review':['transcription: scan-collated lexical reading; source inconsistencies preserved','etymology: source comparisons retained as prose; no ancestry or borrowing inferred']})
        out.extend(unclear)
    return out
if __name__=='__main__':
    out=records()
    for r in out:
        assert parse(r), r
    (R/'koraga-cells.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('entries',len(out),'forms',sum(len(parse(r)) for r in out))
