"""Exact physical-cell recovery of the nonpast appendix, including page continuation."""
import json,re,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent

def build():
    pages={}
    for s in (P/'whole-native-pages.jsonl').open():
        r=json.loads(s)
        if r.get('printed_page') in (368,369):pages[r['printed_page']]=r
    def txt(p,l):return unicodedata.normalize('NFC',''.join(s['decoded'] for s in pages[p]['lines'][l-1]['spans']).strip())
    # Explicit source line boundaries; no generation of unattested conjugations.
    specs=[
      (368,[31],'laha','bathe',31,[32,34,36,38,40,42,43,45,47,49]),
      (368,[50],'nida','sleep',51,[52,54,56,58,60,62,63,65,67,69]),
      (368,[70],'uḍʰe','fly',71,[72,74,76,78,80,82,83,85,87,89]),
      (368,[90,91],'bohə/buhə','float',92,[93,95,97,99,101,103,104,106,108,110]),
      (368,[112,113],'rəhə','remain',114,[115,117,119,121,123,125,126,128,130,132]),
      (368,[133],'kʰor','bathe',134,[135,137,139,141,143,145,146,148,150,152]),
      (368,[154],'pucʰ','ask',155,[156,158,160,162,164,166,167,169,171,173]),
      (368,[174],'a','come',175,[176,178,179,180,182,184,185,187,189,191]),
      (368,[193],'dza','go',194,[195,197,199,201,203,205,206,208,210,212]),
      (368,[213],'cʰi','touch',214,[215,217,219,221,223,225,226,228,230,232]),
      (368,[233],'pi','drink',234,[235,237,238,240,242,244,245,247,249,251]),
      (368,[252],'dʰo','wash',253,[254,256,258,260,262,264,265,267,269,271]),
      (368,[273],'ro','wash',274,[275,277,279,281,283,285,286,288,290,292]),
      (368,[293,294],'kʰəs','fall down',295,[296,298,300,302,304,306,307,309,311,313]),
      (369,[18,19],'ɡokʰr','bathe',21,[22,24,26,28,30,32,34,36,38,40]),
      (369,[42,43],'ɡəbs','bury in the mud',44,[45,47,49,51,53,55,56,58,60,62])]
    labels=[txt(368,i) for i in range(21,31)]
    out=[]
    def emit(p,key,form,gloss,refs,kind,**extra):
        evidence=[dict(printed_page=n,**pages[n]['lines'][l-1]) for n,l in refs]
        out.append(dict(source_unit_key=f'dhakal2011:p{p}:appendix3b:{key}',printed_page=p,pdf_page=p+22,
                        form=form,gloss=gloss,kind=kind,source_locations=[dict(printed_page=n,line=l) for n,l in refs],
                        evidence=evidence,status='Owner original-page review passed; independent review pending',**extra))
    for p,rootlines,root,gloss,start,ends in specs:
        rootkey=f'l{rootlines[0]}:root'
        notes='Source labels ro as wash here, differing from cry on p. 367; neither reading is corrected.' if root=='ro' else ''
        emit(p,rootkey,root,gloss,[(p,l) for l in rootlines],'paradigm-root',note=notes)
        for col,end in enumerate(ends):
            refs=[(p,l) for l in range(start,end+1)]
            fragments=[txt(p,l) for _,l in refs]
            if p==368 and start==31:fragments[0]=fragments[0].split('’',1)[1].strip()
            form=''.join(fragments).replace(' (-səb)','(-səb)')
            emit(p,f'l{start}:col{col+1}',form,gloss,refs,'paradigm-inflection',source_grammar='non-past '+labels[col],
                 root_source_unit_key=f'dhakal2011:p{p}:appendix3b:{rootkey}',raw_fragments=fragments,note=notes)
            if '--' in form:
                out[-1]['transcription_uncertainty']='Source prints hyphens at both sides of a line break; both retained without normalization.'
            start=end+1
    # Last row on p. 368 continues in the same columns on p. 369.
    rootkey='dhakal2011:p368:appendix3b:l315:root'
    emit(368,'l315:root','kʰasa','make one fall down',[(368,315),(369,3),(369,4)],'paradigm-root')
    tails=[[5],[6],[7],[8],[10],[11],[13],[14],[15],[16,17]]
    for col,ls in enumerate(tails):
        refs=[(368,316+col)]+[(369,l) for l in ls]
        fragments=[txt(n,l) for n,l in refs]
        emit(368,f'l{316+col}:col{col+1}',''.join(fragments),'make one fall down',refs,'paradigm-inflection',
             source_grammar='non-past '+labels[col],root_source_unit_key=rootkey,raw_fragments=fragments,
             note='One physical table row continues across pp. 368–369; cell fragments joined by explicit column alignment.')
    assert len(out)==187
    assert len({r['source_unit_key'] for r in out})==187
    return out

if __name__=='__main__':
    rows=build()
    (P/'appendix3b-reviewed-records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(len(rows),'physical cells: 17 roots and 170 inflections; no installation')
