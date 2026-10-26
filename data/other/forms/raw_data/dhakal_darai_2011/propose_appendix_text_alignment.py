"""Unreviewed alignment candidates anchored to printed surface-word columns.

A lexical candidate retains the entire source column, including every bound piece.
No morphology is inferred and no candidate is approved by geometry alone.
"""
import json,re,unicodedata,collections
from pathlib import Path
import fitz
from snapshot_chapters12 import decode
P=Path(__file__).resolve().parent
# Original p400 gives aṭə-m. once, already segmented, directly above its gloss.
# Preserve that single printed witness in both roles, not an invented repetition.
SURFACE_AS_ANALYSIS={(400,52)}

def build():
    expressions=[json.loads(s) for s in (P/'appendix5-text-first-reading.jsonl').open()]
    bypage=collections.defaultdict(list)
    for expr in expressions:
        for n in sorted({l['printed_page'] for l in expr['surface_fragments']}):bypage[n].append(expr)
    pdf=fitz.open(P.parents[5]/'tmp/pdfs/darai-survey/dhakal-2011.pdf')
    out=[]
    for n,exprs in sorted(bypage.items()):
        rawlines=[l for b in pdf[n+21].get_text('rawdict')['blocks'] for l in b.get('lines',[])]
        def words(line):
            raw=rawlines[line-1];groups=[];current=[]
            for sp in raw['spans']:
                for ch in sp['chars']:
                    if ch['c'].isspace():
                        if current:groups.append(current);current=[]
                    else:current.append((ch,sp['font']))
            if current:groups.append(current)
            return [dict(text=unicodedata.normalize('NFC',''.join(decode(c['c'],font) for c,font in g)),
                         bbox=[min(c['bbox'][0] for c,font in g),min(c['bbox'][1] for c,font in g),
                               max(c['bbox'][2] for c,font in g),max(c['bbox'][3] for c,font in g)],line=line,
                         token=i+1) for i,g in enumerate(groups)]
        for expr in exprs:
            surfaces=[l for l in expr['surface_fragments'] if l['printed_page']==n]
            levels=[]
            for l in surfaces:
                y=l['bbox'][1]
                if not levels or abs(y-levels[-1][0])>2:levels.append((y,[l]))
                else:levels[-1][1].append(l)
            for level,(y,ls) in enumerate(levels):
                bottom=levels[level+1][0] if level+1<len(levels) else max(l['bbox'][3] for l in expr['source_lines'] if l['printed_page']==n)+1
                sw=sorted([w for l in ls for w in words(l['line'])],key=lambda w:w['bbox'][0])
                analyzed=[w for l in expr['interlinear_fragments'] if l['printed_page']==n and y+2<l['bbox'][1]<bottom for w in words(l['line'])]
                glosses=[w for l in expr['gloss_fragments'] if l['printed_page']==n and y+2<l['bbox'][1]<bottom for w in words(l['line'])]
                def assign(pieces):
                    groups=[[] for _ in sw];previous=None;hyphen_continuation=False
                    for w in sorted(pieces,key=lambda w:(round(w['bbox'][1],1),w['bbox'][0])):
                        owner=max([i for i,anchor in enumerate(sw) if anchor['bbox'][0]-3<=w['bbox'][0]] or [0])
                        if previous is not None and (w['text'].startswith('-') or hyphen_continuation):owner=previous
                        groups[owner].append(w);previous=owner;hyphen_continuation=w['text']=='-'
                    return groups
                analyzed_groups=assign(analyzed);gloss_groups=assign(glosses)
                for idx,surface in enumerate(sw):
                    left=surface['bbox'][0]-3
                    right=sw[idx+1]['bbox'][0]-3 if idx+1<len(sw) else 10000
                    aw=analyzed_groups[idx]
                    gw=gloss_groups[idx]
                    if (n,surface['line']) in SURFACE_AS_ANALYSIS:
                        assert not aw
                        aw=[dict(surface,text=surface['text'].rstrip('.'))]
                    out.append(dict(source_unit_key=expr['source_unit_key']+f":word:p{n}:l{surface['line']}:w{surface['token']}",
                                    parent_expression_key=expr['source_unit_key'],text_id=expr['text_id'],printed_page=n,pdf_page=n+22,
                                    source_surface_form=surface['text'],form=''.join(w['text'] for w in aw),
                                    candidate_gloss=' '.join(w['text'] for w in gw),surface_evidence=surface,
                                    analysis_evidence=aw,gloss_evidence=gw,
                                    status='Unreviewed source-column candidate; not approved for export',
                                    needs_manual_alignment=not aw or not gw or len([w for w in aw if not w['text'].startswith('-')])>1))
    return out

if __name__=='__main__':
    rows=build();(P/'appendix5-interlinear-column-candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(len(rows),'unreviewed word columns;',sum(r['needs_manual_alignment'] for r in rows),'explicit alignment warnings')
