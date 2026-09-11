#!/usr/bin/env python3
"""Source-preserving extraction of Perder's 2013 Dameli grammar.

Like the Knobloch Sauji ingestion, this uses positioned PDF text, explicit table
regions, aligned interlinear tiers and prose citations. The PDF is not distributed;
the checked-in lexical snapshot makes ordinary rebuilds independent of it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_ID = 'perder2013dameli'
DATE = '2026-09-09'
ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / 'data/other/forms/raw_data'
PREFIX = '20260909-perder-dameli'
SNAPSHOT = RAW / f'{PREFIX}-extract.jsonl'
FORM_OUTPUT = ROOT / f'data/other/forms/{PREFIX}.csv'
PDF_PAGES = 242
PAGE_OFFSET = 24
PDF_SHA256 = '6d740b309f86534157ea8e8c741e2bcc27ac464ac9e9c719e530c3193219fe1c'
PDF_SHA512 = '5315e4cdbb9c1b104720ca9cbe3d336cedc09239a3f50a7ba4df0e25880f7fae540da77234039ad89adae0f2d87d48c4613e526596265eb24cb11ac46aaa6669'
FIELDS = ['Language_ID', 'Parameter_ID', 'Form', 'Gloss', 'Native', 'Phonemic',
          'Notes', 'Source', 'Cognateset', 'Etymology', 'Entry_Key',
          'Variant_Of_Key', 'Borrowed_From_Key', 'Derivation_Parent_Keys', 'Tags']


def nfc(text):
    return unicodedata.normalize('NFC', text)


def tokens(words):
    """Rejoin font/size changes and combine marks using their printed positions."""
    bands = []
    for word in sorted(words, key=lambda w: (w['top'], w['x0'])):
        word = dict(word)
        font = word['fontname'].split('+')[-1]
        y = word['top']
        if 'Gentium' in font:
            y -= .235 * word['size']
        elif word['size'] <= 8.1:
            y -= 1.5
        if word['size'] < 7:  # footnote numbers never belong to a form
            continue
        word['obj'] = 'Gentium' in font or 'Doulos' in font
        word['bold'] = 'Bold' in font
        word['font'] = font
        word['y'] = y
        band = next((b for b in bands if abs(b[0] - y) < 3.0), None)
        if band is None:
            bands.append([y, [word]])
        else:
            band[1].append(word)
    lines = []
    for y, ws in sorted(bands):
        merged = []
        for w in sorted(ws, key=lambda w: w['x0']):
            if merged and w['x0'] < merged[-1]['x1'] - 1 and w['text'] == merged[-1]['text']:
                continue  # faux-bold overprinting
            mark = all(unicodedata.combining(c) for c in w['text'])
            if merged and (mark or w['x0'] - merged[-1]['x1'] < 1.05):
                merged[-1]['text'] += w['text']
                merged[-1]['x1'] = max(w['x1'], merged[-1]['x1'])
                merged[-1]['bold'] |= w['bold']
                merged[-1]['obj'] |= w['obj']
            else:
                merged.append(w)
        for w in merged:
            w['text'] = nfc(w['text'])
        lines.append((round(y, 2), merged))
    return lines


def txt(ws):
    return ' '.join(w['text'] for w in ws).strip()


def cell(ws, left, right):
    return txt([w for w in ws if left-.5 <= w['x0'] < right-.5])


def rec(page, region, unit, form, gloss, context='', tags='', **extra):
    return dict(unit=unit, region=region, pdf_page=page, page=page-PAGE_OFFSET,
                form=nfc(form.strip()), gloss=nfc(gloss.strip()), context=nfc(context),
                tags=tags, **extra)


def interlinear(pages, used):
    result = []
    for p in pages:
        page = p['page']
        if not (73 <= page <= 220 or 234 <= page <= 240):
            continue
        lines = p['lines']
        example = ''
        for i, (y, ws) in enumerate(lines[:-1]):
            if not 45 < y < 615 or in_table(page, y):
                continue
            for w in ws:
                if w['x0'] < 93 and re.fullmatch(r'\(\d+\)', w['text']):
                    example = w['text'][1:-1]
            body = [w for w in ws if w['x0'] >= 60 and not re.fullmatch(r'\(\d+\)', w['text'])]
            if re.search(r'\b(?:PRSPTCP|PFV|IMPFV|CAUS|TOPSH|POSS)\b',txt(body)):
                continue  # a gloss tier printed in object-language/bold type
            if not body or sum(len(w['text']) for w in body if w['obj'] or w['bold']) < .7 * sum(len(w['text']) for w in body):
                continue
            yn, nxt = lines[i+1]
            if not 8 < yn-y < 21 or any(w['text'].startswith(('‘', '“')) for w in nxt[:1]):
                continue
            obj = [w for w in body if w['obj'] or w['bold']]
            matches = [min((abs(w['x0']-g['x0']) for g in nxt), default=999) < 2.8 for w in obj]
            if not obj or sum(matches) < .7 * len(obj):
                continue
            context = '\n'.join(txt(a) for _, a in lines[max(0,i-1):min(len(lines),i+7)])
            for k, w in enumerate(obj):
                right = obj[k+1]['x0'] - 1.4 if k+1 < len(obj) else 420
                gloss = cell(nxt, w['x0']-2.8, right)
                unit = f'p{page-PAGE_OFFSET}:ex{example or "text"}:y{round(y)}:w{k+1}'
                result.append(rec(page,'interlinear',unit,w['text'],gloss,context,
                                  example=example,x=round(w['x0'],2),y=y))
                used.add((page,y,round(w['x0'],2)))
    return result


# Each region is an explicitly inspected numbered table. Continuations are
# separate physical-page regions, including the caption-only start of Table 26.
TABLES = {
    49:[(4,181,414)],50:[(5,51,133)],51:[(6,251,387)],
    55:[(7,525,590)],56:[(7,50,212)],57:[(8,236,358)],58:[(9,212,335)],
    65:[(10,195,347),(11,372,452)],66:[(12,293,388),(13,470,551)],
    74:[(14,203,297)],76:[(15,545,612)],77:[(15,50,130)],
    81:[(16,327,451)],91:[(17,58,490)],92:[(17,50,510)],93:[(17,50,161)],
    96:[(18,459,612)],97:[(18,50,90)],102:[(19,336,589)],
    103:[(20,195,333)],104:[(21,245,314)],108:[(22,51,273)],
    109:[(23,156,252)],110:[(24,51,185),(25,353,449)],
    112:[(26,590,626)],113:[(26,50,336)],119:[(27,51,612)],120:[(27,51,360)],
    122:[(28,51,273)],123:[(29,304,386),(30,506,562)],126:[(31,268,531)],
    128:[(32,212,463)],131:[(33,50,620)],132:[(33,50,380)],
    136:[(34,360,429),(35,524,593)],140:[(36,100,226)],141:[(37,475,616)],
    147:[(38,51,614)],148:[(38,50,188)],149:[(39,407,573)],150:[(39,50,258)],
    159:[(40,72,392)],161:[(41,51,232)],176:[(42,51,130)],
    186:[(43,142,280)],187:[(44,295,391)],189:[(45,51,135),(46,204,347)],
    203:[(47,51,232)],231:[(48,51,614)],232:[(48,51,611)],
}


def in_table(page,y):
    return any(lo <= y <= hi for _,lo,hi in TABLES.get(page, []))


def prose(pages, used):
    result = []
    for p in pages:
        page = p['page']
        if not 27 <= page <= 223:
            continue
        lines = p['lines']
        for li,(y,ws) in enumerate(lines):
            if not 45 < y < 627 or in_table(page,y):
                continue
            i=0
            while i<len(ws):
                if not ws[i]['obj'] or (page,y,round(ws[i]['x0'],2)) in used:
                    i+=1; continue
                end=i+1
                while end<len(ws) and ws[end]['obj']:
                    end+=1
                form=txt(ws[i:end]); tail=txt(ws[end:])
                # A quoted gloss can wrap onto the following prose line.
                if (not tail or tail.count('‘')>tail.count('’')) and li+1<len(lines):
                    tail+=' '+txt(lines[li+1][1])
                match=re.match(r'^\s*[,;]?\s*(?:\([itc]\)\s*)?[‘\'](.+?)[’\'](?![a-z])(.*)',tail)
                gloss=match.group(1) if match else ''
                after=match.group(2) if match else ''
                # Keep source grammatical labels immediately following a definition.
                tagpart=re.match(r'^\s*(\([^()]*\))',after)
                if tagpart and re.search(r'[A-Z]{2}|^[ ]*\([mfi]\)',tagpart.group(1)):
                    gloss+=' '+tagpart.group(1)
                context='\n'.join(txt(a) for _,a in lines[max(0,li-1):li+2])
                unit=f'p{page-PAGE_OFFSET}:prose:y{round(y)}:x{round(ws[i]["x0"])}'
                result.append(rec(page,'prose',unit,form,gloss,context,
                                  x=round(ws[i]['x0'],2),y=y))
                i=end
    return result


def tables(pages):
    bypage={p['page']:p['lines'] for p in pages}
    out=[]
    def lines(page,lo,hi):return [(y,w) for y,w in bypage[page] if lo<=y<=hi]
    def emit(page,t,y,col,form,gloss,tags='',context='',**extra):
        unit=f'p{page-PAGE_OFFSET}:t{t}:y{round(y)}:c{col}'
        r=rec(page,f'table{t}',unit,form,gloss,context,tags,table=t,y=y,column=col,**extra)
        out.append(r);return unit
    def quoted(value):
        m=re.match(r"(.*?)\s*[‘'](.*?)[’'](?![a-z])(.*)$",value.strip())
        if not m:
            if '‘' in value:
                f,g=value.split('‘',1);return f.strip(),g.strip()
            return value.strip(),''
        return m[1].strip(),(m[2]+m[3]).strip().rstrip('.,')
    def block(page,lo,hi,left,right):
        return ' '.join(cell(w,left,right) for y,w in lines(page,lo,hi) if cell(w,left,right))
    def anchored(page,t,lo,hi,formcols,glosscols=None,tags='',notescols=None):
        # Each column is independent: a wrapped right-hand glossary entry must
        # not create a blank record in the independently aligned left column.
        for ci,(left,right) in enumerate(formcols,1):
            selected=lines(page,lo,hi)
            starts=[y for y,w in selected if any(a['obj'] and left<=a['x0']<right for a in w)]
            for j,y in enumerate(starts):
                stop=starts[j+1]-.1 if j+1<len(starts) else hi
                value=block(page,y,stop,left,right)
                if glosscols:
                    form=value;glo=block(page,y,stop,*glosscols[ci-1])
                else:form,glo=quoted(value)
                note=block(page,y,stop,*notescols) if notescols else ''
                emit(page,t,y,ci,form,glo,tags,block(page,y,stop,0,420),notes=note)
    # Phonology examples are complete lexical words, not phoneme inventory cells.
    for page,t,lo,hi,col in [(49,4,210,414,(210,410)),(51,6,282,388,(202,410)),
            (55,7,554,590,(166,410)),(56,7,51,212,(166,410)),
            (57,8,266,358,(233,410))]:
        anchored(page,t,lo,hi,[col])
    anchored(50,5,82,133,[(181,281),(281,410)])
    anchored(58,9,243,335,[(108,163)],[(163,261)],notescols=(262,410))
    # Sandhi tables: root and attested suffixed phrase are retained, affix and
    # donor control cells stay in the raw context, not the Dameli inventory.
    for t,ys,hi in [(10,[228.5,242.5,256.6,284.5,312.5,340.6],347),
                    (11,[405.5,419.5,433.6,447.5],452)]:
        for j,y in enumerate(ys):
            stop=ys[j+1]-1 if j+1<len(ys) else hi
            form,glo=quoted(block(65,y-1,y+1,68,178))
            emit(65,t,y,1,form,glo,context=block(65,y-1,stop,0,420))
            f,g=quoted(block(65,y-1,stop,227,420))
            emit(65,t,y,3,f,g,context=block(65,y-1,stop,0,420))
    for y,w in lines(66,324,388):
        if not any(a['obj'] for a in w):continue
        for ci,(l,r) in enumerate([(87,176),(176,273),(273,420)],1):
            f,g=quoted(cell(w,l,r))
            if ci==1:g=quoted(cell(w,176,273))[1]
            emit(66,12,y,ci,f,g,context=txt(w))
    for y,w in lines(66,502,551):
        f,g=quoted(cell(w,135,240))
        if f:emit(66,13,y,1,f,g,'loanword',txt(w),etymology='Borrowed from Urdu '+cell(w,240,420)+'.')
    for y,w in lines(74,233,297):
        gloss=cell(w,324,420)
        if not gloss:continue
        parents=[]
        for ci,(l,r) in enumerate([(138,227),(228,323)],2):
            f,g=quoted(cell(w,l,r));parents.append(emit(74,14,y,ci,f,g,context=txt(w)))
        emit(74,14,y,1,cell(w,68,137),gloss,'compound',txt(w),parents=parents,
             etymology='Compound of the two components explicitly identified in Table 14.')
    for page,lo,hi in [(76,576,612),(77,52,130)]:
        for y,w in lines(page,lo,hi):
            f=cell(w,76,154)
            if f:emit(page,15,y,1,f,cell(w,154,254),'noun '+cell(w,254,287).lower(),txt(w),
                      notes=cell(w,287,420))
    for y,w in lines(81,358,451):
        f,g=quoted(cell(w,76,174))
        if not f:continue
        emit(81,16,y,1,f,g,'noun loanword',txt(w),etymology='Borrowed via Pashto: '+cell(w,174,301)+'.')
        emit(81,16,y,3,cell(w,301,420),g,'noun loanword pl',txt(w))
    for page,lo,hi in [(91,116,490),(92,52,130),(92,170,510),(93,67,161)]:
        anchored(page,17,lo,hi,[(88,164)],[(164,264)],'noun kinship',(264,420))
    # Pronoun cells spanning case columns deliberately retain all applicable cases.
    for page,lo,hi in [(96,491,612),(97,53,90)]:
        for y,w in lines(page,lo,hi):
            label=cell(w,114,215)
            for ci,a in enumerate([a for a in w if a['obj'] and a['x0']>=215],1):
                if 215<=a['x0']<245:case='NOM'
                elif 265<=a['x0']<275:case='NOM.OBL.ERG'
                elif 275<=a['x0']<289:case='OBL'
                elif 289<=a['x0']<315:case='OBL.ERG'
                else:case='ERG'
                emit(page,18,y,ci,a['text'],label+'.'+case,'pron personal',txt(w))
    # Table 19 has irregular wrapped/merged cells; transcribed cell for cell.
    t19=[
      ('mas','3SG.OBL.PROX','pron'),('tas','3SG.OBL.DIST','pron'),('kii/kurey','who?','pron interr'),
      ('keeraa','which?','pron interr'),('kya','what','noun interr'),
      ('masãã','3SG.PROX.POSS','pron poss'),('tasãã','3SG.DIST.POSS','pron poss'),('kasãã','whose?','pron poss interr'),
      ('manuu','thus, in this way; this kind','adv adj manner prox'),('tanuu','in that way; that kind','adv adj manner dist'),('kanuu','how?','adv adj manner interr'),
      ('manee','thus, in this way','adv manner prox'),('tanee','in that way','adv manner dist'),('kutaal','where to?','adv manner interr'),
      ('ayaa','here','adv spatial prox'),('tara','there','adv spatial dist'),('kaa','where?','adv spatial interr'),
      ('matiki','so many','adj quantifier prox'),('tatiki','that many','adj quantifier dist'),('kati','how many','adj quantifier interr'),('katiki','so many','adj quantifier interr')]
    for i,(f,g,tags) in enumerate(t19,1):
        emit(102,19,0,i,f,g,tags,block(102,336,590,0,420),extraction='manually delimited irregular table cell')
    for y,w in lines(103,227,333):
        label=cell(w,103,182)
        for ci,(l,r,gender) in enumerate([(182,281,'m'),(281,420,'f')],1):
            f=cell(w,l,r)
            if f:emit(103,20,y,ci,f,label+'.POSS','pron poss '+gender,txt(w))
    for y,w in lines(104,291,314):
        label=cell(w,126,216)
        for ci,(l,r,g) in enumerate([(216,280,'m'),(280,420,'f')],1):
            f=cell(w,l,r)
            if f:emit(104,21,y,ci,f,'possessive marker','poss '+g+' '+label.lower(),txt(w))
    for y,w in lines(108,83,273):
        for ci,(l,r,g) in enumerate([(128,198,'m'),(198,277,'f')],1):
            f=cell(w,l,r)
            if f:emit(108,22,y,ci,f,cell(w,277,420),'adj '+g,txt(w))
    anchored(109,23,187,252,[(160,233)],[(233,420)],'adj')
    # Derivational tables: links refer to the precise same-row base, never a
    # surface-form search elsewhere in the corpus.
    for y0,y1 in [(85,98),(99,112),(113,126),(128,140),(142,155),(156,166),(169,185)]:
        f,g=quoted(block(110,y0-.5,y1,68,234));base,bg=quoted(block(110,y0-.5,y1,234,420))
        parent=emit(110,24,y0,2,base,bg,context=block(110,y0-.5,y1,0,420))
        emit(110,24,y0,1,f,g,'adj derived',block(110,y0-.5,y1,0,420),parents=[parent])
    for y,w in lines(110,384,449):
        if not cell(w,84,145):continue
        parent=emit(110,25,y,2,cell(w,259,308),cell(w,308,420),'verb stem',txt(w))
        emit(110,25,y,1,cell(w,83,145),cell(w,145,259),'adj derived',txt(w),parents=[parent])
    for y,w in lines(113,51,336):
        for ci,(nl,nr,fl,fr) in enumerate([(80,110,158,220),(223,250,321,420)],1):
            number=cell(w,nl,nr);f=cell(w,fl,fr)
            if number.isdigit() and f:emit(113,26,y,ci,f,number,'num',txt(w))
    # Verb overview: gender-spanning cells are one syncretic form, not empty F rows.
    meanings=['play','count','bring down','make someone read/study']
    valency=['intr','tr','caus','caus second-causative']
    for page,lo,hi,tam in [(119,173,264,'ipfv'),(119,292,383,'pfv'),
         (119,411,502,'indirect-past pret'),(119,530,607,'potential-past pret'),
         (120,113,190,'fut'),(120,218,240,'impv')]:
        for y,w in lines(page,lo,hi):
            label=cell(w,65,106)
            # Three masculine/feminine cells are centred vertically between labels.
            if not label:label='3SG'
            for ci,(l,r) in enumerate([(106,177),(177,255),(255,326),(326,420)],1):
                f=cell(w,l,r)
                if f:emit(page,27,y,ci,f,meanings[ci-1],f'verb {valency[ci-1]} {tam} '+label,txt(w))
    # Roots in the header have their own meanings (not causative-column meanings).
    for i,(f,g,v) in enumerate([('muṣ','play','intr'),('leekʰ','count','tr'),('naɡ','come down','intr'),('matr','read','tr')],1):
        emit(119,27,147,i,f,g,'verb stem '+v,block(119,139,161,0,420))
    nonfinite=['inf','pres participle','pret participle','inchoative-participle','conjunctive-participle']
    for j,(y,w) in enumerate(lines(120,291,360)):
        for ci,(l,r) in enumerate([(170,227),(227,284),(284,340),(340,420)],1):
            f=cell(w,l,r)
            if f:emit(120,27,y,ci,f,meanings[ci-1],f'verb {valency[ci-1]} {nonfinite[j]}',txt(w))
    for y,w in lines(122,83,273):
        for ci,(l,r,tam) in enumerate([(68,116,'ipfv'),(116,201,'pfv')],1):
            emit(122,28,y,ci,cell(w,l,r),cell(w,201,420),'verb stem '+tam,txt(w))
    for page,t,lo,hi,cols in [(123,29,350,386,(68,137,214,291)),
          (123,30,539,562,(97,145,252,294)),(126,31,313,531,(68,139,231,309)),
          (128,32,271,463,(68,130,208,272))]:
        l,g,dl,dg=cols
        starts=[y for y,w in lines(page,lo,hi) if any(a['obj'] and l<=a['x0']<g for a in w)]
        for j,y in enumerate(starts):
            stop=starts[j+1]-.1 if j+1<len(starts) else hi
            base=block(page,y,stop,l,g);bg=block(page,y,stop,g,dl)
            f=block(page,y,stop,dl,dg);gloss=block(page,y,stop,dg,420)
            parent=emit(page,t,y,1,base,bg,'verb stem',block(page,y,stop,0,420))
            emit(page,t,y,3,f,gloss,'verb stem '+('tr' if t==30 else 'caus')+(' second-causative' if t==32 else ''),
                 block(page,y,stop,0,420),**({'parents':[parent]} if t!=30 else {'etymology':'Perder suggests this pair may be a vestige of an earlier causative vowel-strengthening process; no certain derivation asserted.'}))
    for t,lo,hi,fc,gc,nc in [(34,392,429,(129,175),(175,269),(269,420)),(35,557,593,(94,149),(149,262),(262,420))]:
        anchored(136,t,lo,hi,[fc],[gc],'verb stem',nc)
    for j,(y,w) in enumerate(lines(140,131,226)):
        for ci,(l,r) in enumerate([(207,297),(297,420)],1):
            f,g=quoted(cell(w,l,r));emit(140,36,y,ci,f,g,'verb '+('stem' if j==0 else nonfinite[j-1]),txt(w))
    for page,lo,hi,tam in [(147,84,90,'inf'),(147,112,203,'ipfv'),(147,239,331,'pfv'),
        (147,366,459,'indirect-past pret'),(147,494,571,'potential-past pret'),
        (147,606,614,'fut'),(148,53,117,'fut'),(148,151,188,'participle')]:
        for y,w in lines(page,lo,hi):
            label=cell(w,142,210)
            for ci,(l,r) in enumerate([(210,272),(272,420)],1):
                f=cell(w,l,r)
                if f:emit(page,38,y,ci,f,'be','verb copula animate '+tam+' '+('' if label=='Infinitive' else label),txt(w))
    # Composite lexical constructions, plus explicitly glossed components.
    for page,t,lo,hi,c1,c2,gc in [(149,39,438,573,(68,171),(171,255),(255,420)),
                              (150,39,52,258,(68,171),(171,255),(255,420)),
                              (187,44,326,391,(68,182),(182,295),(295,420))]:
        previous=None
        for y,w in lines(page,lo,hi):
            v1=cell(w,*c1);v2=cell(w,*c2)
            if not v2:continue
            parents=[]
            if v1:
                f1,g1=quoted(v1);parent=emit(page,t,y,1,f1,g1,context=txt(w),allow_blank=True);previous=(f1,parent)
            else:f1,parent=previous
            parents.append(parent)
            f2,g2=quoted(v2)
            if 'become’' in f2:f2,g2=f2.replace('become’','').strip(),'become'
            parents.append(emit(page,t,y,2,f2,g2,context=txt(w)))
            emit(page,t,y,3,f1+' '+re.sub(r'\s*\([itc]\)$','',f2),cell(w,*gc),
                 'verb conjunct-verb' if t==39 else 'adj multiword-expression',txt(w),parents=parents)
    for y,w in lines(159,103,392):
        typ=cell(w,231,311);tag={'time':'temporal','space':'spatial','manner':'manner','intensifier':'degree','modality':'modal'}.get(typ,'')
        emit(159,40,y,1,cell(w,80,127),cell(w,127,231),'adv '+tag,txt(w),notes=cell(w,311,420))
    for page,t in [(161,41),(203,47)]:
        for y,w in lines(page,83,232):
            f=cell(w,90,195);g=cell(w,195,280);label=cell(w,280,420)
            if not f:continue
            extra={}
            if page==203 and g=='kya':
                g='what';label='noun';extra={'repair':'Table 47 repeats kya in all three columns; restored gloss and category from the identical row in Table 41 (p. 137).'}
            tags={'pronoun':'pron','pronoun (possessive)':'pron poss','adverb':'adv','adjective':'adj','noun':'noun'}[label]
            emit(page,t,y,1,f,g,tags+' interr',txt(w),**extra)
    for y,w in lines(186,159,280):
        f=cell(w,68,258);gloss=cell(w,258,420)
        if 'also' in f:
            emit(186,43,y,2,'ɡurma ki','in the morning; tomorrow','adv temporal',txt(w))
            f='beraa ki'
        emit(186,43,y,1,f,gloss,'adv temporal',txt(w))
    for t,lo,hi in [(45,84,135),(46,237,347)]:
        anchored(189,t,lo,hi,[(68,238)],[(238,420)],'multiword-expression')
    for page,lo in [(231,88),(232,71)]:
        anchored(page,48,lo,612,[(66,132),(250,317)],[(132,249),(317,420)],'verb stem')
    return out


def pdf_words(page):
    """Restore combining-mark order from the PDF content stream before x-sorting.

    The dot below c is often drawn to the right of the following i/h glyph's
    left edge. PDF content order is correct; positional sorting alone is not.
    Moving its zero-width box to the base's right edge fixes extraction only.
    """
    page = page.dedupe_chars()
    previous = None
    seen = []
    for char in page.chars:
        if (char['text'] and all(unicodedata.combining(c) for c in char['text'])
                and previous and abs(char['top']-previous['top']) < 5):
            base=previous
            if char['text']=='\u0323':
                eligible=[c for c in seen[-12:] if c['text']=='c'
                          and abs(c['top']-char['top'])<3
                          and 0<=char['x0']-c['x0']<12]
                if eligible:base=eligible[-1]
            char['x0'] = char['x1'] = base['x1'] - .05
        elif not char['text'].isspace():
            previous = char
        seen.append(char)
    return page.extract_words(x_tolerance=1, y_tolerance=3,
                              extra_attrs=['fontname','size'])


def extract(pdf_path):
    import pdfplumber
    raw=pdf_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PDF_SHA256 or hashlib.sha512(raw).hexdigest()!=PDF_SHA512:
        raise ValueError('PDF differs from the pinned DiVA source')
    with pdfplumber.open(pdf_path) as pdf:
        assert len(pdf.pages)==PDF_PAGES
        pages=[{'page':i+1,'lines':tokens(pdf_words(p))}
            for i,p in enumerate(pdf.pages)]
    return extract_pages(pages)


def extract_pages(pages):
    used=set()
    return [*tables(pages), *interlinear(pages,used), *prose(pages,used)]


# Source abbreviations (pp. vii-viii), including spelled variants in the examples.
GRAMMAR = {
 'ANIM':'animate','INANIM':'inanimate','APPR':'appropriate-place','APP':'appropriate-place',
 'CAUS':'caus','CAUS2':'caus second-causative','COLL':'collective','COP':'copula verb',
 'CP':'conjunctive-participle verb','ECHO':'echo','EP':'epenthetic','ERG':'erg',
 'F':'f','FILL':'filler','FUT':'fut verb','IMP':'impv verb','IMPFV':'ipfv verb',
 'INCHPTCP':'inchoative-participle participle verb','INDIRPST':'indirect-past pret verb',
 'INF':'inf verb','INS':'instr','INSTR':'instr','KIN':'kinship','KIN3':'kinship third-person',
 'LOC':'loc','M':'m','NEG':'neg','NOM':'nom','OBL':'obl','ORD':'ord num',
 'PFV':'pfv verb','PL':'pl','POSS':'poss','POTPST':'potential-past pret verb',
 'PROH':'prohibitive neg','PROX':'prox','DIST':'dist','PRSPTCP':'pres participle verb',
 'PRS':'pres verb','PST':'pret','PAST':'pret','DIRPST':'pret','PSTPTCP':'pret participle verb',
 'Q':'interr','QUOT':'quotative','REFL':'refl','SG':'sg','TOPSH':'topic-shift part',
 'TOPSM':'topic-same part','VOC':'voc','PART':'part','ACC':'acc',
 **{f'{p}{n}':f'{p}{n.lower()} {n.lower()}' for p in '123' for n in ('SG','PL')},
 '1':'first-person','2':'second-person','3':'third-person',
}
GRAMMAR_RE=re.compile(r'(?<![A-Za-z0-9])('+ '|'.join(sorted(GRAMMAR,key=len,reverse=True))+r')(?![A-Za-z0-9])')
LEXICAL_MARKERS={'COP':'be','TOPSH':'shift-topic marker','TOPSM':'same-topic marker',
 'POSS':'possessive marker','REFL':'self','NEG':'not','PROH':'do not','Q':'question marker',
 'QUOT':'quotative marker','VOC':'vocative marker','ECHO':'echo form','FILL':'filler',
 'PART':'particle','APPR':'appropriate place','APP':'appropriate place'}
TAM_TAGS={'ipfv','pfv','pres','fut','inf','impv','participle','conjunctive-participle','inchoative-participle','potential-past','indirect-past'}
CURATION=RAW/f'{PREFIX}-curation.json'


def number_gloss(n):
    ones=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
    tens={20:'twenty',30:'thirty',40:'forty',50:'fifty',60:'sixty',70:'seventy',80:'eighty',90:'ninety'}
    if n<20:return ones[n]
    if n<100:return tens[n//10*10]+('-'+ones[n%10] if n%10 else '')
    return {100:'hundred',500:'five hundred',1000:'thousand'}[n]


def parse_gloss(raw,tags='',interlinear=False):
    """Remove only recognised source categories; retain lexical morphemes/senses."""
    found=[]
    categories=[]
    def category(m):
        categories.append(m.group());found.extend(GRAMMAR[m.group()].split());return ''
    # A slash followed by a free sentence is the author's explanatory translation,
    # not another morpheme: store the lexical interlinear analysis before it.
    raw=raw.strip()
    if '/' in raw and re.search(r'\b(?:IMPFV|PFV|POTPST|FUT|LOC|INSTR)\b',raw.split('/')[0]):
        raw=raw.split('/')[0]
    # In prose, a parenthesised interlinear gloss supersedes the free translation
    # as the lexical definition, e.g. "I stopped (stop.PFV.1SG)".
    m=re.search(r'\(([^()]*\b(?:IMPFV|PFV|POTPST|FUT|INF)[^()]*)\)',raw)
    if m and re.match(r'[a-z]+[. −-]',m[1]):raw=m[1]
    lexical=GRAMMAR_RE.sub(category,raw)
    more=GRAMMAR_RE.sub(category,tags)
    found.extend(re.sub(r'[.−()]',' ',more).split())
    lexical=re.sub(r'\(\s*\)','',lexical)
    lexical=re.sub(r'[−–]+',' ',lexical)
    lexical=re.sub(r'(?<=\s)\.|\.(?=\s*[).]|\s*$)',' ',lexical)
    # Hyphens are morpheme separators in interlinear glosses, English punctuation
    # in table/prose definitions (e.g. brother-in-law).
    if interlinear:lexical=lexical.replace('-',' ').replace('.',' ').replace('_',' ')
    lexical=re.sub(r'\s+',' ',lexical).strip(' .−-;,')
    lexical=re.sub(r'\(\s*\)','',lexical).strip()
    lexical=re.sub(r'\s+([,;])',r'\1',lexical)
    if not lexical:
        person=next((c for c in categories if re.fullmatch('[123](SG|PL)',c)),None)
        if 'COP' in categories:lexical='be'
        elif person and not set(found)&TAM_TAGS and 'kinship' not in found:
            lexical={'1SG':'I','2SG':'you (singular)','3SG':'he, she, it','1PL':'we','2PL':'you (plural)','3PL':'they'}[person]
            found.append('pron')
            if 'poss' in found:
                lexical={'1SG':'my','2SG':'your (singular)','3SG':'his, her, its','1PL':'our','2PL':'your (plural)','3PL':'their'}[person]
        else:lexical=next((v for k,v in LEXICAL_MARKERS.items() if k in categories),'')
    if 'pron' in found and 'prox' in found:lexical+=' (proximal)'
    elif 'pron' in found and 'dist' in found:lexical+=' (distal)'
    if set(found)&TAM_TAGS:found.append('verb')
    if (any(re.fullmatch('[123](SG|PL)',c) for c in categories) and
            not set(found)&(TAM_TAGS|{'kinship'}) and lexical in
            {'I','we','you','me','us','my','our','your','he','she','it','him/her/it','his/her','they','them','their'}):
        found.append('pron')
    if (set(found)&{'loc','obl','erg','instr','acc'} and not set(found)&TAM_TAGS
            and 'pron' not in found and lexical and 'REFL' not in categories):found.append('noun')
    if 'kinship' in found and lexical:found.append('noun')
    if lexical in {'be','is','am','are','was','were'}:found.extend(['verb','copula'])
    if lexical=='question marker':found.append('part')
    # Person-number in verbal agreement is never a pronoun POS.
    return lexical,sorted(set(found))


def records():
    rows=[json.loads(line) for line in SNAPSHOT.read_text().splitlines()]
    curation=json.loads(CURATION.read_text()) if CURATION.exists() else {}
    extra=[]
    for r in rows:
        for i,child in enumerate(curation.get(r['unit'],{}).get('children',[]),1):
            extra.append({**r,**child,'unit':r['unit']+f':part{i}',
                          'repair':'Manually delimited member of a compound citation; checked against the source context.'})
    return rows+extra


def locator(r):
    suffix=f"Table {r['table']}, cell {r['column']}, y {round(r['y'])}" if 'table' in r else (
        f"example ({r['example']}), {r['unit'].split(':')[-1]}, y {round(r['y'])}" if r.get('example') else
        f"Appendix 2, y {round(r['y'])}, {r['unit'].split(':')[-1]}" if r['page']>=210 else
        f"prose, y {round(r['y'])}, x {round(r.get('x',0))}")
    refs=sorted(set(re.findall(r'\b(?:[TEQ][W]?[0-9]{4}(?:-[0-9]{2})?|E[0-9]{4})\b',r['context'])))
    return f"p. {r['page']}, {suffix}"+(', data '+', '.join(refs) if refs else '')


def build():
    curated=json.loads(CURATION.read_text()) if CURATION.exists() else {}
    raw=records()
    units={r['unit'] for r in raw}
    if set(curated)-units:raise ValueError(f'stale curation keys: {set(curated)-units}')
    forms=[];audit=[];raw_to_keys={}
    for original in raw:
        r={**original,**curated.get(original['unit'],{})}
        form=r['form'].strip().strip(',;.“”‘’')
        form=form.removeprefix('(').removeprefix('<').removesuffix('…')
        rawgloss=r['gloss'];tags=r.get('tags','')
        if r['region']=='table26':rawgloss=number_gloss(int(rawgloss))
        # A valency annotation belongs to the root immediately before it.
        m=re.search(r'\s*\(([itc])\)\s*$',form)
        if m:form=form[:m.start()].strip();tags+=' verb '+{'i':'intr','t':'tr','c':'caus'}[m[1]]
        elif r['region']=='prose':
            m=re.search(re.escape(original['form'])+r'\s*\(([itc])\)',r['context'])
            if m:tags+=' verb '+{'i':'intr','t':'tr','c':'caus'}[m[1]]
        gloss,taglist=parse_gloss(rawgloss,tags,r['region']=='interlinear')
        status='installed';reason=r.get('repair','') or 'Source form and scoped definition retained; recognised categories separated into Tags.'
        skip=r.get('exclude','')
        if not skip and (form.startswith(('−','–','-')) or form.endswith(('−','–','-'))):
            skip='Isolated bound affix or phonological template; full lexical forms retained separately.'
        if not skip and (not form or form in {'…','...','∅','Ø','−','[ ]'}):skip='Nonlexical punctuation, zero morph, or omission marker.'
        if not skip and not gloss and not r.get('allow_blank'):
            skip='Unglossed metalinguistic citation, repeated form, phoneme, or fragment; no independent lexical definition asserted.'
        if not skip and r['region']=='prose' and r['page'] in {23,24,65,66}:
            skip='Phoneme inventory or kinship diagram; lexical examples/list entries extracted from numbered tables.'
        if skip:status='skipped';reason=skip
        elif r.get('repair'):status='installed_after_repair'
        if r.get('uncertainty') or ('?' in gloss and 'interr' not in taglist):
            taglist.append('uncertain')
            if not r.get('uncertainty'):
                r['uncertainty']='The source questions the lexical gloss or a glossed morpheme.'
        gloss=gloss.replace('(?)','').replace('?','').strip()
        if gloss=='':
            if status!='skipped':r['uncertainty']=r.get('uncertainty') or 'No lexical gloss supplied for this explicitly printed component.'
        variants=r.get('variants')
        if variants is None:
            # Expand only complete printed alternatives, not shorthand suffixes.
            variants=[p.strip() for p in re.split(r'\s*/\s*',form)] if '/' in form else [form]
            if r['region']=='table26' and ',' in form:variants=[p.strip() for p in form.split(',')]
        if any(not v for v in variants) and not skip:raise ValueError(('incomplete alternative',r))
        emitted=[]
        if status!='skipped':
            for vi,v in enumerate(variants,1):
                key=SOURCE_ID+':'+r['unit']+(f':v{vi}' if vi>1 else '')
                row={field:'' for field in FIELDS}
                row.update(Language_ID='Dm',Form=nfc(v),Gloss=gloss,
                    Phonemic=r.get('phonemic',''),Notes=r.get('notes',''),
                    Source=f'{SOURCE_ID}[{locator(r)}]'+(';' + r['citation'] if r.get('citation') else ''),
                    Etymology=r.get('etymology',''),Entry_Key=key,
                    Variant_Of_Key=emitted[0] if vi>1 else r.get('variant_of',''),
                    Derivation_Parent_Keys='|'.join(SOURCE_ID+':'+p for p in r.get('parents',[])),
                    Parameter_ID=r.get('parameter',''),Tags=' '.join(sorted(set(taglist))))
                forms.append(row);emitted.append(key)
        raw_to_keys[r['unit']]=emitted
        audit.append(dict(Unit_ID=r['unit'],Region=r['region'],PDF_Page=r['pdf_page'],Printed_Page=r['page'],
            Raw_Form=original['form'],Raw_Gloss=original['gloss'],Context=original['context'],
            Status=status,Reason=reason,Uncertainty=r.get('uncertainty',''),
            Emitted_Key=';'.join(emitted),Merged_Into=''))
    # Collapse only identical lexical analyses; different grammatical functions,
    # dialects, etymologies and parents cannot be folded into an invented union.
    survivors={};aliases={};output=[]
    for r in forms:
        identity=tuple(r[k] for k in ['Language_ID','Form','Gloss','Phonemic','Tags','Notes','Etymology','Parameter_ID','Variant_Of_Key','Derivation_Parent_Keys'])
        if identity not in survivors:
            survivors[identity]=r;output.append(r)
        else:
            target=survivors[identity]
            target['Source']=';'.join(dict.fromkeys((target['Source']+';'+r['Source']).split(';')))
        aliases[r['Entry_Key']]=survivors[identity]['Entry_Key']
    for r in output:
        for field in ['Variant_Of_Key','Derivation_Parent_Keys']:
            if r[field]:
                keys=r[field].split('|')
                if any(k not in aliases for k in keys):raise ValueError(('missing parent',r,keys))
                r[field]='|'.join(dict.fromkeys(aliases[k] for k in keys if aliases[k]!=r['Entry_Key']))
    by_key={r['Entry_Key']:r for r in output}
    for a in audit:
        keys=a['Emitted_Key'].split(';') if a['Emitted_Key'] else []
        a['Merged_Into']=';'.join(aliases[k] for k in keys if aliases[k]!=k)
        a['Parsed_Records']=json.dumps(
            [by_key[k] for k in dict.fromkeys(aliases[k] for k in keys)],
            ensure_ascii=False,sort_keys=True)
    assert len(units)==len(raw),'duplicate source unit'
    assert len({r['Entry_Key'] for r in output})==len(output)
    return output,audit


def write():
    forms,audit=build()
    with FORM_OUTPUT.open('w',newline='') as f:
        csv.writer(f,lineterminator='\n').writerows([[r[k] for k in FIELDS] for r in forms])
    with (RAW/f'{PREFIX}-audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]),lineterminator='\n');w.writeheader();w.writerows(audit)
    write_manifest(forms,audit)
    print(len(forms),'forms;',Counter(r['Status'] for r in audit))


def write_manifest(forms,audit):
    """Keep the acquisition, census and checked output hashes reproducible."""
    paths=[FORM_OUTPUT,SNAPSHOT,CURATION,RAW/f'{PREFIX}-audit.csv',
           RAW/f'{PREFIX}-sample-round1.csv',RAW/f'{PREFIX}-sample-round2.csv',RAW/f'{PREFIX}-sample.csv',
           ROOT/'conversion/perder-dameli.txt',Path(__file__)]
    manifest={
        'source_id':SOURCE_ID,'snapshot_date':DATE,
        'bibliography':'Perder, Emil. 2013. A Grammatical Description of Dameli. PhD thesis, Stockholm University. ISBN 978-91-7447-770-2.',
        'record_url':'https://urn.kb.se/resolve?urn=urn:nbn:se:su:diva-93888',
        'pdf_url':'https://su.diva-portal.org/smash/get/diva2:651418/FULLTEXT02.pdf',
        'pdf_pages':PDF_PAGES,'printed_to_physical_page_offset':PAGE_OFFSET,
        'pdf_sha256':PDF_SHA256,'pdf_sha512':PDF_SHA512,
        'pdf_sha512_matches_diva_record':True,'pdf_redistributed':False,
        'rights':'Copyright Emil Perder, Stockholm 2013. Open access in DiVA; no explicit reuse licence stated. Only extracted lexical facts are installed; the PDF is not redistributed.',
        'extraction':{
            'method':'Positioned pdfplumber text with explicit column geometry, x-aligned interlinear tiers and font-aware prose citations; no OCR.',
            'precedent':'knobloch_sauji_2020.py (Sauji grammar ingestion)',
            'snapshot_units':sum(1 for line in SNAPSHOT.read_text().splitlines() if line),
            'audited_units':len(audit),'regions':dict(sorted(Counter(r['Region'] for r in audit).items())),
            'curated_units':len(json.loads(CURATION.read_text())),
            'key_policy':'Source ID plus printed page, table/example, vertical position and column/word; explicit :partN child and :vN variant keys. Keys are independent of lexical spelling/gloss. Exact repeated analyses collapse with all source locators retained and aliases documented in the audit.',
        },
        'scope':{
            'included':'Glossed Dameli examples in lexical/phonological/grammatical tables, all 152 verb-root appendix records on pp. 207–208, numbered examples 1–177 (example 38 recovered from prose), Appendix 2 on pp. 210–216 and glossed running-prose citations.',
            'excluded':'Tables 1–3, 33, 37 and 42 are metadata, transcription conventions, corpus index, affix or grammatical schemas; phoneme inventories, repeated kinship diagrams, bare bound affixes, rejected/hypothetical strings, non-Dameli comparanda, unglossed fragments, free translations, and bibliography are not independent Dameli lexical records.',
            'language':'Canonical Dm, dame1241, Kunar. Only source-explicit Aspar forms receive a dialect tag; consultants and recording sessions remain provenance.',
            'etymology':'Source-explicit compounds and derivations use source keys. Only explicit Sanskrit strii has a unique CDIAL match (13734). Named or tentative donor languages remain qualified prose without guessed donor edges.',
        },
        'transcription':{
            'original':'Source Standard Orientalist transcription, preserving tone, length, nasality, source segmentation, syllable boundaries and zero markers.',
            'phonemic':'Only separately bracketed source IPA; no phonetic transcription inferred.',
            'form':'conversion/perder-dameli.txt: č→c, ċ→ʦ, c̣→ʦ̣, š→ś, ǰ→j, ɡ→g, w→v, ẉ→ɻ; length normalized to macrons; source u/o, ŋ/æ, tone/nasality and ṛ/ẉ distinctions retained.',
            'repairs':'Font splits, combining underdots, caron placement, broken table spans and joint citations are explicit in curation and the per-record audit; source lexical uncertainty is retained, not silently resolved.',
        },
        'dialect_evidence':{
            'tag':'dialect:Dm:perder2013-Aspar:Aspar','source_pages':[10,67,68,69],
            'location':'Aspar, Domel (Damel) Valley, Chitral, Khyber Pakhtunkhwa, Pakistan',
            'latitude':35.36798,'longitude':71.70814,'geonames_id':1417235,
            'coordinate_reference':'https://mapcarta.com/15180148 (GeoNames-sourced locality; disambiguated from the northern Aspar locality)',
        },
        'outputs':{
            'form_count':len(forms),'audit_count':len(audit),
            'statuses':dict(Counter(r['Status'] for r in audit)),
            'precollapse_forms':sum(len(r['Emitted_Key'].split(';')) for r in audit if r['Emitted_Key']),
            'variant_rows':sum(bool(r['Variant_Of_Key']) for r in forms),
            'derivation_rows':sum(bool(r['Derivation_Parent_Keys']) for r in forms),
            'derivation_parent_links':sum(len(r['Derivation_Parent_Keys'].split('|')) for r in forms if r['Derivation_Parent_Keys']),
            'external_etymon_rows':sum(bool(r['Parameter_ID']) for r in forms),
            'explicit_dialect_rows':sum('dialect:' in r['Tags'] for r in forms),
            'uncertainty_records':sum(bool(r['Uncertainty']) for r in audit if r['Status']!='skipped'),
        },
        'visual_audits':[
            {'round':1,'seed':3662392299860923243,'population':1859,'sample':20,'material_errors':1,'resolution':'Question clitic no longer assigns particle POS to its host; copular word class recovered and regression tested.'},
            {'round':2,'seed':766152442941636403,'population':1856,'sample':20,'material_errors':0,'result':'20/20 visually checked against rendered source pages; source-scoped form, meaning and grammar agree.'},
            {'round':3,'seed':7698818706101321111,'population':1856,'sample':20,'material_errors':0,'result':'Fresh 20/20 rendered-page audit after restoring the mixed-language p. 30 finger citation and retaining Cacopardo’s alternative notation in notes.'},
        ],
        'unresolved':'Source-qualified historical azâr numeral interpretation, tentative derivations/donor claims, questioned glosses, and four conjunct complements with no independent lexical gloss remain explicitly marked in the audit. No unresolved source glyph is installed.',
        'validation_review':f'data/other/forms/raw_data/{PREFIX}-review.md',
        'checksums':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()},
    }
    (RAW/f'{PREFIX}-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+'\n')


def sample_report(seed,output,count=20):
    """Reproduce a raw-vs-parsed sample without overwriting reviewed verdicts."""
    forms,audit=build()
    units={r['unit']:r for r in records()}
    sampled=[]
    for i,row in enumerate(random.Random(seed).sample(forms,count),1):
        unit=re.sub(r':v[0-9]+$','',row['Entry_Key'].split(':',1)[1])
        raw=units[unit]
        sampled.append(dict(Sample=i,Seed=seed,Entry_Key=row['Entry_Key'],Unit_ID=unit,
            PDF_Page=raw['pdf_page'],Printed_Page=raw['page'],Raw_Form=raw['form'],
            Raw_Gloss=raw['gloss'],Context=raw['context'],Source_Form=row['Form'],
            Gloss=row['Gloss'],Tags=row['Tags'],Source=row['Source'],
            Variant_Of_Key=row['Variant_Of_Key'],Derivation_Parent_Keys=row['Derivation_Parent_Keys'],
            Parameter_ID=row['Parameter_ID'],Result='pending',Observation=''))
    with output.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sampled[0]),lineterminator='\n')
        w.writeheader();w.writerows(sampled)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pdf',type=Path)
    parser.add_argument('--words',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--sample',type=int,help='reproducible sample seed; requires a new --output path')
    args=parser.parse_args()
    if args.write:
        write();raise SystemExit
    if args.sample is not None:
        if not args.output:parser.error('--sample requires --output')
        sample_report(args.sample,args.output);raise SystemExit
    if args.words:
        pages=json.loads(args.words.read_text())
        for p in pages:p['lines']=tokens(p['words'])
        rows=extract_pages(pages)
    else: rows=extract(args.pdf)
    if args.output:args.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(Counter(r['region'] for r in rows))
