#!/usr/bin/env python3
"""Lossless, review-only Lāḷas DDSA proposal. No installation until scope gates pass."""
import argparse
import collections
import csv
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[4]
SOURCE = 'lalasa2013'
# Verified from printed frontmatter abbreviation list, PDF pp. 288–291.
GRAMMAR = {'सं.पु.':'noun m','सं.स्त्री.':'noun f','सं.उ.':'noun mf',
           'क्रि.स.':'verb tr','क्रि.अ.':'verb intr','क्रि.वि.':'adv',
           'क्रि.प्रे.':'verb caus','क्रि.':'verb','वि.':'adj',
           'सर्व.':'pron','अव्य.':'indecl','भू.का.कृ.':'participle pp',
           'व.का.कृ.':'participle pres'}
POS = re.compile('|'.join(re.escape(k) for k in sorted(GRAMMAR,key=len,reverse=True)))
MORPH = re.compile(r'(?:क्रि\.प्र\.|स्त्रीलिंग|प्रे\.रू\.|भाव\s*वा\.|कर्मवा\.)')
DONOR = re.compile(r'^\[?(?:सं\.|प्रा\.|अप\.|फा\.|अ\.|अं\.|तु\.|रा\.|राज\.)')
CIT = re.compile(r'⟪C(\d+)⟫')


def plain(x):
    text=x.get_text(' ',strip=True) if hasattr(x,'get_text') else str(x)
    return unicodedata.normalize('NFC',re.sub(r'\s+',' ',text)).strip()


def parse_article(markup,page,ordinal):
    key=f'{SOURCE}:p{page}:e{ordinal}'
    audit={'entry_key':key,'page':page,'ordinal':ordinal,'raw_markup':markup,
           'status':'proposed','rows':[],'review':[],'citations':[],
           'subentry_regions':[],'crossreferences':[]}
    node=BeautifulSoup(markup,'html.parser')
    heads=node.find_all('hw')
    if len(heads)!=1:
        audit.update(status='excluded',review=['structure:headword-count']);return audit
    pairs=[plain(b) for b in heads[0].find_all('b')]
    if not pairs or len(pairs)%2:
        audit.update(status='excluded',review=['structure:unpaired-headwords']);return audit
    forms=list(zip(pairs[::2],pairs[1::2]));heads[0].decompose()
    if any(not re.search(r'[\u0900-\u097f]',n) or re.search(r'[\u0900-\u097f�]',r) for n,r in forms):
        audit.update(status='excluded',review=['transcription:invalid-headword-pair']);return audit
    for cit in node.find_all('cit'):
        i=len(audit['citations'])
        audit['citations'].append({'quote':plain(cit.find('quote') or ''),
                                  'reference':plain(cit.find('ref') or ''),'raw_markup':str(cit)})
        cit.replace_with(f' ⟪C{i}⟫ ')
    # Bold lexical families require independent native-to-roman expansion and
    # grammatical scope review. Retain the entire tail, not merely the bold head.
    for i,b in enumerate(node.find_all('b')):
        b.replace_with(f' ⟪B{i}⟫ '+plain(b))
    body=plain(node)
    feminine=re.match(r'^स्त्रीलिंग\s*[-–—]+\s*([^\s]+)\s*',body)
    if feminine:
        audit['subentry_regions'].append(feminine[0].strip())
        audit['review'].append('morphology:feminine-head-pending')
        body=body[feminine.end():]
    child=re.search(r'⟪B\d+⟫',body)
    if child:
        audit['subentry_regions'].append(body[child.start():])
        audit['review'].append('structure:derived-family-pending')
        body=body[:child.start()].rstrip()
    # Verbal usage (क्रि.प्र.) is not the causative label क्रि.प्रे.
    morph=MORPH.search(body)
    if morph:
        audit['subentry_regions'].append(body[morph.start():])
        audit['review'].append('morphology:usage-or-inflection-pending')
        body=body[:morph.start()].rstrip()
    # वि.वि. is विशेष विवरण, not two adjective labels (printed p. 236).
    detail=re.search(r'वि\.वि\.\s*[-–—]*',body)
    detail_note=''
    if detail:
        detail_note=body[detail.end():].strip()
        body=body[:detail.start()].rstrip()
    # Labels inside a See qualifier or a Sanskrit equivalent do not start a
    # grammatical section. In particular, (वि.) must not create a blank sense.
    protected=[m.span() for m in re.finditer(r'\([^)]*\)|\[[^]]*\]',body)]
    matches=[m for m in POS.finditer(body)
             if not any(a<=m.start()<b for a,b in protected)]
    intro=body[:matches[0].start()] if matches else ''
    etymology=plain(intro) if DONOR.match(intro) else ''
    common_notes=plain(intro) if intro and not etymology else ''
    sections=[(GRAMMAR[m.group()],body[m.end():matches[i+1].start() if i+1<len(matches) else len(body)]) for i,m in enumerate(matches)]
    if not sections:sections=[('',body)]
    sense=0
    for tags,section in sections:
        section=section.strip(' -–—')
        candidates=list(re.finditer(r'(?<!\S)([0-9०-९]+)\.',section))
        nums=[]
        for candidate in candidates:
            n=int(candidate[1])
            prefix=section[:candidate.start()].strip()
            if not nums and (n==1 or (1<=n<100 and prefix in {'','यौ.'})):
                nums.append(candidate)
            elif nums and n in {1,int(nums[-1][1]),int(nums[-1][1])+1}:
                if n==int(nums[-1][1]):
                    audit['review'].append('source:duplicate-sense-number')
                elif n==1:
                    audit['review'].append('scope:numbering-reset-without-POS')
                nums.append(candidate)
        units=[(m.group(1),section[m.end():nums[i+1].start() if i+1<len(nums) else len(section)]) for i,m in enumerate(nums)] if nums else [('',section)]
        if nums and section[:nums[0].start()].strip():
            audit['review'].append('scope:prose-before-numbered-sense')
        for number,text in units:
            sense+=1
            refs=[]
            def remove_cit(m):
                refs.append(int(m[1]));return ''
            gloss=CIT.sub(remove_cit,text).strip(' ।.;,-–—')
            alternate=re.search(r'रू\.भे\.\s+([^।]+)',gloss)
            if alternate:
                audit['subentry_regions'].append(alternate[0])
                gloss=(gloss[:alternate.start()]+gloss[alternate.end():]).strip(' ।.;,')
                audit['review'].append('structure:inline-alternates-pending')
            gloss=re.sub(r'^यौ\.\s*','',gloss)
            notes=[common_notes] if common_notes else []
            if detail_note:notes.append(detail_note)
            notes += [audit['citations'][i]['quote'] for i in refs]
            cross=re.fullmatch(r"देखो\s+['‘]([^'’]+)['’][।.\s]*(?:\([^)]*\)[।.\s]*)*",gloss)
            if cross:
                audit['crossreferences'].append({'sense':sense,'target':cross[1]})
                notes.append(gloss);gloss=''
            review=list(audit['review'])
            if not gloss and not cross:review.append('source:missing-definition-or-displaced-POS')
            if re.search(r'\([^)]*\)|\[[^]]*\]',gloss):review.append('scope:parenthetical-label-or-equivalent')
            if 'देखो' in gloss:review.append('reference:inline-see-pending')
            if refs:review.append('reference:abbreviation-resolution-pending')
            for vi,(native,roman) in enumerate(forms,1):
                poetic='*' in roman or '*' in gloss
                native=native.replace('*','');roman=roman.replace('*','')
                rowtags=tags.split()+(['poetic'] if poetic else [])+(['uncertain'] if review else [])
                rowkey=f'{key}:s{sense}:v{vi}'
                locator=f'DDSA web page {page}, article {ordinal}'+(f', sense {number}' if number else '')
                row=['Rj','',roman,gloss.replace('*',''),native,'','; '.join(notes),
                     f'{SOURCE}[{locator}]','',etymology,rowkey,
                     f'{key}:s{sense}:v1' if vi>1 else '','','',' '.join(dict.fromkeys(rowtags))]
                audit['rows'].append({'row':row,'review':review,'printed_sense':number,'citation_indices':refs})
    return audit


def propose(cache,output,allow_partial=False,seed=20260914):
    manifest=json.loads((cache/'manifest.json').read_text())
    if not manifest['complete'] and not allow_partial:raise ValueError('Acquisition incomplete')
    output.mkdir(parents=True,exist_ok=True);counts=collections.Counter();flags=collections.Counter();sample=[];rng=random.Random(seed)
    with (output/'audit.jsonl').open('w') as af,(output/'proposed.csv').open('w') as cf:
        writer=csv.writer(cf)
        for p in manifest['pages']:
            raw=(cache/f"{p['page']:04d}.html").read_bytes();assert hashlib.sha256(raw).hexdigest()==p['sha256']
            heads=BeautifulSoup(raw.decode(),'html.parser').find_all('hw');assert len(heads)==p['headwords']
            for ordinal,h in enumerate(heads,1):
                record=parse_article(str(h.parent),p['page'],ordinal)
                af.write(json.dumps(record,ensure_ascii=False)+'\n');counts['articles']+=1;counts[record['status']]+=1
                for item in record['rows']:writer.writerow(item['row']);counts['rows']+=1;flags.update(item['review'])
                if len(sample)<20:sample.append(record)
                else:
                    n=rng.randrange(counts['articles'])
                    if n<20:sample[n]=record
    report={'source':SOURCE,'snapshot_complete':manifest['complete'],'counts':dict(counts),'review_classes':dict(flags),'audit_seed':seed,
            'installation_ready':False,'deferred_gates':['edition-coverage','derived-families','grammar-and-sense-scope','references','dialects','sound-profile','fresh-source-audit','full-build','browser-QA']}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (output/'sample.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=ROOT/'tmp/lalasa-ddsa-20260911');p.add_argument('--output',type=Path,default=ROOT/'tmp/lalasa-proposal-20260911');p.add_argument('--allow-partial',action='store_true');p.add_argument('--seed',type=int,default=20260914);a=p.parse_args()
    print(json.dumps(propose(a.cache,a.output,a.allow_partial,a.seed),ensure_ascii=False))
