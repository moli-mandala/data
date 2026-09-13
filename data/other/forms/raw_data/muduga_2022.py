#!/usr/bin/env python3
"""Import printed Muduga examples from Arsenault & Abraham's author preprint.

The 1100-word elicitation corpus is not published in this article. Tables 3–19,
their phonetic realizations, and the additional glossed Muduga examples in prose
are the lexical scope. Comparison-language citations are retained as evidence,
not imported as new field attestations.
"""
import argparse,csv,hashlib,json,re,unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
RAW=Path(__file__).with_suffix('')
SOURCE='arsenault-abraham2022muduga'
OUT=ROOT/'data/other/forms/20260912-muduga.csv'
DIALECT='dialect:Muduga:muduga_chindakki:Chindakki'
TABLE_RANGES={3:(203,212),4:(215,220),5:(231,240),6:(243,250),7:(268,277),8:(288,296),9:(299,304),10:(327,334),11:(337,344),12:(370,379),13:(391,399),14:(412,419),15:(424,433),16:(654,660),17:(663,666),18:(678,683),19:(691,698)}
# Independently reviewed prose examples. Locators use section/footnote in the
# preprint, avoiding a false mapping to the differently paginated final version.
PROSE=[
 ('fn2',143,'ʧaːrɯ','curry, essence','','saːrɨ','Authors’ pronunciation; distinct from the cited Rajendran forms.'),
 ('s3.1-potherb',255,'kiːɾe','potherb','1617','',''),
 ('s3.1-buffalo',256,'eɾɯme','buffalo','816','',''),
 ('s3.1-hears',279,'kɤːkkæ','he hears','2017a','','Inflected form of kɤːɭɯ.'),
 ('s3.1-bran',316,'t̪aʋuɖɯ','bran','','',''),
 ('s3.1-fly',317,'peːl-t̪urukki','a kind of fly','','',''),
 ('s3.2-green',353,'pæʧʧe','green','3821','pætʧe',''),
 ('s3.2-door',354,'bæːʧa','door','5354','bæːsa',''),
 ('s3.2-wash',359,'kaʧʧɯ-raːd̪ɯ','wash','','katʧɨraːd̪ɨ','The starred fronted form is explicitly rejected and is not an attestation.'),
 ('s3.2-rock',361,'are','rock','321','',''),
 ('s3.2-six',362,'aːrɯ','six','2485','',''),
 ('s3.2-sleep',435,'uraːkkɯ','sleep','707','uraːkkɨ',''),
 ('s3.2-dove',436,'ʧoːre','dove','2885','soːre',''),
 ('fn10',456,'beː-ra','she gets cooked','','','Retraction does not appear to apply across this morpheme boundary.'),
 ('s4.5-beak',628,'t̪otti','beak','','t̪øtti',''),
]

def snapshot():
    cache=ROOT.parent/'tmp/pdfs/keed-muduga-20260912'
    lines={}
    for p in sorted(cache.glob('muduga-*.txt')):
        for n,t in re.findall(r'^L(\d+)@P0: (.*)$',p.read_text(),re.M):
            n=int(n)
            if n in lines:assert lines[n]==t,(n,lines[n],t)
            lines[n]=t
    assert set(lines)==set(range(1065)), sorted(set(range(1065))-set(lines))
    RAW.mkdir(exist_ok=True)
    (RAW/'source-lines.json').write_text(json.dumps(lines,ensure_ascii=False,indent=2)+'\n')
    return lines

def records(lines):
    result=[]
    for table,(start,end) in TABLE_RANGES.items():
        group='';sub=0;current=None
        for n in range(start,end+1):
            t=lines[n]
            match=re.match(r'(?:(\w)\. )?/([^/]+)/ [‘ʻ]([^’ʼ]+)[’ʼ](.*)',t)
            if match:
                g,form,gloss,tail=match.groups()
                if g:group=g;sub=0
                sub+=1
                ident=re.search(r'\s(\d+[a-cd]?)(?:5)?$',tail)
                dedr=ident.group(1) if ident else ''
                if table==8 and form=='mɯɡa':dedr='4764'
                current=dict(key=f'muduga2022:table{table}:{group}:{sub}',line=n,locator=f'preprint Table {table}, {group}.{sub}',form=form,gloss=gloss,dedr=dedr,phonetic='',raw=[t],table=table)
                result.append(current)
            elif current:
                current['raw'].append(t)
                phone=re.match(r'^\[([^]]+)\]',t)
                if phone:current['phonetic']=phone.group(1)
    for key,line,form,gloss,dedr,phone,note in PROSE:
        result.append(dict(key='muduga2022:'+key,line=line,locator='preprint '+key,form=form,gloss=gloss,dedr=dedr,phonetic=phone,note=note,raw=[lines[n] for n in range(max(0,line-1),min(1065,line+3))],table=0))
    return result

def build(rr):
    valid={r[0] for r in csv.reader((ROOT/'data/dedr/params.csv').open())}
    rows=[];audit=[]
    for rec in rr:
        r=dict(rec); form=r['form'];gloss=r['gloss'];dedr=r['dedr']
        tags=[DIALECT];etym=r.get('note','')
        param=''
        if dedr:
            exact=[v for v in valid if re.sub('[()]','',v[1:]).lower()==dedr.lower()]
            base='d'+re.sub('[a-z]$','',dedr)
            param=exact[0] if len(exact)==1 else base if base in valid else ''
        status='linked' if param else 'unlinked'
        if dedr and not param:
            status='unresolved-reference';tags+=['uncertain']
        if form=='mɯɡa':
            param='';status='uncertain-etymology';tags+=['uncertain']
            etym='The authors suspect a relationship to Kuwi mirka (DEDR 4764), but state the etymology is uncertain; they doubt a connection with DEDR 4616 (footnote 5). Neither proposal is installed as accepted ancestry.'
        elif form=='bɯːɳe':
            tags+=['loanword'];status='unresolved-borrowing';etym='The authors identify this as a Sanskrit-origin loan; compare Tamil vīṇai and Sanskrit vīṇā (Table 4c). No immediate donor chain is specified.'
        elif dedr:etym=('The authors compare this form with DEDR '+dedr+'. '+etym).strip()
        if '(DIST)' in gloss or '(PROX)' in gloss:
            tags+=['pron','third-person','sg','m' if gloss.startswith('he') else 'f','dist' if '(DIST)' in gloss else 'prox']
            gloss=re.sub(r' \((DIST|PROX)\)','',gloss)
        if gloss.startswith('to '):tags+=['verb'];gloss=gloss[3:]
        if form=='kɤːkkæ':tags+=['verb','third-person','sg','derived']
        citations=f'{SOURCE}[{r["locator"]}]'+(f';dedr[{dedr}]' if dedr else '')
        derivation='muduga2022:table6:b:1' if form=='kɤːkkæ' else ''
        if derivation:param=''  # Reach the cited DEDR etymon through the printed lexical base.
        if any('TLex' in line for line in r['raw']):citations+=';madras-tamil-lexicon'
        row=['Muduga',param,form,gloss,'',form,'',citations,'',etym,r['key'],'','',derivation,' '.join(tags)]
        emitted=[row]
        if r['phonetic']:
            phone=list(row);phone[1]='';phone[2]=r['phonetic'];phone[9]='Printed phonetic realization of /'+form+'/.';phone[10]+=':phonetic';phone[11]=r['key'];phone[14]+=' sound-variant';emitted.append(phone)
        emitted=[[unicodedata.normalize('NFC',v) for v in row] for row in emitted]
        rows+=emitted;r.update(status=status,parameter=param,rows=emitted,review_reason='web-extraction: author preprint text; no downloaded PDF, verify against typeset copy when available')
        audit.append(r)
    return rows,audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',action='store_true');p.add_argument('--install',action='store_true');args=p.parse_args()
    lines=snapshot() if args.snapshot else {int(k):v for k,v in json.loads((RAW/'source-lines.json').read_text()).items()}
    rr=records(lines);rows,audit=build(rr)
    RAW.mkdir(exist_ok=True)
    with (RAW/'proposed.csv').open('w') as f:csv.writer(f).writerows(rows)
    with (RAW/'audit.jsonl').open('w') as f:
        for r in audit:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    used={n for a,b in TABLE_RANGES.values() for n in range(a,b+1)}|{r[1] for r in PROSE}
    with (RAW/'coverage.jsonl').open('w') as f:
        for n,t in lines.items():f.write(json.dumps(dict(line=n,sha256=hashlib.sha256(t.encode()).hexdigest(),status='lexical-record-or-comparison' if n in used else 'context-or-nonlexical; reviewed for additional Muduga examples'),ensure_ascii=False)+'\n')
    print(json.dumps(dict(records=len(rr),rows=len(rows),linked=sum(bool(r[1]) for r in rows),variants=sum(bool(r[11]) for r in rows))))
    if args.install:
        with OUT.open('w') as f:csv.writer(f).writerows(rows)

if __name__=='__main__':main()
