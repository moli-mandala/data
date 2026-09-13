#!/usr/bin/env python3
"""Ingest the 2018 second/first electronic KEED edition from glyph-aware evidence.

No OCR. Re-extract with `uv run --with pymupdf python .../keed_2018.py
--extract --pdf PATH --cache PATH`; the small pinned record cache is checked in.
"""
import argparse,collections,csv,gzip,hashlib,importlib.util,json,re,subprocess,sys,unicodedata,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
PACKAGE=Path(__file__).with_suffix('')
SOURCE='uchida-rajapurohit2018'
OUT=ROOT/'data/other/forms/20260912-keed.csv'
spec=importlib.util.spec_from_file_location('keed_parse',PACKAGE/'parse.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)

def norm(s):return unicodedata.normalize('NFC',s)
def native_form(s):return re.sub(r'[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]','',s).strip()
def units():
    records=parser.split_records();result=[]
    for r in records:
        for u in r['units']:result.append(dict(parser.parse(u),page=r['page'],column=r['column'],ordinal=r['ordinal']))
    return result,records

def build():
    uu,records=units();valid={r[0] for r in csv.reader((ROOT/'data/dedr/params.csv').open())}
    cdial={r[0] for r in csv.reader((ROOT/'data/cdial/params.csv').open())}
    names=collections.defaultdict(list)
    for u in uu:names[u['form']].append(u['key'])
    dialects=json.loads((PACKAGE/'dialect-map.json').read_text())
    biblios=json.loads((PACKAGE/'reference-map.json').read_text())
    rows=[];audit=[]
    for u in uu:
        key=u['key'];form=u['form'];tags=u['tags'][:];issues=[]
        if '?' in u['gloss']:issues.append('gloss: source question mark retained in definition')
        if '?' in u['reg']:issues.append('transcription-or-attestation: source headword question mark')
        if any(x in u['reg'] for x in ('♠','♢')):issues.append('orthography: source marks nonstandard or uncommon spelling')
        citations=[f'{SOURCE}[p. {u["physical_page"]}, col. {u["physical_column"]}, entry {u["ordinal"]}'+(f', subentry {key.split(":sub:")[1]}' if ':sub:' in key else '')+']']
        ety=u['ety'];param='';borrow='';parents=[];donors=[];ref_targets=[]
        for e in ety:
            if re.search(r'^(?:Ka\.?)?\s*\d|\[\d+\]',e):issues.append('etymology: unprefixed numerical reference: '+e)
            for m in re.findall(r'\bM(\d+(?:\.\d+)*)',e):citations.append('mayrhofer-kewa['+m+']')
            for kind,num in re.findall(r'\b([DACT])(\d+(?:\([a-z]\)|[a-z])?)',e):
                target=('da' if kind=='A' else 'd')+num if kind not in {'T','C'} else num
                if kind not in {'T','C'} and target not in valid:
                    plain=re.sub('[()]','',target)
                    matches=[v for v in valid if re.sub('[()]','',v)==plain]
                    target=matches[0] if len(matches)==1 else re.sub(r'\(?[a-z]\)?$','',target)
                available=target in (cdial if kind in {'T','C'} else valid)
                citations.append(('CDIAL' if kind in {'T','C'} else 'dedr')+'['+('App. ' if kind=='A' else '')+num+']')
                ref_targets.append(dict(printed=kind+num,target=target,exists=available))
                if not available:issues.append('etymology: unavailable printed reference '+kind+num)
            # Only a direct Kannada etymon citation becomes inherited ancestry.
            direct=re.fullmatch(r'(?:Ka\.\s*)?(?:(?:caus|onom|mim)\.\s*)?\*?([DA])(\d+(?:\([a-z]\)|[a-z])?)\.?',e)
            if direct:
                tt=[t for t in ref_targets if t['printed']==''.join(direct.groups()) and t['exists']]
                if len(tt)==1:
                    if param and param!=tt[0]['target']:issues.append('etymology: multiple direct etyma');param=''
                    else:param=tt[0]['target']
            if '?' in e:issues.append('etymology: source question mark: '+e)
            # Explicit immediate donor only; exclude comparative chains and reverse loans.
            donor_text=re.sub(r'(?<=[a-zāīūēō])([DAT]\d)',r' \1',e)
            dm=re.fullmatch(r'(Sk|H|M|Eg|Ar|Pe)\.?\s+([^ ,;/?<>‰]+?)(?:[ .]+(?:\*?[DACTM]\d+(?:\.\d+)*(?:\([a-z]\))?[, .]*)+)?',donor_text)
            if dm and not dm[2].startswith('*'):
                donor=dm[2].rstrip('.');donor_lang={'Sk':'Sk','H':'H','M':'M','Eg':'Eng','Ar':'Ar','Pe':'Pers'}[dm[1]];dk=key+':donor:'+str(len(donors)+1)
                if donor and re.fullmatch(r'[a-záāīūēōñśṣṭḍṇṃḥʰ\u0300-\u036f-]+',donor):
                    ts=[t['target'] for t in ref_targets if t['printed'].startswith(('T','C')) and t['exists']]
                    donors.append([donor_lang,ts[0] if len(set(ts))==1 else '',donor,'','','','',';'.join(dict.fromkeys(citations)),'','Explicit '+donor_lang+' donor cited by KEED.',dk,'','','',''])
                    borrow=dk;tags.append('loanword');param=''
            # Resolve explicitly printed compounds/derivations only by unique source headwords.
            if '+' in e and '?' not in e:
                if e.startswith('+') and u['parent']:parents.append(u['parent'])
                for component in re.split(r'\s*\+\s*',e):
                    if re.match(r'(?:Sk|H|M|Eg|Ar|Ta|Ma|Te)\.',component):continue
                    component=re.sub(r'^Ka\.\s*','',component)
                    component=re.sub(r'\s+[DAT]\d.*','',component).strip()
                    if component in names and len(names[component])==1 and names[component][0]!=key:parents.append(names[component][0])
        if any('multiple direct' in x or 'source question' in x for x in issues):param=''
        for label,tag in dialects.items():
            if re.search(r'(?<!\w)'+re.escape(label)+r'(?!\w)', ' '.join(u['aux_citations'])):tags.append(tag)
        unresolved=[]
        for citation in u['aux_citations']:
            found=False
            for label,bib in sorted(biblios.items(),key=lambda x:-len(x[0])):
                if label.startswith('Si') and re.search(r'Si\.\s*&\s*Pl',citation):continue
                if re.search(r'(?<!\w)'+re.escape(label)+r'(?![^\W\d_])',citation):citations.append(bib+'['+parser.tidy(citation).replace(';',',').replace('[','(').replace(']',')')+']');found=True
            if not found and not any(label in citation for label in dialects) and citation not in ('C.','NK','SK','SK.'):
                unresolved.append(citation)
        if unresolved:issues.append('reference: unresolved auxiliary citation')
        if ref_targets and not param and not borrow and not parents:issues.append('etymology: complex comparison retained without an accepted ancestor')
        if issues:tags.append('uncertain')
        # Preserve source prose and unresolved relationships in their etymological field.
        commentary='; '.join(ety)
        if u['crossrefs']:commentary+=('; ' if commentary else '')+'Cross-references: '+ '; '.join(n+' ('+r+')'+h for n,r,h in u['crossrefs'])
        row=['Kannada',param,form,u['gloss'],u['native'],u['ipa'],'',';'.join(dict.fromkeys(citations)),'',commentary,key,'',borrow,'|'.join(dict.fromkeys(parents)),' '.join(dict.fromkeys(tags))]
        emitted=[row]+donors
        abbreviated=re.fullmatch(r'(.+)([rlṟ])/\2u',form)
        if abbreviated:
            first=abbreviated[1]+abbreviated[2];second=first+'u'
            phones=u['ipa'].split('/');native=u['native'].split('/')[0]
            row[2]=first;row[4]=native;row[5]=phones[0]
            child=row.copy();child[1]='';child[2]=second;child[4]=native.removesuffix('್')+'ು';child[5]=phones[1] if len(phones)==2 else '';child[10]=key+':head-variant:2';child[11]=key;child[12]=child[13]='';child[14]+=' alternate';emitted.append(child)

        for i,alt in enumerate(u['alternates'],1):
            native=native_form(alt)
            if not native or native==u['native']:continue
            child=row.copy();child[1]='';child[2]=child[4]=native;child[5]='';child[10]=key+f':variant:{i}';child[11]=key;child[12]=child[13]='';child[14]+=' alternate';emitted.append(child)
        for i,phone in enumerate(u['extra_ipa'],1):
            child=row.copy();child[1]='';child[5]=phone;child[10]=key+f':phonetic:{i}';child[11]=key;child[12]=child[13]='';child[14]+=' sound-variant';emitted.append(child)
        # Source paradigms supply native forms only; preserve script as Original rather than inventing romanization.
        for i,p in enumerate(u['paradigms'],1):
            label=re.match(r'\s*\.?\s*(honorific pl\.|gen\./dat\.|dat\./gen\.|p\.part\.|fut\.|past\.?|redup\.|redp\.|caus\.|ibc\.|ifc\.|obl\.|gen\.?|dat\.|pl\.|sg\.|mf\.|f\.?|m\.|vt\.|n\.|neg\.)\s*',p)
            if not label:continue
            native=p[label.end():].strip();colloquial=native.startswith('*');native=native.lstrip('*').strip()
            if not re.fullmatch(r'[\u0c80-\u0cff/ ,–—-]+',native):continue
            for j,n in enumerate(re.split('[/,]',native),1):
                n=n.strip()
                if not n or n.startswith(('—','–','-')):continue
                gt={'f.':'noun f','f':'noun f','m.':'noun m','mf.':'noun mf','n.':'noun','pl.':'pl','sg.':'sg','past':'pret','past.':'pret','p.part.':'pp','honorific pl.':'honorific pl','gen./dat.':'gen dat','dat./gen.':'dat gen','fut.':'fut','redup.':'reduplicated','redp.':'reduplicated','caus.':'verb caus','ibc.':'stem compound','ifc.':'stem compound','obl.':'obl','gen':'gen','gen.':'gen','dat.':'dat','vt.':'verb tr','neg.':'neg'}[label[1]]+(' colloquial' if colloquial else '')
                child=['Kannada','',n,'',n,'','',citations[0],'','Printed '+label[1]+' form of '+form+'.',key+f':paradigm:{i}:{j}','','',key,gt+' derived']
                emitted.append(child)
        emitted=[[norm(v) for v in r] for r in emitted];rows.extend(emitted)
        audit.append(dict(**u,status='linked' if param else 'borrowed' if borrow else 'unlinked',rows=emitted,printed_references=ref_targets,unresolved_citations=unresolved,issues=issues))
    return rows,audit,records

def main():
    p=argparse.ArgumentParser();p.add_argument('--install',action='store_true');p.add_argument('--extract',action='store_true');p.add_argument('--pdf',type=Path);p.add_argument('--cache',type=Path);args=p.parse_args()
    if args.extract:
        if not args.pdf or not args.cache:p.error('--extract requires --pdf and --cache')
        manifest=json.loads((PACKAGE/'acquisition.json').read_text())
        assert hashlib.sha256(args.pdf.read_bytes()).hexdigest()==manifest['sha256']
        args.cache.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,KEED_PDF=str(args.pdf.resolve()),KEED_CACHE=str(args.cache.resolve()))
        subprocess.run([sys.executable,str(PACKAGE/'trace_extract.py')],env=env,check=True)
        subprocess.run([sys.executable,str(PACKAGE/'trace_parse.py')],env=env,check=True)
        with (args.cache/'trace-records.jsonl').open('rb') as f,gzip.open(PACKAGE/'trace-records.jsonl.gz','wb') as g:g.write(f.read())
    rows,audit,records=build()
    with (PACKAGE/'proposed.csv').open('w') as f:csv.writer(f).writerows(rows)
    with gzip.open(PACKAGE/'audit.jsonl.gz','wt') as f:
        for r in audit:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with gzip.open(PACKAGE/'coverage.jsonl.gz','wt') as f:
        for r in records:
            r=dict(r);r.pop('units',None)
            if r['key']=='keed2018:p698:c1:e21':r['status']='wrapped-headword-prefix: continued in e22'
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps(dict(source_records=len(records),lexical_units=len(audit),rows=len(rows),languages=dict(collections.Counter(r[0] for r in rows)),linked=sum(bool(r[1]) for r in rows),borrowed=sum(bool(r[12]) for r in rows),variants=sum(bool(r[11]) for r in rows),derived=sum(bool(r[13]) for r in rows),unresolved_references=collections.Counter(x for a in audit for x in a['unresolved_citations']))))
    if args.install:
        with OUT.open('w') as f:csv.writer(f).writerows(rows)
if __name__=='__main__':main()
