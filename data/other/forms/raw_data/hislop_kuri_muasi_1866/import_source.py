"""Reproduce the complete Hislop/Temple Kuri/Muasi lexical evidence inventory."""
import argparse
import csv
import json
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = '20260925-hislop-kuri-muasi'
SOURCE = 'hislop1866papers'
DIALECT = 'dialect:ko:hislop-1866-kuri-muasi:Kuri%20or%20Muasi'
SITE_TAGS = {'Elliott': 'dialect:ko:hislop-1866-korku-kalibheet:Kalibheet', 'Bradley': 'dialect:ko:hislop-1866-korku-gawil:Gawil%20hills', 'Voysey': 'dialect:ko:hislop-1866-korku-hoshungabad-berar:Hoshungabad%E2%80%93Berar%20hills'}
EXPECTED_BY_PAGE = {1: 7, **{p: (8 if p % 2 == 0 else 9) for p in range(2,43)}, 43: 4}
NUMERALS = {'one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','twenty','thirty','ninety','hundred','three quarters','half','quarter'}
PRONOUNS = {'I':['pron','1sg'], 'we':['pron','1pl'], 'ye':['pron','2pl'], 'thou':['pron','2sg'], 'he':['pron','3sg'], 'she':['pron','3sg'], 'they':['pron','3pl'], 'mine':['pron','1sg','poss'], 'ours':['pron','1pl','poss'], 'thine':['pron','2sg','poss'], 'what':['pron','interr'], 'who':['pron','interr']}

def read(name):
    with (ROOT / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))

def build():
    main, comparison, prose = read('full_inventory.tsv'), read('comparison_inventory.tsv'), read('prose_inventory.tsv')
    assert Counter(int(x['page']) for x in main) == EXPECTED_BY_PAGE
    assert len(main)==359 and len(comparison)==100 and len(prose)==5
    assert Counter(x['status'] for x in main)=={'selected':240,'source_blank':119}
    assert Counter(x['status'] for x in comparison)=={'selected':63,'held':2,'source_blank':35}
    rows, audit, exact = [], [], {}
    for section, inventory in [('vocabulary',main),('comparison',comparison),('essay',prose)]:
        for item in inventory:
            number = int(item['item']); page = int(item.get('page') or 0)
            witness = item.get('witness','Hislop')
            if section=='vocabulary':
                base=f'{SOURCE}:kuri:p{page}:item{number:02}'
                locator=f'Vocabulary p. {page}, Kuri or Muasi, item {number}'
            elif section=='comparison':
                base=f'{SOURCE}:kuri:comparison:item{number:02}:{witness.lower()}'
                locator=f'Comparative Vocabulary of Muasi or Kuri, {witness}, item {number}'
            else:
                base=f'{SOURCE}:kuri:essay:p{page}:item{number:02}'
                locator=f'Essay p. {page}, comparative prose, item {number}'
            forms=[x.strip() for x in item['forms'].split(';') if x.strip()]
            status=item['status'];assert bool(forms)==(status!='source_blank')
            record={'entry_key':base,'section':section,'printed_vocabulary_page':page if section=='vocabulary' else None,'printed_page':page or None,'scan_page':int(item['scan_page']),'item':number,'witness':witness,'gloss':item['gloss'],'source_forms':forms,'raw_record':dict(item),'status':status,'review_flags':[],'note':item['note'],'emitted_keys':[],'reuse_entry_keys':[],'parsed_rows':[]}
            if 'review:' in item['note']:record['review_flags'].append(item['note'])
            audit.append(record)
            if status!='selected':continue
            for alt, form in enumerate(forms,1):
                assert form==unicodedata.normalize('NFC',form)
                key=base if len(forms)==1 else f'{base}:alt{alt}'
                gloss=item['gloss'];tags=[DIALECT];notes=[]
                if section=='comparison' and witness in SITE_TAGS:tags=[SITE_TAGS[witness]]
                if gloss.endswith(' (v.)'):gloss=gloss[:-5];tags.append('verb')
                if gloss in NUMERALS:tags.append('num')
                tags.extend(PRONOUNS.get(gloss,[]))
                if (page,number,form,section)==(22,4,'Halka','vocabulary'):tags.append('adj')
                if section=='vocabulary' and (page,number)==(27,8) and form=='Thora':gloss='wild plantain'
                if section=='vocabulary' and (page,number)==(26,8):gloss='old (age)'
                if record['review_flags']:tags.append('uncertain')
                if section=='comparison':
                    notes.append(f'Collector witness: {witness}.')
                    if witness=='Elliott':notes.append('Korkus of Kalibheet, hills southwest of Hoshungabad; memorandum transmitted 1865.')
                    elif witness=='Voysey':notes.append('1821 vocabulary of hill tribes between Hoshungabad and Berar, as cited by Temple.')
                    elif witness=='Bradley':notes.append('Gawil hills north of Berar, vocabulary cited by Temple from 1846 proceedings.')
                    elif witness=='Pearson':notes.append('Korku words furnished to Hislop in 1863, identified with Muasi by Pearson.')
                if section=='essay':notes.append('Source jointly attributes this form to Kúrs and Kóls; the individual lect attribution is unresolved.')
                row=['ko','',form,gloss,'','',' '.join(notes),f'{SOURCE}[{locator}]','',item.get('comparison',''),key,'','','',' '.join(tags)]
                # Reuse only this book's own identical Hislop head and sense, never another collector.
                lookup=(form,gloss)
                if section=='comparison' and witness=='Hislop' and lookup in exact:
                    target=exact[lookup];citation=row[7]
                    if citation not in target[7].split(';'):target[7]+=';'+citation
                    record['reuse_entry_keys'].append(target[10]);record['status']='reused_same_source_attestation'
                else:
                    rows.append(row);record['emitted_keys'].append(key)
                    if section=='vocabulary':exact[lookup]=row
                record['parsed_rows'].append(row)
    assert len(audit)==464 and len({a['entry_key'] for a in audit})==464
    assert len({r[10] for r in rows})==len(rows)
    # All original installed source keys are durable across repaired transcription.
    legacy={f'{SOURCE}:kuri:p{p}:item{n:02}'+suffix for p,n,suffix in [(1,3,''),(1,5,''),(2,2,''),(2,4,''),(2,5,''),(2,6,''),(2,8,''),(3,2,''),(3,5,''),(3,6,''),(3,7,''),(3,8,''),(3,9,''),(4,1,':alt1'),(4,1,':alt2'),(4,4,''),(4,5,''),(4,8,'')]}
    assert legacy <= {r[10] for r in rows}
    return rows,audit

def write(install=False):
    rows,audit=build()
    with (ROOT/'staged.csv').open('w',encoding='utf-8',newline='') as f:csv.writer(f).writerows(rows)
    with (ROOT/'staged-audit.jsonl').open('w',encoding='utf-8') as f:
        for a in audit:f.write(json.dumps(a,ensure_ascii=False,sort_keys=True)+'\n')
    if install:
        shutil.copyfile(ROOT/'staged.csv',ROOT.parents[1]/f'{STEM}.csv')
        shutil.copyfile(ROOT/'staged-audit.jsonl',ROOT/'audit.jsonl')
    print(json.dumps({'rows':len(rows),'audit_units':len(audit),'statuses':dict(Counter(a['status'] for a in audit))}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--install',action='store_true');write(p.parse_args().install)
