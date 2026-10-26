"""Prepare Birhor source rows and complete audit. No install/build action."""
import argparse
import collections
import csv
import json
import re
import unicodedata
from pathlib import Path
from prepare_senses import ROOT, OPTIONAL, AMBIGUOUS_HEADS, prepare

SOURCE='living-birhor2026'

def nfc(text):return unicodedata.normalize('NFC',text or '')

def readings(entry_id, phonetic):
    if entry_id not in OPTIONAL:return [phonetic]
    char=OPTIONAL[entry_id];marker=f'({char})'
    assert phonetic.count(marker)==1
    return [phonetic.replace(marker,char),phonetic.replace(marker,'')]

def references():
    result={}
    for ref in json.loads((ROOT/'source-references.json').read_text()):
        citation=ref['citation']; slug=ref['slug']
        if citation.startswith('https://www.tropicos.org/name/'):
            number=citation.rsplit('/',1)[1];assert number.isdigit()
            result[slug]={'role':'botanical-identification','citation':f'tropicos[name {number}]','upstream':ref}
        elif citation.startswith('Lakra, S., Jora, B., Anderson, G.D.S. 2022.'):
            result[slug]={'role':'ethnobotanical-source','citation':'lakra-etal2022birhor-ethnobotany','upstream':ref}
        elif citation.startswith('Photo Source:'):
            result[slug]={'role':'excluded-media-credit','citation':None,'upstream':ref}
        else:raise ValueError(f'Unreviewed source reference {slug}')
    return result

def build():
    candidates,_=prepare();by_entry=collections.defaultdict(list)
    for r in candidates:by_entry[r['entry_id']].append(r)
    refs=references();rows=[];audit=[]
    for line in (ROOT/'lexical-snapshot.jsonl').open():
        raw=json.loads(line);entry=raw['entry'];eid=entry['id'];cs=by_entry[eid]
        record={'entry_id':eid,'raw':raw,'emitted_rows':[],'status':'proposed','issues':list(cs[0]['review']),'source_roles':[]}
        for slug in json.loads(entry['sources'] or '[]'):
            record['source_roles'].append({'slug':slug,**{k:refs[slug][k] for k in ('role','citation')}})
        if eid in AMBIGUOUS_HEADS:
            record['status']='withheld-lexical-scope';audit.append(record);continue
        if any(not c['gloss'] for c in cs):
            record['status']='withheld-missing-English-gloss';audit.append(record);continue
        issues=[v for v in record['issues'] if v!='transcription:optional-segment-expansion-pending']
        record['issues']=issues
        annotations=[]
        if entry['notes']:
            note=json.loads(entry['notes']);assert set(note)=={'default'}
            annotations.append(note['default'])
        if entry['morphology']:annotations.append('Source morphological segmentation: '+entry['morphology'])
        if entry['interlinearization']:annotations.append('Source interlinear analysis: '+entry['interlinearization'])
        note='\n\n'.join(annotations)
        for candidate in cs:
            forms=readings(eid,candidate['source_ipa'])
            for vi,phonetic in enumerate(forms,1):
                key=candidate['entry_key']+f':reading:{vi}'
                tags=list(candidate['tags'])
                if issues:tags.append('uncertain')
                locator=f"entry {eid}, sense {candidate['sense_id']}, meaning {candidate['meaning_number']}"
                citations=[f'{SOURCE}[{locator}]']+[r['citation'] for r in record['source_roles'] if r['citation']]
                row=['Birhor','',candidate['source_headword'],candidate['gloss'],candidate['native'],phonetic,note,';'.join(citations),'','',key,candidate['entry_key']+':reading:1' if vi>1 else '','','',' '.join(tags)]
                row=[nfc(value) for value in row];rows.append(row);record['emitted_rows'].append(row)
        audit.append(record)
    return rows,audit,refs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args();rows,audit,refs=build();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'draft.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
    (a.output/'audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
    (a.output/'reference-roles.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2)+'\n')
    summary={'source_entries':len(audit),'proposed_rows':len(rows),'entry_statuses':dict(collections.Counter(r['status'] for r in audit)),'optional_variant_rows':sum(bool(r[11]) for r in rows),'uncertain_rows':sum('uncertain' in r[14].split() for r in rows),'notes_rows':sum(bool(r[6]) for r in rows),'installed_rows':0}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
