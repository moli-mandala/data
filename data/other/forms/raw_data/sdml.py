"""Import the pinned SDML lexical export; run without --install for a proposal."""
import argparse, collections, csv, hashlib, json, random, re, shutil, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
RAW = Path(__file__).with_name('sdml_2026')
SOURCE = 'sdml2026'
STEM = '20260911-sdml'
def nfc(s): return unicodedata.normalize('NFC', s.strip())
def records():
    manifest = json.loads((RAW/'snapshot.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((RAW/name).read_bytes()).hexdigest() == digest, name
    assert json.loads((RAW/'database.json').read_text()) == []
    return list(csv.DictReader((RAW/'lexical.csv').open()))
def site_id(r):
    return 'sdml-' + '-'.join(re.sub('[^a-z0-9]+','-',r[k].lower().replace('ɡ','g')).strip('-') for k in ('district','taluka','village'))
def gloss(k):
    special={'utensil':'utensil used for drinking water','roof_1':'tiled roof','roof_2':'concrete roof','bolt_1':'door bolt (new fashioned)','bolt_2':'door bolt (old fashioned)','in_front_of':'in front of','drumstick':'drumstick (vegetable)'}
    if k in special:return special[k]
    if k.startswith('broom_'):return 'broom'
    return k.replace('egos',"ego's").replace('fathers',"father's").replace('mothers',"mother's").replace('sisters',"sister's").replace('brothers',"brother's").replace('husbands',"husband's").replace('wifes',"wife's").replace('daughters',"daughter's").replace('sons',"son's").replace('_',' ')
def propose(out):
    raw=records();assert len(raw)==271
    concepts=[k for k in raw[0] if k+'_frequency' in raw[0]];assert len(concepts)==73
    rows=[];audit=[];dialects=[]
    for ri,r in enumerate(raw,1):
        sid=site_id(r);name=nfc(r['village']);lat=float(r['latitudes']);lon=float(r['longitudes'])
        assert 15<lat<23 and 72<lon<82,(sid,lat,lon)
        tag=f'dialect:M:{sid}:{name.replace(" ","_")}'
        dialects.append([sid,tag,'M',sid,name,'',str(lat),str(lon),'Marathi-Konkani',', '.join(nfc(r[k]) for k in ('village','taluka','district'))+', Maharashtra, India','A'])
        for k in concepts:
            value=r[k];tokens=value.split(',');freq=r[k+'_frequency'].split(',')
            aligned=len(tokens)==len(freq) and all(f.strip().isdigit() for f in freq)
            for vi,token in enumerate(tokens,1):
                form=nfc(token);key=f'{SOURCE}:{sid}:{k}:v{vi}'
                a=dict(entry_key=key,row=ri,site=sid,concept=k,raw_cell=value,raw_token=token,raw_frequency=r[k+'_frequency'],frequency=int(freq[vi-1]) if aligned else None,form=form,gloss=gloss(k),status='unlinked',reasons=[])
                if not aligned:a['reasons'].append('frequency:missing-or-unmatched-list')
                if form in ('','NA','.'):
                    a['status']='skipped';a['reasons'].append('missing-response')
                elif re.search(r'[()/~.0-9]',form):
                    a['status']='ambiguous';a['reasons'].append('transcription:unresolved-alternation-or-malformed-cell')
                else:
                    if re.search(r'[A-Zⱳ̧᷄ᵏᵬ͏̝̞ⁿɫ]',form):a['reasons'].append('transcription:unusual-source-symbol-preserved')
                    tags=tag+(' uncertain' if any(s.startswith('transcription:') for s in a['reasons']) else '')
                    rows.append(['M','',form,gloss(k),'',form,'',f'{SOURCE}[{sid}, {k}, variant {vi}]','','',key,'','','',tags])
                audit.append(a)
    assert len({d[0] for d in dialects})==271
    assert len({r[10] for r in rows})==len(rows)
    out.mkdir(parents=True,exist_ok=True)
    with (out/(STEM+'.csv')).open('w') as f:csv.writer(f).writerows(rows)
    with (out/(STEM+'-audit.jsonl')).open('w') as f:
        for a in audit:f.write(json.dumps(a,ensure_ascii=False)+'\n')
    with (out/'dialects.csv').open('w') as f:csv.writer(f).writerows(dialects)
    report={'villages':len(raw),'concept_columns':len(concepts),'raw_cells':len(raw)*len(concepts),'tokens':len(audit),'installed':len(rows),'statuses':dict(collections.Counter(a['status'] for a in audit)),'frequency_unresolved_tokens':sum(a['frequency'] is None for a in audit),'uncertain_forms':sum(' uncertain' in r[14] for r in rows)}
    (out/(STEM+'-manifest.json')).write_text(json.dumps(report,indent=2)+'\n')
    (out/(STEM+'-sample.json')).write_text(json.dumps(random.Random(20260911).sample([a for a in audit if a['status']=='unlinked'],20),ensure_ascii=False,indent=2)+'\n')
    return report

def install(out):
    shutil.copyfile(out/(STEM+'.csv'),ROOT/'data/other/forms'/ (STEM+'.csv'))
    for suffix in ('-audit.jsonl','-manifest.json','-sample.json'):
        shutil.copyfile(out/(STEM+suffix),RAW/(STEM+suffix))
    path=ROOT/'cldf/dialects.csv'
    with path.open() as f: existing=list(csv.reader(f))
    ids={r[0] for r in existing}; additions=list(csv.reader((out/'dialects.csv').open()))
    for d in additions:
        if d[0] in ids:assert next(r for r in existing if r[0]==d[0])==d
        else:existing.append(d)
    with path.open('w') as f:csv.writer(f).writerows(existing)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT.parent/'tmp/sdml-20260911/proposal');p.add_argument('--install',action='store_true');args=p.parse_args()
    print(json.dumps(propose(args.output),indent=2))
    if args.install:install(args.output)
