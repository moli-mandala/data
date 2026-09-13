"""Pinned Angika, Majhi, Koraga and Orissa lexical sources with complete audits.

Offline --output stages a proposal, --install installs it. No inferred ancestry.
"""
import argparse, csv, hashlib, json, re, shutil, sys, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
RAW = Path(__file__).with_name('selected_surveys_2026')
sys.path.insert(0, str(ROOT))
from dialects import dialect_tag
SOURCES = {'angika': 'regmi2017angika', 'majhi': 'chalise2014majhi', 'koraga':'bhat1971koraga', 'orissa':'census2002orissa'}
EXPECTED = {'angika': 1050, 'majhi': 1050, 'koraga':1192, 'orissa':5065}
def load(name): return json.loads((RAW / name).read_text())
def nfc(s): return unicodedata.normalize('NFC', s)
def space(s): return nfc(' '.join(s.split()))
def writecsv(path, rows):
    with path.open('w', newline='') as f: csv.writer(f).writerows(rows)
def cellkey(k, r): return f"selected-{k}:p{r['printed_page']}:i{r['item']}:c{r['column']}"
def parse(k, r):
    # Source missing-glyph boxes, not guessed lexical readings.
    if '\uf07f' in r['raw_text'] or '(cid:2)' in r['raw_text']: return []
    key = f"{r['item']}:{r['column']}"
    text = load('readings.json').get(k, {}).get(key, r['text'].replace('\n', ''))
    gloss = load('glosses.json')[str(r['item'])]
    tags = []
    for label in ['informal', 'formal', 'inclusive', 'exclusive', 'plural']:
        if f' ({label})' in gloss:
            gloss = gloss.replace(f' ({label})', '')
            tags.append('pl' if label == 'plural' else label)
    return [{'form': space(f), 'gloss': gloss, 'tags': tags,
             'review': ['transcription: published notation retained; ambiguous symbols are not assigned guessed phonemic values']}
            for f in re.split(r'\s*/\s*', text) if f.strip()]
def build(out):
    out.mkdir(parents=True, exist_ok=True)
    for name, digest in load('snapshot.json')['extraction_sha256'].items():
        assert hashlib.sha256((RAW/name).read_bytes()).hexdigest() == digest, name
    meta = load('language-map.json')
    langs = {r['ID']:r for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
    for r in meta['languages']: langs.setdefault(r[0], {'Clade':r[5]})
    existing = {r['ID']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
    dialects = []
    for lects in meta['lects'].values():
        for m in lects:
            dialects.append([m['id'], dialect_tag(m['base'],m['id'],m['name']),m['base'],m['source_label'],m['name'],'','','',langs[m['base']]['Clade'],m['location'],''])
    dt = {r[0]:existing[r[0]]['Tag'] if r[0] in existing else r[1] for r in dialects}
    report = {}
    for k, source in SOURCES.items():
        if k=='koraga':
            sys.path.insert(0,str(RAW));import koraga
            raw=koraga.records();assert len(raw)==EXPECTED[k]
            rows=[];audit=[]
            for r in raw:
                key=f"selected-koraga:p{r['printed_page']}:i{r['item']}"
                readings=[]
                for j,p in enumerate(koraga.parse(r),1):
                    tags=[existing[p['lect']]['Tag']] if p['lect'] else []
                    tags += p['tags']+['uncertain']
                    row=['Koraga','',p['form'],p['gloss'],'','','; '.join(p['notes']),f"{source}[p. {r['printed_page']}, entry {r['item']}]",'',p['comparison'],key+f':v{j}','','','',' '.join(tags)]
                    rows.append(row);readings.append({'row':row,'review':p['review']})
                audit.append({'entry_key':key,'raw':r,'status':'unlinked','readings':readings})
            writecsv(out/f'20260911-selected-{k}.csv',rows)
            (out/(k+'-audit.jsonl')).write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
            report[k]={'raw_entries':len(raw),'installed_rows':len(rows),'excluded_entries':0,'unspecified_lect_rows':sum('dialect:' not in row[14] for row in rows)}
            continue
        raw = load(k+'-cells.json')
        assert len(raw) == EXPECTED[k]
        assert {(r['item'],r['column']) for r in raw} == {(i,c) for i in range(1,1014 if k=='orissa' else 211) for c in range(5)}
        audit, rows = [], []
        for r in raw:
            m = meta['lects'][k][r['column']]; key = cellkey(k,r)
            if k=='orissa':
                sys.path.insert(0,str(RAW));import orissa
                parts=orissa.parse(r)
            else:parts=parse(k,r)
            a = {'entry_key':key,'raw':r,'mapping':m,'status':'unlinked' if parts else 'excluded',
                 'reason':'' if parts else ('Blank or structurally damaged OCR cell; no lexical reading guessed' if k=='orissa' else 'Missing-glyph box visible in the source PDF; no lexical reading guessed'),'readings':[]}
            for j,p in enumerate(parts,1):
                assert p['form'] and not any(c in p['form'] for c in ['(cid:', '\uf07f', '�'])
                tags = [dt[m['id']], *p['tags'], 'uncertain']
                row = [m['base'],'',p['form'],p['gloss'],'','','; '.join(p.get('notes',[])),f"{source}[p. {r['printed_page']}, item {r['item']}, {m['source_label']}]",'','',key+f':v{j}','','','',' '.join(tags)]
                rows.append(row); a['readings'].append({'row':row,'review':p['review']})
            audit.append(a)
        assert len({r[10] for r in rows}) == len(rows)
        writecsv(out/f'20260911-selected-{k}.csv',rows)
        (out/(k+'-audit.jsonl')).write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
        report[k] = {'raw_cells':len(raw),'installed_rows':len(rows),'excluded_cells':sum(not a['readings'] for a in audit)}
    writecsv(out/'languages.csv',meta['languages']); writecsv(out/'dialects.csv',dialects)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
def install(out):
    for k in SOURCES:
        shutil.copyfile(out/f'20260911-selected-{k}.csv',ROOT/f'data/other/forms/20260911-selected-{k}.csv')
        shutil.copyfile(out/(k+'-audit.jsonl'),RAW/(k+'-audit.jsonl'))
    for fn in ['languages.csv','dialects.csv']:
        p=ROOT/'cldf'/fn; old=list(csv.reader(p.open())); ids={r[0] for r in old}
        old.extend(r for r in csv.reader((out/fn).open()) if r[0] not in ids); writecsv(p,old)
    shutil.copyfile(out/'report.json',RAW/'report.json')
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('/tmp/selected-surveys-proposal'));p.add_argument('--install',action='store_true')
    a=p.parse_args(); print(json.dumps(build(a.output),indent=2))
    if a.install: install(a.output)
