"""Reproduce every lexical line of the pinned Gta MLA archive, preserving notation."""
from __future__ import annotations
import argparse,csv,hashlib,json,re,unicodedata
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
FORMS=HERE.parents[1]
SOURCE=HERE/'source-wayback.txt'
CSV=FORMS/'20260925-donegan-stampe-gta-chatterji.csv'
AUDIT=HERE/'audit.jsonl'
SOURCE_SHA256='0dffedf7ee08bb6d0318a1fd4b73a2834442c3297458d0f3ab14850141ef9ea1'
SOURCE_KEY='DSGT'
POS={'N':['noun'],'NB':['noun'],'NK':['noun','kinship'],'V':['verb'],'ADJ':['adj'],'ADV':['adv'],'NUM':['num'],'PP':['postp'],'PRON':['pron'],'PRO':['pron'],'INTERR':['interr'],'DEM':['demonstrative'],'CONJ':['conj'],'INTERJ':['interj'],'NEG PRON':['neg','pron'],'NEG PX(V)':['neg','prefix'],'PX(V)':['prefix'],'PX(PRO)':['prefix'],'PX(DEM)':['prefix'],'NK(VOC)':['noun','kinship','voc'],'VT':['verb','tr'],'VO':['verb'],'NCF':['noun'],'X':[],'D':[]}

def raw_records(data:bytes)->list[dict]:
    if hashlib.sha256(data).hexdigest()!=SOURCE_SHA256: raise ValueError('Snapshot hash changed')
    chunks=data.decode('ascii').split('\f')
    if len(chunks)!=6: raise ValueError('Source sections changed')
    out=[]; unnumbered=0
    for line_number,line in enumerate(chunks[1].splitlines(),1):
        if not line.strip():continue
        m=re.search(r'#(\d+)\.$',line)
        if m: sid=m.group(1)
        else:
            unnumbered+=1;sid=f'unnumbered:{unnumbered}'
        out.append({'source_id':sid,'body_line':line_number,'raw':line.strip()})
    assert len(out)==2066 and unnumbered==3
    assert len({x['source_id'] for x in out})==len(out)
    return out

def clean_gloss(text):
    text=re.sub(r'\s*\(=\s*(?:De|Des)\.<[^>]+>\)?','',text)
    return re.sub(r'\s+',' ',text.replace('^','').replace('_',' ').replace('<','').replace('>','')).strip()

def parse(record):
    sid=record['source_id']; key=f'gta-chatterji-mla2004:{sid}'; raw=record['raw']; text=raw
    repairs=[]
    if sid in {'11031','13272'}: text=text.replace("^beam'.","^beam''.");repairs.append('single closing gloss quote restored; raw preserved')
    if sid=='6172': text=text.replace('{N ``','{N} ``');repairs.append('missing closing POS brace restored; raw preserved')
    if sid in {'unnumbered:2','unnumbered:3'}:
        text=text.replace('<jibon-lEe-ke-ne}','<jibon-lEe-ke-ne>');repairs.append('closing headword } corrected to >; witness absent and not invented')
    # The first label begins the definition. Editorial tail starts only after
    # the last consecutive headword sense, preventing cited causatives becoming senses.
    start=text.find('{')
    header=text[:start] if start>=0 else re.split(r'\s+(?:\?\?|@|#)',text,maxsplit=1)[0]
    headmatches=list(re.finditer(r'<([^<>]+)>(?:\(([CM:]*)\)(CM|M)?)?',header))
    if not headmatches:raise ValueError(('headword boundary',sid,text))
    heads=[m.group(1) for m in headmatches]
    witnesses=[{'label':m.group(2) or '', 'qualifier':m.group(3) or ''} for m in headmatches]
    remainder=re.sub(r'<[^<>]+>(?:\([CM:]*\)(?:CM|M)?)?','',header).strip()
    if remainder.strip(' ,./\\'):raise ValueError(('unparsed header',sid,remainder))
    senses=[];pos=start
    while pos>=0:
        match=re.match(r'\{([^}]+)\}\s*``(.*?)\'\'',text[pos:])
        if not match:raise ValueError(('sense boundary',sid,text[pos:]))
        label,gloss=match.groups()
        if label not in POS:raise ValueError(('unknown grammar',sid,label))
        senses.append((label,gloss));pos+=match.end()
        nextmatch=re.match(r'\.?\s*(?=\{)',text[pos:])
        if nextmatch:pos+=nextmatch.end()
        else:break
    tail=text[pos:] if pos>=0 else text[len(header):]
    if not senses:senses=[('', '')]
    audit={**record,'entry_key':key,'headwords':heads,'witnesses':witnesses,'header_separators':remainder,'repairs':repairs,'source_commentary':tail,'status':'ingested','rows':[],'review_reasons':[]}
    if repairs:audit['review_reasons'].append('transcription: recoverable malformed delimiters')
    if re.search(r'\?\?',raw):audit['review_reasons'].append('source editorial uncertainty')
    if any(re.search(r'[^a-z -]',h) for h in heads):audit['review_reasons'].append('transcription: ASCII notation preserved without unsupported phonemic conversion')
    if any(w['label']!='C' or w['qualifier'] for w in witnesses):audit['review_reasons'].append('provenance: raw witness/editorial labels retained; no dialect inferred')
    rows=[]
    for si,(label,original_gloss) in enumerate(senses,1):
        gloss=clean_gloss(original_gloss)
        flags=list(audit['review_reasons'])
        if not gloss or re.fullmatch(r'[? ]+|G',gloss):
            flags.append('gloss: source gives no lexical definition' if not gloss else 'gloss: source placeholder retained in audit; installed gloss blank');gloss=''
        elif '?' in gloss:flags.append('gloss: source explicitly questions definition')
        tags=list(POS.get(label,[]))
        if re.search(r'\*Loan\b',tail):tags.append('loanword')
        if label=='PX(V)' and gloss=='causative':tags.append('caus')
        if label=='D': flags.append('grammar: unexplained source D label retained without guessing POS')
        if label=='VO': flags.append('grammar: source VO retained; only verb category certain')
        if label=='INTERR':
            tags+=['adv'] if gloss in {'when','where','whence','whither','why, for what reason','how'} else ['pron'] if gloss in {'who','whom','what','which'} else []
        for marker,tag in [('tr.','tr'),('intr.','intr'),('caus.','caus')]:
            if '('+marker+')' in gloss:tags.append(tag);gloss=gloss.replace('('+marker+')','').strip()
        if flags:tags.append('uncertain')
        # Only explicit alternate lists (,,) define variant edges; backslash and
        # slash analyses are preserved as distinct heads without guessed relations.
        variant_list=',,' in remainder and '\\' not in remainder and '/' not in remainder and not re.search(r'\?\?.*phrase',raw,re.I)
        parent=key if si==1 else f'{key}:sense:{si}'
        for hi,head in enumerate(heads,1):
            child=parent if hi==1 else f'{parent}:variant:{hi}'
            citation=f'DSGT[entry {sid}]' if not sid.startswith('unnumbered:') else f'DSGT[body line {record["body_line"]}, {sid}]'
            residual=re.sub(r'#[0-9]+\.$|\?\?#\.$','',tail).strip(' .')
            archive_refs=re.findall(r'@([A-Z][A-Za-z0-9,]*)',residual)
            residual=re.sub(r'@[^ .]*\.?','',residual)
            residual=re.split(r'\?\?',residual,maxsplit=1)[0].strip(' .')
            notes='; '.join(re.findall(r'!([^|*]+)',residual)).strip(' .')
            etymology=re.sub(r'![^|*]+','',residual).strip(' .')
            if archive_refs:citation=citation[:-1]+', archive '+','.join(archive_refs)+']'
            row=['gt','',unicodedata.normalize('NFC',head),gloss,'','',notes,citation,'',etymology,child,parent if hi>1 and variant_list else '','','',' '.join(dict.fromkeys(tags))]
            rows.append(row)
            audit['rows'].append({'entry_key':child,'form':head,'gloss':gloss,'source_gloss':original_gloss,'source_label':label,'tags':row[14],'citation':citation,'review_reasons':flags,'variant_of':row[11],'archive_locators':archive_refs})
    return rows,audit

def prepare():
    rows=[];audit=[]
    for record in raw_records(SOURCE.read_bytes()):
        emitted,decision=parse(record)
        # Fully glossed target-language causative subentries are separate heads,
        # not extra senses of the preceding main headword.
        for ci,match in enumerate(re.finditer(r"Caus\.\s+(<[^>]+>\(C\)\s+\{[^}]+\}\s*``.*?''(?:\.)?)",decision['source_commentary']),1):
            child_record={**record,'source_id':record['source_id']+f':causative:{ci}','raw':match.group(1)}
            child_rows,child_audit=parse(child_record)
            for child,x in zip(child_rows,child_audit['rows']):
                child[7]=f'DSGT[entry {record["source_id"]}, causative subentry {ci}]'
                child[13]=emitted[0][10]
                child[14]+=' caus'
                x.update({'citation':child[7],'derivation_parent':child[13],'tags':child[14]})
            emitted.extend(child_rows)
            decision.setdefault('causative_subentries',[]).append(child_audit)
            decision['rows'].extend(child_audit['rows'])
        rows.extend(emitted);audit.append(decision)
    by_form={}
    by_key={r[10]:r for r in rows}
    for r in rows:by_form.setdefault(r[2],[]).append(r)
    for decision in audit:
        match=re.search(r'\|Caus\. of <([^>]+)>',decision['raw'])
        if not match:continue
        candidates=[r for r in by_form.get(match.group(1),[]) if r[10] not in {x['entry_key'] for x in decision['rows']}]
        decision['derivation']={'source_parent':match.group(1),'candidate_keys':[r[10] for r in candidates],'status':'unique' if len(candidates)==1 else 'ambiguous' if candidates else 'unmatched'}
        if '?' in decision['raw']:
            decision['derivation']['status']='source-uncertain'
            continue
        if len(candidates)==1:
            for x in decision['rows']:
                by_key[x['entry_key']][13]=candidates[0][10]
                x['derivation_parent']=candidates[0][10]
    assert len({r[10] for r in rows})==len(rows)
    return rows,audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--install',action='store_true');args=p.parse_args();rows,audit=prepare()
    if args.install:
        with CSV.open('w',newline='') as f:csv.writer(f).writerows(rows)
        AUDIT.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in audit))
    print(json.dumps({'raw_records':len(audit),'numbered_records':2063,'unnumbered_records':3,'installed_rows':len(rows),'blank_gloss_rows':sum(not r[3] for r in rows),'variant_edges':sum(bool(r[11]) for r in rows),'excluded_records':0}))
if __name__=='__main__':main()
