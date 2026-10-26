"""Build an explicitly unfinished source-local whole-source proposal; never install."""
from pathlib import Path
import csv, json, hashlib, unicodedata, re
from collections import Counter
from interlinear_metadata import analyze
from apply_legacy_review import apply as apply_legacy
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=PACKAGE.parents[4]
LEGACY=HERE/'legacy-before-whole-recovery.csv'
PROFILE=DATA/'conversion/bailey-kagani-1920.txt'
DIALECT='dialect:Northern%20Hindko:bailey1920-kagani:Kagani%20%28Bailey%201920%29'
SOURCE='bailey1920kagani'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def jsonl(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def nfc(s): return unicodedata.normalize('NFC',s)

def public_note(unit):
    text=unit.get('source_analysis','')
    text=re.sub(r'(?:Original(?:9x|13x| crop)|Root independently|First word narrow12x)[^.]*\. ?', '',text)
    text=text.replace('; both t underdots preserved from original9x crop.','.')
    text=text.replace('Exact do infinitive has plain final a. ', '')
    return text.strip()


def build():
    inputs={'prepare_integration_scaffold.py':sha(Path(__file__)),str(LEGACY.relative_to(DATA)):sha(LEGACY),str(PROFILE.relative_to(DATA)):sha(PROFILE)}
    def load(name,lines=False):
        path=HERE/name;inputs[name]=sha(path)
        return jsonl(path) if lines else json.loads(path.read_text())
    baseline=load('legacy-baseline-20260927.json')
    assert sha(LEGACY)==baseline['csv_sha256'], 'Legacy source changed; reconcile explicitly'
    with LEGACY.open(newline='') as f: rows=list(csv.reader(f))
    assert len(rows)==329 and [r[10] for r in rows]==baseline['entry_keys']
    audit=jsonl(PACKAGE/'audit.jsonl');inputs['../audit.jsonl']=sha(PACKAGE/'audit.jsonl')
    for r in audit:r['integration_disposition']='legacy_glossary_with_reviewed_delta'
    rows,audit=apply_legacy(rows,audit)
    for name in ['apply_legacy_review.py','independent-glossary-metadata-review-20260927.json']:
        inputs[name]=sha(HERE/name)
    inputs['original-scope-review.md']=sha(HERE/'original-scope-review.md')
    for page,reason in [(87,'Geographical language context and explanation of four tones; no separately translated lexical unit in this opening scope. Context labels Gujari/Shina/Pashto do not create target attestations.'),(91,'Separately headed Bahramgala contextual introduction, not Kagani; original scope census records no lexical examples. Later Kagani heading92 governs the recovered paradigms.')]:
        audit.append({'source_unit_key':f'{SOURCE}:whole:p{page}:scope-context','printed_page':page,'status':'source_context_control','entry_keys':[],'reason':reason,'evidence':'original-scope-review.md'})
    newunits=[]
    newunits.extend(load('owner-first-readings.jsonl',True))
    introreview=load('root-intro-independent-review-20260927.json')
    introcorrections={r['source_unit_key']:r['correction'] for r in introreview['per_unit'] if r['correction']}
    reviewed_intro=[]
    for p in [89,90]:
        name=f'intro-p{p}-first-reading-20260927.jsonl'
        assert sha(HERE/name)==introreview['input_sha256'][name]
        for raw in load(name,True):
            unit=dict(raw)
            if raw['source_unit_key'] in introcorrections:
                fix=introcorrections[raw['source_unit_key']]
                unit['first_reading_evidence']=raw
                unit['forms']=fix['forms'];unit['raw_source']=';'.join(fix['forms'])
                if 'scope' in fix:unit['scope']=fix['scope'];unit['source_language']='Panjabi'
                unit['source_analysis']=fix['reason']
                unit['reviewed_correction']=fix
                if 'comparison-scope'==fix['type']:
                    unit['source_analysis']='Explicit Panjabi chăḍḍṇā comparison for lexical leave; target chōhṛnā contrasts intensive chŭṛnā.'
                    unit['owner_confirmation']='Owner13x Rect215,430,296,452 confirms the separate source occurrence; original input remains immutable.'
            unit['review']='Root complete61-unit original review applied; final-output audit pending'
            reviewed_intro.append(unit);newunits.append(unit)
    (HERE/'intro-p89-90-reviewed-20260927.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reviewed_intro))
    for p in [92,93,94,97,98,99,100,101]:newunits.extend(load(f'grammar-p{p}-first-reading-20260927.jsonl',True))
    for filename in ['p95-96-controls.json','p105-controls.json']:
        for r in load(filename)['controls']:
            audit.append({'source_unit_key':r['source_unit_key'],'status':'source_context_control','input':r,'entry_keys':[]})
    inputs['interlinear_metadata.py']=sha(HERE/'interlinear_metadata.py')
    pending=[]
    narrative=HERE/'parable-aligned-reviewed.jsonl'
    if narrative.exists():
        for r in load(narrative.name,True):
            # Owner raw alignment retained: no inferred internal morphology.
            row=dict(r);row.setdefault('forms',[r.get('raw_form','')]);row.setdefault('gloss',r.get('raw_alignment',''))
            row.setdefault('section','parable-aligned');row.setdefault('tags',[])
            newunits.append(row)
    else:
        pending.append('Reviewed parable aligned atoms: awaiting owner derivative; first reading is not emitted')
    parentpath=HERE/'parable-parentheticals-reviewed.json'
    spanpath=HERE/'parable-expression-spans-reviewed.json'
    if narrative.exists() and parentpath.exists() and spanpath.exists():
        primary={r['source_unit_key']:r for r in load(narrative.name,True)}
        parentheses=load(parentpath.name)
        for r in parentheses:
            row=dict(r,forms=[r['raw_form']],gloss=r['raw_alignment'],section='parable-parenthetical',tags=[])
            # Lexically distinct parentheticals must never become variants.
            newunits.append(row)
        for span in load(spanpath.name):
            group=[primary[k] for k in span['primary_token_keys']]
            forms=[];glosses=[]
            for atom in group:
                forms.append(atom['raw_form']);glosses.append(atom['raw_alignment'])
                for extra in parentheses:
                    if extra['anchor_key']==atom['source_unit_key']:
                        forms.append('('+extra['raw_form']+')');glosses.append('('+extra['raw_alignment']+')')
            newunits.append(dict(span,printed_page=span['printed_pages'][0],section='parable-whole-expression',forms=[' '.join(forms)],gloss=' '.join(glosses),source_analysis=span['translation_policy']))
    else:pending.append('Reviewed narrative parenthetical and whole-expression scope pending')
    story=load('p105-story-first-reading.json')
    tokens=[]
    for line in story['lines']:
        for i,(form,gloss) in enumerate(line['pairs'],1):
            key=f'{SOURCE}:whole:p105:story:line{line["line"]}:word{i}'
            item=dict(source_unit_key=key,printed_page=105,section='story-aligned',line=line['line'],token=i,forms=[form],gloss=gloss,tags=[],source_analysis='Printed interlinear English alignment; no inferred word segmentation.')
            if gloss in ['strong','strong-is']:item['source_analysis']+=' Source note says strong means stronger here.'
            tokens.append(item);newunits.append(item)
    for i,span in enumerate(story['whole_expression_spans'],1):
        group=tokens[span['first_token']-1:span['last_token']]
        newunits.append(dict(source_unit_key=f'{SOURCE}:whole:p105:story:expression{i}',printed_page=105,section='story-whole-expression',forms=[' '.join(t['forms'][0] for t in group)],gloss=' '.join(t['gloss'] for t in group),tags=['sentential'],source_analysis=span['analysis'],aligned_unit_keys=[t['source_unit_key'] for t in group]))
    grammatical_evidence=[]
    ergative_forms={}
    for u in newunits:
        if 'erg' in u.get('tags',[]) and ':whole:' not in u['source_unit_key']:
            for form in u.get('forms',[]):ergative_forms.setdefault(nfc(form),[]).append(u['source_unit_key'])
    keys={r[10] for r in rows};unitkeys=set()
    for unit in newunits:
        key=unit['source_unit_key'];assert key not in unitkeys,key;unitkeys.add(key)
        scope=unit.get('scope','target')
        if scope!='target':
            audit.append({'source_unit_key':key,'status':'source_context_control','input':unit,'entry_keys':[]});continue
        forms=unit.get('forms',[])
        if isinstance(forms,str):forms=[forms]
        if not forms:
            audit.append({'source_unit_key':key,'status':'pending_empty_form','input':unit,'entry_keys':[]});pending.append(key);continue
        emitted=[]
        for i,form in enumerate(forms,1):
            ek=key if i==1 else f'{key}:answer{i}'
            assert ek not in keys,ek;keys.add(ek);emitted.append(ek)
            tags=unit.get('tags',[])
            gloss=unit.get('gloss','')
            parsed_interlinear=None
            if unit.get('section') in {'parable-aligned','parable-parenthetical','story-aligned'}:
                parsed_interlinear=analyze(gloss)
                gloss=parsed_interlinear[0]
            if isinstance(tags,str):tags=tags.split()
            if ' ' in form.strip() and not {'multiword-expression','sentential'} & set(tags):
                tags=list(tags)+['multiword-expression']
            note=public_note(unit)
            if parsed_interlinear:
                tags=list(dict.fromkeys(tags+parsed_interlinear[1]))
                raw_analysis=unit.get('raw_alignment',unit.get('gloss','')).lower()
                if raw_analysis.startswith('by-') or raw_analysis=='father-by':
                    matched=ergative_forms.get(nfc(form),[])
                    if matched:tags=list(dict.fromkeys(tags+['erg']))
                    grammatical_evidence.append({'source_unit_key':key,'raw_form':form,'raw_analysis':raw_analysis,'ergative_tag':bool(matched),'source_paradigm_keys':matched,'decision':'Exact form under source agent heading plus agent/by alignment' if matched else 'No ergative tag inferred from by alone; literal alignment preserved'})
                if key=='bailey1920kagani:whole:p101:parable:line1:word6':
                    tags=list(dict.fromkeys(tags+['pl']))
                    grammatical_evidence.append({'source_unit_key':key,'added_tags':['pl'],'basis':'Same original line has dō pŭttăr, two sons, as explicit plural subject; not inferred from English were alone.'})
                if key=='bailey1920kagani:whole:p102:parable:line7:word4':
                    tags=list(dict.fromkeys(tags+['pl']))
                    grammatical_evidence.append({'source_unit_key':key,'added_tags':['pl'],'basis':'Original lines6–7 kĭtne măzūr, how-many labourers, explicitly plural subject; not an ambiguous English you.'})
                if raw_analysis.startswith('was-'):
                    grammatical_evidence.append({'source_unit_key':key,'raw_form':form,'raw_analysis':raw_analysis,'decision':'Retain general source past analysis; no passive-voice or perfect/pluperfect subtype inferred from English was-X.'})
                if gloss!=unit.get('gloss',''):
                    note+=' Printed interlinear analysis: '+unit['gloss']+'.'
                unit=dict(unit,exported_gloss=gloss,exported_tags=tags)
            for label in ['transcription_uncertainty','uncertainty','source_gloss_uncertainty']:
                if unit.get(label):
                    note+=' Source uncertainty: '+json.dumps(unit[label],ensure_ascii=False,sort_keys=True)
                    if 'uncertain' not in tags:tags=list(tags)+['uncertain']
            if unit.get('expansion_basis'):note+=' Printed abbreviation expansion: '+unit['expansion_basis']
            if not unit.get('gloss'):note+=' No independent lexical translation printed; source function/context retained.'
            locator=f'p. {unit["printed_page"]}, {unit.get("section","source")}, unit {key.rsplit(":",1)[-1]}'
            if 'line' in unit:
                kind='story' if unit.get('section','').startswith('story') else 'parable'
                if 'after_token' in unit:
                    locator=f'p. {unit["printed_page"]}, {kind} line {unit["line"]}, parenthetical after word {unit["after_token"]}'
                else:locator=f'p. {unit["printed_page"]}, {kind} line {unit["line"]}, word {unit["token"]}'
            if unit.get('primary_token_keys'):
                def boundary(k):
                    a=re.search(r':p(\d+):parable:line(\d+):word(\d+)$',k)
                    assert a,k
                    return f'p. {a[1]}, line {a[2]}, word {a[3]}'
                locator='parable '+boundary(unit['first_key'])+' through '+boundary(unit['last_key'])
            elif unit.get('aligned_unit_keys'):
                def story_boundary(k):
                    a=re.search(r':p(\d+):story:line(\d+):word(\d+)$',k)
                    assert a,k
                    return f'p. {a[1]}, line {a[2]}, word {a[3]}'
                locator='story '+story_boundary(unit['aligned_unit_keys'][0])+' through '+story_boundary(unit['aligned_unit_keys'][-1])
            rows.append(['Northern Hindko','',nfc(form),gloss,'','',note.strip(),f'{SOURCE}[{locator}]','','',ek,'','','',' '.join([DIALECT]+tags)])
        audit.append({'source_unit_key':key,'status':'proposed_pending_whole_audit','input':unit,'entry_keys':emitted,'alternatives_policy':'Separate printed answers; no variant edge inferred solely from shared cell.'})
    pending.extend(['Final integrated metadata review pending','Complete source metadata and narrative boundary reconciliation pending','Fresh actual-output audit and canonical parser validation pending'])
    # Preserve all established profile rules; only literal missing clusters are added.
    profile=PROFILE.read_text();mapping={}
    for line in profile.splitlines()[1:]:
        if '\t' in line:
            k,v=line.split('\t',1);mapping[k]=v
    def tokenize(s):
        at=0;unknown=[]
        while at<len(s):
            found=next((k for k in sorted(mapping,key=len,reverse=True) if s.startswith(k,at)),None)
            if found:at+=len(found);continue
            end=at+1
            while end<len(s) and unicodedata.combining(s[end]):end+=1
            unknown.append(s[at:end]);at=end
        return unknown
    additions={}
    for r in rows:
        for cluster in tokenize(r[2]):
            additions[cluster]=cluster.lower()
            mapping[cluster]=cluster.lower()
    # Explicit NFD aliases preserve stacked combining marks under the actual tokenizer.
    for key,value in list(mapping.items()):
        decomposed=unicodedata.normalize('NFD',key)
        if decomposed!=key and decomposed not in mapping:
            additions[decomposed]=value;mapping[decomposed]=value
    for row in rows:
        decomposed=unicodedata.normalize('NFD',row[2])
        clusters=[]
        for char in decomposed:
            if unicodedata.combining(char) and clusters:clusters[-1]+=char
            else:clusters.append(char)
        for cluster in clusters:
            if len(cluster)>1:
                value=nfc(cluster).lower().replace('w','v').replace('ṅ','ŋ')
                for alias in [cluster,nfc(cluster)]:
                    if alias not in mapping:additions[alias]=value;mapping[alias]=value
    profile=profile.rstrip('\n')+'\n'+''.join(k+'\t'+v+'\n' for k,v in sorted(additions.items()))
    assert not [x for r in rows for x in tokenize(r[2])]
    (HERE/'interlinear-keyed-grammar-evidence-20260927.json').write_text(json.dumps(grammatical_evidence,ensure_ascii=False,indent=2)+'\n')
    out=HERE/'integration-working';out.mkdir(exist_ok=True)
    with (out/'proposal.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
    (out/'proposal-audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
    (out/'profile-proposal.txt').write_text(profile)
    manifest={'status':'UNFROZEN WORKING SCAFFOLD; NOT ACCEPTED OR INSTALLABLE','rows':len(rows),'legacy_rows_preserved':329,'audit_units':len(audit),'new_input_units':len(newunits),'scope_counts':dict(Counter(u.get('scope','target') for u in newunits)),'pending':pending,'input_sha256':inputs,'outputs':{p.name:sha(p) for p in out.iterdir() if p.name!='manifest.json'},'profile_additions':additions,'profile_policy':'Literal coverage scaffold only; new rules need house-policy/NFC-NFD review. No phonological interpretation of tone strokes.'}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['status','rows','audit_units','new_input_units','pending']},ensure_ascii=False))

if __name__=='__main__':build()
