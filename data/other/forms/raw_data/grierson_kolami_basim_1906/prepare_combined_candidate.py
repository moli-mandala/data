"""Complete source-local review candidate; never writes canonical source assets."""
import collections,copy,csv,hashlib,json,re,shutil,sys,unicodedata as ud
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
sys.path.insert(0,str(ROOT/'data'))
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
import yaml
OUT=HERE/'integration'
GRAMMAR=HERE/'grammar-review/sharp-reviewed-revision2'
SPECIMENS=HERE/'specimens/sharp-reviewed-revision'
SOURCE='grierson1906kolamibasim'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def write(p,rr):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
def guard(r):return {f:r[f] for f in ['form','gloss','language_label','printed_label']}
def clean(value):return value.strip(' .;,?!‘’“”\"').lower()
NUM={'one','two'}
NOUN={'man','sons','son','father','younger-brother','property','country','famine','difficulty','swine','husks','belly','bread','servants','sin','god','neck','kiss','robe','hand','ring','foot','shoes','music','dance','house','feast','years','service','order','friends','kid','child','money','village','sense','shame','body','people','anger','goat','share','time','pity','servant'}
PAST={'said','gave','went','spent','fell','became','lived','sent','ate','asked','came','heard','made','saw','took','began','called','arose','put','met','lost','squandered','entreated'}
VERBS={'die','fill','give','keep','make','do','see','going','arising','running','seeing','bringing','collecting','dividing','coming','eating','dancing'}
PERSON={'i':['pron','personal','1sg','first-person'],'me':['pron','personal','1sg','first-person','obj'],
    'my':['pron','poss','1sg','first-person'],'he':['pron','personal','3sg','third-person','m'],
    'him':['pron','personal','3sg','third-person','m','obj'],'his':['pron','poss','3sg','third-person','m'],
    'it':['pron','3sg','third-person','n'],'thou':['pron','personal','2sg','second-person'],
    'thy':['pron','poss','2sg','second-person'],'thine':['pron','poss','2sg','second-person'],
    'thee':['pron','personal','2sg','second-person','obj'],'they':['pron','personal','3pl','third-person'],
    'them':['pron','personal','3pl','third-person','obj'],'your':['pron','poss','second-person']}
CASE={'to':'dat','from':'abl','of':'gen','in':'loc','on':'loc','near':'loc','with':'comitative','for':'dat'}
def atomic_metadata(a):
    """Printed English scopes only; no inferred target morphemes or -ing subtype."""
    raw=a['raw_interlinear_gloss'];g=clean(raw).replace('(-','-').replace(')','')
    tags=[];notes=[];base=g
    # English article/personal possessive in a gloss doesn't define target morphology.
    for prefix in ['the-','my-']:
        if base.startswith(prefix):base=base[len(prefix):]
    if base.endswith('’s'):base=base[:-2];tags.append('gen')
    elif base.endswith("'s"):base=base[:-2];tags.append('gen')
    for ending,case in CASE.items():
        if base.endswith('-'+ending):base=base[:-(len(ending)+1)];tags.append(case);break
    if base in PERSON:tags.extend(PERSON[base])
    elif base in NUM:tags.append('num')
    elif base in NOUN:
        tags.append('noun')
        if base in ['sons','servants','husks','shoes','friends','years','harlots','people']:tags.append('pl')
        if base in ['father','son','sons','younger-brother']:tags.append('kinship')
    elif base in PAST:tags.extend(['verb','pret'])
    elif base in VERBS:tags.append('verb')
    elif base in ['was','were']:tags.extend(['verb','copula','pret'])
    elif base in ['is','art']:tags.extend(['verb','copula','pres'])
    elif base in ['this','that']:tags.extend(['pron','demonstrative'])
    elif base in ['what']:tags.extend(['pron','interr'])
    elif base.startswith('will-'):
        tags.extend(['verb','fut']);notes.append('Printed aligned English: '+raw)
    elif base.startswith('to-') and base[3:] in VERBS|PAST:
        tags.extend(['verb','inf']);notes.append('Printed aligned English: '+raw)
    elif base.endswith('-not'):
        tags.extend(['verb','neg']);notes.append('Printed aligned English: '+raw)
    elif base.startswith('having-'):
        tags.append('verb');notes.append('Printed aligned English: '+raw+'; nonfinite subtype not inferred from the English alone.')
    if not tags:notes.append('Printed aligned English retained without an inferred target morphology: '+raw)
    return list(dict.fromkeys(tags)),notes
def clusters(value):
    rr=[]
    for c in value:
        if ud.combining(c) and rr:rr[-1]+=c
        else:rr.append(c)
    return rr
def main():
    OUT.mkdir(exist_ok=True)
    archive=OUT/'review-scaffold-r1'
    if (OUT/'status.json').exists() and not archive.exists():
        archive.mkdir()
        for p in OUT.iterdir():
            if p.is_file() and p.suffix in ['.csv','.json','.jsonl','.txt']:shutil.copyfile(p,archive/p.name)
    paths=[GRAMMAR/n for n in ['reviewed-child-witnesses.jsonl','grammar-tag-proposal.jsonl','grammar-graph-proposal.jsonl']]
    paths += [SPECIMENS/n for n in ['atomic-emission-proposal.jsonl','expression-emission-proposal.jsonl','physical-atom-emission-audit.jsonl','status.json']]
    paths += [HERE/'scope/physical-unit-census.jsonl',HERE/'outside-review/outside-census-review.json',HERE/'metadata-profile-proposal/dialects-proposed.csv']
    pins={str(p.relative_to(HERE)):sha(p) for p in paths}
    grammar=rows(paths[0]);tagby={r['witness_key']:r for r in rows(paths[1])};graphby={r['witness_key']:r for r in rows(paths[2])}
    assert len(grammar)==224 and len(tagby)==224 and len(graphby)==213
    dialects={r['Language_ID']:r['Tag'] for r in csv.DictReader(paths[-1].open())}
    languages={r['ID'] for r in csv.DictReader((ROOT/'data/cldf/languages.csv').open())}
    route={'Basim / so-called Bhili':'Naikri','Kanarese':'Kannada',**{x:x for x in ['Kolami','Tamil','Tulu','Telugu','Toda','Kui']}}
    candidate=[];audit=[];metadata=[];mapping={};counts=collections.Counter()
    kui=ROOT/'data/data/other/forms/20260927-grierson-kui.csv'
    with kui.open() as f:oldkui={r[10]:r for r in csv.reader(f)}
    reused='grierson1906kui:p566:outside-comparand:06'
    assert oldkui[reused][0]=='Kolami' and oldkui[reused][2]=='amd' and oldkui[reused][3]=='he'
    def emit(key,language,form,gloss,tags,notes,page,kind,source_record,variant='',parents=None,uncertainties=None):
        assert language in languages and set(tags)-set(dialects.values())<=GRAMMATICAL_TAGS|GENDER_TAGS,(key,tags)
        if uncertainties and 'uncertain' not in tags:tags.append('uncertain')
        row=[language,'',form,gloss,'','',' '.join(notes),f'{SOURCE}[p. {page}]','','',key,variant,'','|'.join(parents or []),' '.join(dict.fromkeys(tags))]
        assert len(row)==15
        candidate.append(row);counts[kind]+=1;mapping[key]=key
        audit.append({'source_unit_key':key,'kind':kind,'disposition':'review_candidate_emission','entry_keys':[key],
                      'source_record':source_record,'typed_uncertainties':uncertainties or [],'actual_row':row,'accepted_for_installation':False})
    for r in grammar:
        key=r['source_unit_key'];language=route[r['language_label']]
        t=tagby[key];assert t['exact_before_guard']==guard(r)
        if r['proposed_disposition']!='eligible_review_candidate':
            audit.append({'source_unit_key':key,'kind':'grammar_child','entry_keys':[],'disposition':r['proposed_disposition'],'source_record':r,'accepted_for_installation':False});continue
        if r['printed_page']==566 and language=='Kolami' and r['form']=='amd':
            mapping[key]=reused
            audit.append({'source_unit_key':key,'kind':'grammar_child','entry_keys':[reused],'disposition':'same_occurrence_existing_key_reuse',
                          'canonical_path':str(kui.relative_to(ROOT)),'canonical_row':oldkui[reused],'source_record':r,'accepted_for_installation':False});continue
        g=graphby[key];assert g['exact_before_guard']==guard(r)
        tags=t['proposed_tags'][:]
        if g['variant_of_witness_key']:tags.append('alternate')
        if language=='Naikri':tags.append(dialects[language])
        notes=[r['qualification']] if r['qualification'] else []
        if g['variant_of_witness_key'] or g['ordered_parents']:notes += [g['source_claim'],g['scope_caveat']]
        emit(key,language,r['form'],r['gloss'],tags,notes,r['printed_page'],'grammar',r,
             g['variant_of_witness_key'],g['ordered_parents'],r['typed_uncertainties']+t['typed_metadata_uncertainties'])
    uncertain=json.loads((SPECIMENS/'status.json').read_text())['typed_glyph_uncertainties']
    for a in rows(SPECIMENS/'atomic-emission-proposal.jsonl'):
        tags,notes=atomic_metadata(a);language='Kolami' if a['printed_page'] in [564,565] else 'Naikri'
        tags.append(dialects[language]);uu=[{'type':'source-glyph','detail':uncertain[k]} for k in a['physical_source_keys'] if k in uncertain]
        metadata.append({'source_unit_key':a['source_unit_key'],'exact_form_guard':a['raw_source_form'],'exact_raw_gloss_guard':a['raw_interlinear_gloss'],
                         'proposed_tags':tags,'source_scope':'Printed aligned English and source-bounded grammatical use only; no unprinted morphological segmentation.',
                         'audit_only_analysis':notes,
                         'review_status':'Owner proposal; independent structured-metadata review pending'})
        emit(a['source_unit_key'],language,a['raw_source_form'],a['raw_interlinear_gloss'],tags,[],a['printed_page'],'atomic',a,uncertainties=uu)
    for e in rows(SPECIMENS/'expression-emission-proposal.jsonl'):
        language='Kolami' if e['specimen']==75 else 'Naikri'
        pp='–'.join(str(p) for p in e['printed_pages'])
        uu=[{'type':'source-glyph','detail':uncertain[k]} for k in e['physical_source_keys'] if k in uncertain]
        emit(e['source_unit_key'],language,e['raw_source_form'],e['raw_aligned_interlinear_gloss'],['sentential',dialects[language]],[],pp,'expression',e,uncertainties=uu)
    outside=json.loads((HERE/'outside-review/outside-census-review.json').read_text())
    for o in outside['physical_lexical_witnesses_outside_chapter']:
        if o['disposition']=='exact_existing_occurrence_reuse':
            audit.append({'source_unit_key':o['key'],'kind':'outside','disposition':'same_occurrence_existing_keys_reuse',
                          'entry_keys':[o['existing_kolami_key'],o['existing_naiki_key']],'source_record':o,'accepted_for_installation':False});continue
        language='Naikri' if 'Bhili' in o['source_label'] else 'Kolami'
        tags=['negator'];uu=[]
        if o.get('accepted_route'):tags=o['accepted_route']['tags']+tags;uu=[o['accepted_route']['typed_uncertainty']]
        emit(o['key'],language,o['form_candidate'],o['gloss'],tags,[o.get('source_analysis',o.get('analysis',''))],o['printed_page'],'outside',o,uncertainties=uu)
    # Every physical structural unit and atom remains audited, even where no row is a lemma.
    for p in rows(HERE/'scope/physical-unit-census.jsonl'):
        audit.append({'source_unit_key':p['source_unit_key'],'kind':'physical_parent','disposition':'structural_source_accounting',
                      'entry_keys':[],'source_record':p,'accepted_for_installation':False})
    for p in rows(SPECIMENS/'physical-atom-emission-audit.jsonl'):
        audit.append({'source_unit_key':p['source_unit_key']+':physical-audit','physical_source_key':p['source_unit_key'],'kind':'physical_atom_accounting',
                      'disposition':p['disposition'],'entry_keys':p['proposed_entry_keys'],'source_record':p,'accepted_for_installation':False})
    assert len(candidate)==827 and len({r[10] for r in candidate})==827,counts
    for r in candidate:
        r[11]=mapping.get(r[11],r[11])
        r[13]='|'.join(mapping.get(k,k) for k in r[13].split('|') if k)
    allkeys={r[10] for r in candidate}|set(oldkui)
    assert all((not r[11] or r[11] in allkeys) and all(k in allkeys for k in r[13].split('|') if k) for r in candidate)
    with (OUT/'combined-review-candidate.csv').open('w',newline='') as f:csv.writer(f).writerows(candidate)
    write(OUT/'combined-review-audit.jsonl',audit);write(OUT/'specimen-metadata-proposal.jsonl',metadata)
    rules={' ':'#'};inventory=collections.Counter()
    for r in candidate:
        for original in [ud.normalize('NFC',r[2]),ud.normalize('NFD',r[2])]:
            for c in clusters(original):
                inventory[c]+=1;rules[c]='#' if c==' ' else ud.normalize('NFC',c.lower()).replace('ṅ','ŋ')
    with (OUT/'grierson-kolami-basim-1906.txt').open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['Grapheme','IPA']);w.writerows(sorted(rules.items()))
    (OUT/'unicode-inventory.json').write_text(json.dumps({'clusters':[{'literal':c,'count':n,'codepoints':[f'U+{ord(x):04X}' for x in c]} for c,n in sorted(inventory.items())],
        'policy':'Lowercase and ṅ→ŋ display only; literalOriginal retained; ñ/diaeresis/quantity/source punctuation distinct.'},ensure_ascii=False,indent=2)+'\n')
    settings=yaml.safe_load((HERE/'metadata-profile-proposal/settings-proposed.yaml').read_text())
    settings['file']='combined-review-candidate.csv'
    settings['sources'][SOURCE]['forms']={'split_alternates':False}
    settings['sources'][SOURCE]['gloss']={'grammar_tags':False}
    (OUT/'combined-review-candidate.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
    outputs=[OUT/n for n in ['combined-review-candidate.csv','combined-review-audit.jsonl','specimen-metadata-proposal.jsonl','grierson-kolami-basim-1906.txt','unicode-inventory.json','combined-review-candidate.yaml']]
    status={'status':'Complete source-local integration review candidate; no installation approval or freeze',
      'input_pins':pins,'counts':{'emitted_rows':len(candidate),'sections':dict(counts),'grammar_children':224,'physical_atoms':561,'atomic_emissions':560,'source_bounded_expressions':51,
                  'structural_parents':169,'outside_physical_tokens':5,'outside_language_attributions':6,'reused_canonical_rows':3,'audit_records':len(audit)},
      'outputs':{p.name:sha(p) for p in outputs},'canonical_changes':False,
      'pending':['Independent specimen structured-metadata review','Current grammar metadata acceptance','Actual full candidate parser/profile/registry/ref validation','Fresh final actual-output original audit','Guarded installer and source-stage approval','Full compiled/browser gates deferred']}
    (OUT/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(status['counts']))
if __name__=='__main__':main()
