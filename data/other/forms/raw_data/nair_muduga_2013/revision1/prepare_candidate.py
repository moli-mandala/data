"""Complete chapter source-local proposal; no canonical writer or IPA inference."""
import csv
import hashlib
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
LECT='dialect:Muduga:nair2013-muduga:Muduga%20%28Attapady%29'

def tags(unit,w):
    analysis=w['source_analysis']; section=unit['source_unit_key']; ts=[]
    # A whole personal-suffix example is a verb form, not an isolated affix.
    if not w['printed_gloss'] and any(s in analysis for s in ('suffix','marker')):
        ts += ['affix','suffix']
    for needle,tag in [('exclusive','exclusive'),('inclusive','inclusive'),('causative','caus'),
                       ('transitive','tr'),('negative','neg'),('imperative','impv'),('purposive','purposive'),
                       ('optative','opt'),('conditional','conditional'),('accusative','acc'),('sociative','comitative'),
                       ('genitive','gen'),('locative','loc'),('oblique','obl'),('demonstrative','demonstrative'),
                       ('proximal','prox'),('distal','dist'),('masculine','m'),('feminine','f'),('neuter','n'),
                       ('future','fut'),('past','pret'),('present','pres'),('potential','possibility'),('obligatory','obligation')]:
        if needle in analysis:ts.append(tag)
    if 'singular' in analysis:ts.append('sg')
    if 'plural' in analysis:ts.append('pl')
    for person,label in [('first','first-person'),('second','second-person'),('third','third-person')]:
        if person in analysis:
            ts.append(label)
            if 'singular' in analysis:ts.append({'first':'1sg','second':'2sg','third':'3sg'}[person])
            if 'plural' in analysis:ts.append({'first':'1pl','second':'2pl','third':'3pl'}[person])
    if any(x in section for x in ('personal-pronouns','accusative-examples','demonstratives')) or 'pronoun' in analysis:
        ts.append('pron')
    if ':demonstratives' in section:ts.append('demonstrative')
    if 'plural brace' in analysis:ts+=['pron','demonstrative','pl']
    if ':personal-suffixes' in section:ts.append('personal')
    if 'oblique base' in analysis:ts.append('stem')
    if ':personal-suffixes' in section:
        index=int(w['witness_key'].rsplit(':w',1)[1])
        if index in (2,4):ts=['affix','suffix','pl']
        if index==9:ts+=['third-person','3sg','f']
    if any(x in section for x in ('transitive-analysis','causative-examples','present-examples','future-examples','past-examples','purposive-examples','imperative-examples','potential-examples','optative-examples','obligatory-examples','conditional-examples','personal-examples')):
        ts.append('verb')
    if ':transitive-analysis' in section:ts+=['tr']
    if ':personal-examples:' in section:
        index=int(section.rsplit(':',1)[1])
        scoped=[['1sg'],['1sg'],['1pl','exclusive'],['1pl','exclusive'],
                ['1pl','inclusive'],['1pl','inclusive'],['2sg'],['2sg'],['2pl'],['2pl'],
                ['3sg','m'],['3sg','m'],['3sg','f'],['3sg','f'],['3pl'],['3pl'],['3sg','n'],
                ['1pl','exclusive'],['1pl','exclusive']]
        ts+=scoped[index-1]
        if index==19:ts.append('uncertain')
    if 'genitive example' in analysis and w['printed_gloss'] in ('my','your','of her','of us'):ts.append('pron')
    if w['printed_gloss'] in ('father','elder brother'):ts.append('kinship')
    if w['printed_gloss']=='twenty':ts.append('num')
    if w['language_scope']=='Muduga':ts.append(LECT)
    return list(dict.fromkeys(ts))

def main():
    units=[json.loads(x) for x in (HERE/'reviewed-first-reading.jsonl').read_text().splitlines()]
    rows=[];audit=[]
    for u in units:
        accounted=[]
        for w in u['witnesses']:
            key=w['witness_key'];f=w['form'];analysis=w['source_analysis'];lang=w['language_scope']
            held=u['disposition']=='documentary' or lang not in ('Muduga','Malayalam') or f in ('ø','–ø-')
            reason='source documentary/context witness' if u['disposition']=='documentary' else 'isolated zero exponent, not a lexical head' if f in ('ø','–ø-') else ''
            if not held:
                # Labels in a paradigm are grammatical analysis, not lexical translations.
                grammar_cell=any(x in u['source_unit_key'] for x in ('accusative-examples','personal-pronouns','demonstratives','reported-feature')) or not w['printed_gloss']
                gloss='' if grammar_cell else re.sub(r'\((?:You\s+)?(?:Sg|Pl|Excl|Incl|Exl)\.?\)', '', w['printed_gloss'], flags=re.I).strip()
                note='Nair source analysis: '+analysis+'.'
                claim=u['source_claim']
                if ':personal-examples:' in u['source_unit_key']:
                    claim='Example from the personal-suffix table.'
                if ':present-examples:' in u['source_unit_key'] and 'Zero is printed' in claim:
                    claim='Zero is printed inside this whole form.'
                if claim:note+=' '+claim
                scholarly_note=u['note']
                if ':reported-feature-' in u['source_unit_key']:scholarly_note=''
                if ':personal-pronouns' in u['source_unit_key']:scholarly_note='The source prints capital En- and differently dash-marked oblique bases.'
                if ':demonstratives' in u['source_unit_key']:scholarly_note='Plural column labelled (Obl.); ivaru/avaru brace forms span the masculine and feminine rows.'
                if ':transitive-analysis:' in u['source_unit_key']:scholarly_note='Nair joins the decomposition and surface form by an arrow.'
                if scholarly_note:note+=' '+scholarly_note
                if grammar_cell and w['printed_gloss']:note+=' Printed grammatical label: '+w['printed_gloss']+'.'
                if lang=='Malayalam':note='Explicit Malayalam comparator. '+note
                if ':personal-examples:19' in key:note+=' Source final Exl. label conflicts with the preceding first-person inclusive-shaped forms; preserve the attestation and withhold inferred morphology.'
                row=[lang,'',f,gloss,'','',note,f'nair2013muduga[p. {u["printed_page"]}, {u["source_unit_key"].split(":",2)[2]}]','','',key,'','','',' '.join(tags(u,w))]
                rows.append(row)
            accounted.append(dict(physical_key=key,raw=w,disposition='audit_only' if held else 'emitted',hold_reason=reason,entry_keys=[] if held else [key]))
        audit.append(dict(source_unit_key=u['source_unit_key'],printed_page=u['printed_page'],raw_source=u,physical_witnesses=accounted,entry_keys=[k for a in accounted for k in a['entry_keys']]))
    assert len({r[10] for r in rows})==len(rows)
    with (HERE/'proposed.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
    (HERE/'proposal-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
    report={'status':'complete chapter unfrozen owner proposal; metadata/notation review and independent acceptance pending','groups':len(units),'witness_records':sum(len(u['witnesses']) for u in units),'rows':len(rows),'target_rows':sum(r[0]=='Muduga' for r in rows),'comparison_rows':sum(r[0]=='Malayalam' for r in rows),'audit_only':sum(not w['entry_keys'] for a in audit for w in a['physical_witnesses']),'graph_edges':0,'canonical_changes':False}
    for n in ('first-reading.jsonl','reviewed-first-reading.jsonl','proposed.csv','proposal-audit.jsonl'):
        report[n+'_sha256']=hashlib.sha256((HERE/n).read_bytes()).hexdigest()
    (HERE/'proposal-status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
