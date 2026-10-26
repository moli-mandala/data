"""Prepare auditable sense candidates, not installed forms or a database."""
import argparse
import collections
import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAPSHOT_SHA = '9b377c42300e406c10d1571dab998a7ea7f924be8cf912e2b98ecaf0a2bae666'
GRAMMAR = {'n.': ['noun'], 'vb.': ['verb'], 'tr.vb.': ['verb','tr'], 'tr. vb.': ['verb','tr'], 'tr.vb': ['verb','tr'], 'itr.vb.': ['verb','intr'], 'itr. vb.': ['verb','intr'], 'adj.':['adj'], 'adjective':['adj'], 'adv.':['adv'], 'causative':['caus'], 'imperative form':['impv'], '1st singular subject':['1sg']}
AMBIGUOUS_HEADS = {'lV4SFRDZBI0R4KykWpwJ':'Full parenthetical alkusi matches Hindi gloss; do not emit it as a Birhor optional segment or variant.', 'boBzg1Wpance1ExpE3p1':'Comma-separated headword conflicts with unpunctuated IPA; notes distinguish bulbils haser kapu and tubers bawla. Requires lexical-scope decision.'}
OPTIONAL = {'GdrTAsGFBt3EO7s8aZij':'i', 'ST6m46b9cgDXJrKKkdkK':'a', 'oErN3DLXnSAfDwnsdmmU':'m', 'tQe959B9xFG0IbbVsFMQ':'a'}

def senses(gloss):
    markers=list(re.finditer(r'(?<!\S)(\d+)\.\s+',gloss))
    if not markers or markers[0].start()!=0:
        return [(1,gloss)]
    nums=[int(m[1]) for m in markers]
    if nums != list(range(1,len(nums)+1)):
        raise ValueError(f'Nonsequential numbered gloss: {gloss}')
    return [(int(m[1]),gloss[m.end():markers[i+1].start() if i+1<len(markers) else len(gloss)].strip()) for i,m in enumerate(markers)]

def clean_grammar(gloss):
    tags=[]
    def marker(match):
        label=match[1]
        if label not in GRAMMAR:return match[0]
        tags.extend(GRAMMAR[label]);return ''
    text=re.sub(r'\(([^()]*)\)',marker,gloss)
    text=html.unescape(re.sub(r'</?i>', '', text))
    return re.sub(r'\s+',' ',text).strip(),list(dict.fromkeys(tags))

def prepare():
    grammar_decisions=json.loads((ROOT/'grammar-decisions.json').read_text())
    raw=ROOT/'lexical-snapshot.jsonl'
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==SNAPSHOT_SHA
    candidates=[];audit=[]
    for line in raw.open():
        record=json.loads(line);e=record['entry'];lex=json.loads(e['lexeme']);issues=[]
        if lex['default']!=e['phonetic']:issues.append('transcription:headword-IPA-disagreement')
        if e['id'] in AMBIGUOUS_HEADS:issues.append('lexical-scope:'+AMBIGUOUS_HEADS[e['id']])
        if e['id'] in OPTIONAL:issues.append('transcription:optional-segment-expansion-pending')
        if any(not json.loads(s['glosses']).get('en','').strip() for s in record['senses']):issues.append('gloss:missing-English-gloss')
        entry={'entry_id':e['id'],'issues':issues,'candidate_keys':[],'notes_preserved_in_snapshot':bool(e['notes']),'morphology':e['morphology'],'interlinearization':e['interlinearization'],'source_slugs':json.loads(e['sources'] or '[]')}
        for sense in record['senses']:
            glosses=json.loads(sense['glosses']);split=senses(glosses['en'])
            for ordinal,gloss in split:
                cleaned,tags=clean_grammar(gloss)
                if e['id'] in grammar_decisions:
                    decision=grammar_decisions[e['id']]
                    assert gloss==decision['expected_gloss']
                    cleaned,tags=decision['gloss'],decision['tags']
                key=f"living-birhor2026:{e['id']}:sense:{sense['id']}:meaning:{ordinal}"
                candidate={'entry_key':key,'entry_id':e['id'],'sense_id':sense['id'],'meaning_number':ordinal,'source_headword':lex['default'],'source_ipa':e['phonetic'],'native':lex['devanagari-script'],'raw_english_gloss':gloss,'gloss':cleaned,'tags':tags,'review':issues,'status':'candidate-not-installed'}
                candidates.append(candidate);entry['candidate_keys'].append(key)
        audit.append(entry)
    return candidates,audit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    candidates,audit=prepare();a.output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('sense-candidates',candidates),('sense-audit',audit)]:
        (a.output/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    summary={'entries':len(audit),'meaning_candidates':len(candidates),'entries_with_multiple_meanings':sum(len(r['candidate_keys'])>1 for r in audit),'review_entries':sum(bool(r['issues']) for r in audit),'tag_counts':dict(collections.Counter(t for r in candidates for t in r['tags'])),'installed_rows':0,'remaining':'Review grammar scope, source references, annotations, optional forms and transcription disagreements; rights basis and source installation still unresolved.'}
    (a.output/'sense-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary))
