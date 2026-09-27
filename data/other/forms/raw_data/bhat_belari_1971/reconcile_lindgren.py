"""Reproduce reviewed primary/derived correspondences, without changing source rows."""
import collections,csv,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def comparison_spelling(text, ipa=False):
    if ipa:
        text=text.replace('d͡ʒ','J').replace('t͡ʃ','C').replace('j','y').replace('J','j').replace('C','c')
        text=text.translate(str.maketrans({'ɭ':'ḷ','ɖ':'ḍ','ɳ':'ṇ','ʈ':'ṭ'}))
    return re.sub(r'([bcdfghjklmnpqrstvwxyzḷḍṇṭ]):',lambda m:m[1]*2,text)

# Manual source-scope correspondences reviewed against the five appendix pages.
MANUAL={277:('p123:s12:c1:r5','transcription-disagreement'),
        288:('p123:s10:c1:r7','derived-stem'),303:('p121:s7-root:c1:r1','derived-stem'),
        304:('p123:s10:c1:r6','derived-stem'),307:('p123:s10:c1:r5','derived-stem'),
        309:('p123:s10:c1:r1','derived-stem'),327:('p123:s12:c1:r1','transcription-disagreement'),
        358:('p123:s10:c1:r2','derived-stem'),364:('p123:s10:c1:r3','derived-stem')}
UNLOCATED={301,323,363}

def prepare():
    primary=[json.loads(x) for x in (ROOT/'analysis-proposals.jsonl').read_text().splitlines()]
    by_form=collections.defaultdict(list);by_occurrence=collections.defaultdict(list)
    for row in primary:
        by_form[comparison_spelling(row['form'])].append(row)
        by_occurrence[row['source_occurrence_key']].append(row)
    derived=[r for r in csv.reader((ROOT.parent.parent/'20260911-lindgren.csv').open()) if r[0]=='Belari']
    assert len(derived)==108
    output=[]
    for row in derived:
        number=int(row[10].split(':')[1]);matches=by_form.get(comparison_spelling(row[2],True),[])
        if number==266:matches=[m for m in matches if '2sg' in m['tags'] and 'f' in m['tags']]
        if number==373:matches=[m for m in matches if '2pl' in m['tags']]
        status='shared-primary-source';note='Form and meaning correspondence reviewed; preserve distinct source transcription and cognate analysis, not independent field evidence.'
        if number in MANUAL:
            target,status=MANUAL[number];matches=by_occurrence['bhat1971belari:'+target]
            note='Derived citation/stem form differs from the printed primary root or finite form; do not replace primary response by an inferred stem.' if status=='derived-stem' else 'Same pronoun function, but source glyph reading differs; retain typed transcription uncertainty and both source representations.'
        elif number in UNLOCATED:
            assert not matches;status='not-located-in-appendix';note='Coast/hand/tree entry has no identified target occurrence in this five-page appendix. Retain derived row; do not fabricate primary citation or borrow a Koraga form.'
        elif number in (270,271):
            status='grammatical-disagreement';note='Primary gloss specifies this/that woman or thing; derived inanimate plural analysis is not explicitly attested in the appendix.'
        elif number in (367,368):
            status='unsupported-grammatical-specificity';note='Primary we has no inclusive/exclusive contrast; preserve derived classification as a later analysis, not two independently elicited primary forms.'
        elif number==302:
            status='gloss-ambiguity';note='Primary prints chilly among objects; derived cold may misinterpret chili pepper. Preserve literal primary wording and flag semantic uncertainty.'
        elif number in (317,318):
            status='gloss-generalization';note='Primary give gloss restricts recipient person (I/II versus III); preserve restriction instead of reducing to bare give.'
        if number not in UNLOCATED:assert matches,row[10]
        output.append({'derived_entry_key':row[10],'derived_form':row[2],'derived_gloss':row[3],
                       'primary_candidates':[{'analysis_key':m['entry_key'],'form':m['form'],'gloss':m['gloss'],'tags':m['tags']} for m in matches],
                       'status':status,'decision':note,'action':'review ledger only; derived rows unchanged'})
    return output

if __name__=='__main__':
    rows=prepare();(ROOT/'lindgren-overlap-review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(dict(collections.Counter(r['status'] for r in rows)))
