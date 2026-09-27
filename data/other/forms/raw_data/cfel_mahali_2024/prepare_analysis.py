"""Prepare source analysis proposals only; no installation or database build."""
import collections,json,re,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parent
POS={'noun':'noun','verb':'verb','adjective':'adj','adverb':'adv'}
def prepare():
    rows=[json.loads(s) for s in (ROOT/'positioned-candidates.jsonl').read_text().splitlines()]
    domains={r['entry_key']:r['domain'] for r in json.loads((ROOT/'domains.json').read_text())['assignments']}
    reviews={r['entry_key']:r for r in map(json.loads,(ROOT/'targeted-review.jsonl').read_text().splitlines())}
    native_rows=[json.loads(s) for s in (ROOT/'native-recovery-review.jsonl').read_text().splitlines()]
    native={r['entry_key']:r for r in native_rows}
    semantic={r['entry_key']:r for r in map(json.loads,(ROOT/'semantic-review.jsonl').read_text().splitlines())}
    assert len(native)==len(native_rows), 'duplicate native review key'
    assert set(native)<={r['entry_key'] for r in rows}, 'unknown native review key'
    for review in native.values():
        assert 'visually confirmed' in review['status'] and review['evidence']
        assert review['native'] and not any(c in review['native'] for c in ('\x00','�'))
        assert review['native']==unicodedata.normalize('NFC',review['native'])
    description_reviews={r['entry_key']:r for r in json.loads((ROOT/'description-review-20260926.json').read_text())['records']}
    out=[]
    for raw in rows:
        key=raw['entry_key'];domain=domains[key];issues=list(raw['issues']);notes=[];tags=[]
        grammar=' '.join(raw['raw_grammar'].split());components=[p.strip() for p in grammar.split('+')]
        if len(components)>1:
            tags.append('multiword-expression');notes.append('Source component labels: '+grammar+'.')
        else:
            label=grammar.replace('\x00','').strip().lower()
            assert label in POS,(key,grammar)
            tags.append(POS[label])
            if '\x00' in grammar:issues.append('grammar-font:unmapped-mark-before-readable-POS')
        if domain=='Cardinal Numbers':tags.append('num')
        if domain=='Ordinal Numbers':tags.extend(['num','ord'])
        if domain=='Causative Verb':tags.append('caus')
        if domain=='Compound Verb':tags.append('compound')
        form=unicodedata.normalize('NFC',' '.join(re.sub(r'-[ \t]*\n[ \t]*','-',raw['raw_ipa']).split()))
        if ' ' in form:tags.append('multiword-expression')
        gloss=unicodedata.normalize('NFC',' '.join(re.sub(r'-\s*\n\s*','-',raw['candidate_english']).split()))
        if key=='cfelmahali2024:p306:entry:2':
            gloss='Grain (unit of weight)';notes.append('Source definition specifies a weight equal to one and a half Ratti.')
        if key=='cfelmahali2024:p307:entry:6':
            gloss='Glass (material)'
        if key=='cfelmahali2024:p122:entry:5':
            gloss='Fall (waterfall)';notes.append('Source translations and definition specify falling water, not the verb or season.')
        if key=='cfelmahali2024:p362:entry:6':
            gloss='Fall (autumn)';notes.append('Source translations and definition specify the leaf-fall season.')
        if key=='cfelmahali2024:p135:entry:1':
            gloss='Bachelor (degree holder)';notes.append('Source translations and definition specify a graduate, not marital status.')
        if key=='cfelmahali2024:p294:entry:6':
            gloss='Bachelor (unmarried man)'
        if key=='cfelmahali2024:p153:entry:2':
            gloss='Belly (flower)';notes.append('Printed label Belly refers to a fragrant flower in the source definition; botanical species not inferred.')
        if key=='cfelmahali2024:p155:entry:4':
            notes.append('Printed native উতু and IPA /ut̪u/ retained; publisher online edition instead gives অতঅ /ɔt̪ɔ/. See publisher-edition-review.jsonl.')
        if key=='cfelmahali2024:p164:entry:2':
            notes.append('Printed native নুআঃ and IPA /nuaʔ/ retained; publisher online edition instead gives ঞুআঃ /ɲuaʔ/. See publisher-edition-review.jsonl.')
        if key=='cfelmahali2024:p173:entry:2':
            notes.append('Printed noun স্যাবেল /sæbel/ retained. Publisher Taste search returns a different General verb entry, চাখা /tʃakʰa/, listing স্যাবেল only as an alternate; no noun/verb equivalence inferred.')
        if key=='cfelmahali2024:p220:entry:6':
            issues.append('publisher-IPA:leading-corrupt-combining-mark')
        if key=='cfelmahali2024:p238:entry:2':
            notes.append('Printed native হেঁদে হরম and IPA /hẽd̪e hɔrmɔ/ retained; publisher online edition instead gives হ্যাঁদে হরম /hæ̃d̪e hɔrmɔ/. See publisher-edition-review.jsonl.')
        if key=='cfelmahali2024:p251:entry:5':
            notes.append('Printed House pronunciation /ɔɽaʔ/ retained; publisher online edition gives /ɔraʔ/ for the same native spelling অরাঃ. See publisher-edition-review.jsonl.')
        if key=='cfelmahali2024:p269:entry:5':
            issues.append('publisher-IPA:leading-corrupt-combining-mark')
        if key=='cfelmahali2024:p272:entry:5':
            issues.append('publisher:entry-absent-from-exact-query')
        if key in reviews:issues.extend(reviews[key]['issues'])
        if key in semantic:
            issues.append(semantic[key]['issue']);notes.append(semantic[key]['note']);tags.append('uncertain')
        if any(x.startswith(('transcription:','source-punctuation:combining','grammar-font:')) for x in issues):tags.append('uncertain')
        if key in description_reviews:
            notes.extend(description_reviews[key]['notes'])
            tags.extend(description_reviews[key]['tags'])
        out.append({'entry_key':key,'language':'Mahali','form':form,'gloss':gloss,'source_gloss':raw['candidate_english'],
                    'domain':domain,'tags':list(dict.fromkeys(tags)),'source_component_labels':components,'notes':notes,'issues':issues,
                    'native':native[key]['native'] if key in native else '',
                    'native_status':'accepted; visual review recorded in native-recovery-review.jsonl' if key in native else 'pending recovery/review; raw source retained, no accepted Native yet','raw_native':raw['raw_native'],
                    'status':'source analysis reviewed; native print and seeded output audits recorded; compiled build deferred'})
    return out
if __name__=='__main__':
    rows=prepare();(ROOT/'analysis-proposals.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps({'analyses':len(rows),'uncertain':sum('uncertain' in r['tags'] for r in rows),'component_label_records':sum(len(r['source_component_labels'])>1 for r in rows)},indent=2))
