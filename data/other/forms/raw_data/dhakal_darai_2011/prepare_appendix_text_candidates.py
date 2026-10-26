"""Unreviewed physical text candidates; typography assists but never approves extraction."""
import json,re,unicodedata,collections
from pathlib import Path
P=Path(__file__).resolve().parent
IDENTIFIER=re.compile(r'[A-Za-z]+(?:\.[A-Za-z]+)+\.\d+\.?')
# Original-page review: this target predicate is printed in the small upright
# gloss font, but has its own following analysis/gloss rows. Typography alone
# therefore cannot classify it as English or attach it to the previous word.
ORIGINAL_REVIEWED_CATEGORY={(398,79):'surface_fragments',
                            (400,52):'surface_fragments'}

def build():
    units=[];current=None;controls=[]
    for s in (P/'whole-native-pages.jsonl').open():
        page=json.loads(s);n=page.get('printed_page')
        if not isinstance(n,int) or not 388<=n<=450:continue
        for line in page['lines']:
            text=''.join(sp['decoded'] for sp in line['spans']).strip()
            if not text or text==str(n):continue
            if IDENTIFIER.fullmatch(text):
                current=dict(source_unit_key=f'dhakal2011:appendix5:{text.rstrip(".")}',text_id=text.rstrip('.'),
                             printed_page=n,pdf_page=n+22,kind='whole-expression',source_lines=[],
                             surface_fragments=[],interlinear_fragments=[],gloss_fragments=[],translation_fragments=[],
                             unclassified_fragments=[],status='Unreviewed typographic candidate; every page requires original review')
                units.append(current)
                continue
            evidence=dict(printed_page=n,**line)
            if any('Bold' in sp['font'] for sp in line['spans']):
                controls.append(evidence);continue
            if current is None:controls.append(evidence);continue
            current['source_lines'].append(evidence)
            meaningful=[sp for sp in line['spans'] if sp['decoded'].strip()]
            native=[sp for sp in meaningful if 'Doulos' in sp['font'] or 'IPA93' in sp['font']]
            if native:
                # Simulated italic surface text has a sheared left bbox relative to origin.
                # Same-font interlinear text lacks that shear. Raw boxes remain auditable.
                italic=any(sp['origin'][0]-sp['bbox'][0]>.7 for sp in native)
                category='surface_fragments' if italic else 'interlinear_fragments'
            elif meaningful and all('TimesNewRoman' in sp['font'] for sp in meaningful):
                category='translation_fragments' if max(sp['size'] for sp in meaningful)>10 else 'gloss_fragments'
            else:category='unclassified_fragments'
            category=ORIGINAL_REVIEWED_CATEGORY.get((n,line['line']),category)
            current[category].append(evidence)
    for r in units:
        def joined(category):return unicodedata.normalize('NFC',re.sub(r'\s+',' ',' '.join(''.join(sp['decoded'] for sp in l['spans']) for l in r[category])).strip())
        r['form']=joined('surface_fragments');r['gloss']=joined('translation_fragments')
        r['review_warnings']=[]
        for category in ['surface_fragments','interlinear_fragments','gloss_fragments','translation_fragments']:
            if not r[category]:r['review_warnings'].append('Missing '+category)
        if r['unclassified_fragments']:r['review_warnings'].append('Unclassified source lines')
    assert len(units)==337 and len({r['source_unit_key'] for r in units})==337
    return units,controls

if __name__=='__main__':
    units,controls=build()
    (P/'appendix5-text-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in units))
    (P/'appendix5-text-context-lines.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in controls))
    print(len(units),'unreviewed utterance candidates;',sum(bool(r['review_warnings']) for r in units),'with explicit structural warnings')
    print(dict(collections.Counter(r['text_id'].rsplit('.',1)[0] for r in units)))
