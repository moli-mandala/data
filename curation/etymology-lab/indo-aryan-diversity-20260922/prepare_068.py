import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments

P = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted' and r['Rank'] == '1'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
rows = [r for r in csv.DictReader(open('cldf/forms.csv')) if r['Source'].startswith('dhakal2011darai[') and r['ID'] not in linked]
rec = lambda r: {k: r[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')}
groups = [
    ('ʦʰə', {'six'}, '12803-3', 'reflex', '*kṣaṭ', 'Dhakal pp. 65 and 68 prints /cʰə/ six. Turner section 3 has Prakrit cha and Nepali cha from *kṣaṭ, contrasting section 1 ṣa/ṣe-type forms and section 2 ṣu-type forms. The house transcription cʰ→ʦʰ is retained; both source examples share this lexical numeral.'),
    ('hẽṭʰ', {'below', 'below, down'}, '248-2', 'reflex', '*adhiṣṭāt', 'Dhakal pp. 53 and 68 explicitly gives /hẽṭʰ/ below/down with final aspirated retroflex. Turner section 2 gives Pali/Pk heṭṭhā, Punjabi heṭh and Bhojpuri heṭhā̃ under *adhiṣṭāt. The source nasalization is preserved, and section 1 adhastāt is not selected.'),
    ('bəlva', {'sand'}, '11580', 'reflex', 'vālukā', 'Dhakal p. 62 prints /bəl.wa/ sand. Turner11580 gives Prakrit vāluā and Nepali bāluwā/baluwā, matching the b-l-w sequence and sense. Darai vowel length is noncontrastive (p. 42); the first vowel and contracted middle syllable remain as printed.'),
    ('ekṭa', {'one'}, 'f_cz67uv3xekayw', 'derived', 'ek + classifier -ṭa', 'Dhakal p. 59 explicitly labels /ek.ṭa/ one-CLF. The earlier linked Darai ek is the base; -ṭa is a source-attested classifier, so this is a synchronic derived link to the same-source lexical numeral rather than a direct reflex of the whole form.'),
    ('ʦarṭa', {'four'}, 'f_iysprsaey63zi', 'derived', 'ʦar + classifier -ṭa', 'Dhakal p. 62 explicitly labels /car.ṭa/ four-CLF. The earlier linked Darai ʦar is the base; the source c→ʦ house transcription and classifier -ṭa are retained. Link the complete form as a synchronic derivative of the same-source numeral.'),
]

accepted = []
review = ['# Darai: six, below, sand and numeral classifiers', '', 'Seven reviewed source records, with ordinary reflexes distinguished from source-labeled classifier derivatives.', '', '| # | Darai form | Proposed etymology | Evidence |', '|---|---|---|---|']
audit = []
source_dir = Path('data/other/forms/raw_data/dhakal_darai_2011/decoded')
for i, (form, glosses, parent, kind, label, evidence) in enumerate(groups, 1):
    hits = [r for r in rows if r['Form'] == form and r['Gloss'] in glosses]
    assert len(hits) == (2 if form in ('ʦʰə', 'hẽṭʰ') else 1), (form, len(hits))
    for r in hits:
        page = int(r['Source'].split('[p. ')[1].split(']')[0])
        source_text = (source_dir / f'p{page:03}.txt').read_text()
        assert source_text, page
        if form == 'ekṭa': assert 'one-CLF' in source_text
        if form == 'ʦarṭa': assert 'four.CLF' in source_text
        audit.append(dict(id=r['ID'], form=form, gloss=r['Gloss'], source=r['Source'], page=page, primaryText=source_text))
        accepted.append(dict(record=rec(r), parent=parent, kind=kind, citation=r['Source'] + (f';CDIAL[{parent}]' if kind == 'reflex' else ''), evidence=evidence))
    review.append(f'| {i} | **{form}** ‘{" / ".join(sorted(glosses))}’ ({len(hits)} records) | **{label}**' + (f', CDIAL **{parent}**' if kind == 'reflex' else f', derived from Darai **{parent}**') + f' | {evidence} |')

holds = []
for form, reason in [
    ('bʰar', 'Darai vowel length is noncontrastive; compare bhara 9393 with bhāra before choosing a historical stem.'),
    ('babu', 'An expressive nursery form shared across languages; CDIAL9209-2 includes both father and affectionate child senses but does not settle independent nursery use versus local contact for Darai.'),
    ('nakno', 'The -no ending needs morphological analysis before linking the whole nose form to *nakka 6909.'),
]:
    for r in rows:
        if r['Form'] == form: holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 7 and sum(x['kind'] == 'reflex' for x in accepted) == 5 and len(holds) == 3
(P / 'pass068-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass068-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Seven affected records and seven assignment rows: five reflexes, two derived classifier forms. Three records held (*bʰar, babu, nakno*) for the stated historical or morphological uncertainties.', 'Primary source: Dhakal 2011 printed pp. 42, 53, 59, 62, 65, 68, checked against the source PDF and decoded text. Discovery: homepage darai-search012–013; full CDIAL articles darai-primary001 and darai-primary009. No source, identity or build changes.', 'Validation: pending.']
(P / 'pass068-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
