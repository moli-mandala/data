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
    ('ʦirike', 'to split', '4844', '*cīrayati',
     'Dhakal p. 54 prints /cirike/ “to split,” contrasting /cʰirike/ “to enter.” Turner4844 lists Nepali cirnu, Bengali cirā, Oriya ciribā and Maithili cīrab “to split”; Darai -ike is the source-described ordinary infinitive.'),
    ('ʣuʣʰike', 'to fight', '10502', 'yúdhyatē',
     'Dhakal p. 54 prints /dzudzʰike/ “to fight.” Turner10502 gives Pali yujjhati, Bengali jujhā, Oriya jujhibā and Bhojpuri jūjhal. The Darai infixal affricate and aspirate match the regional jujh- stem; -ike is the ordinary infinitive.'),
    ('ʣuʣʰlə', 'fought', '10502', 'yúdhyatē',
     'Dhakal p. 68 prints /dzudzʰlə/ “fought”; p. 75 identifies -lə as the past suffix. It is an inflection of the same jujh- stem that Turner10502 compares with eastern jujh-/jūjh- forms, so it is linked directly as a reflex, not a new derivation.'),
    ('rokike', 'to stop', '10827', '*rōkk-',
     'Dhakal p. 56 contrasts /rokike/ “to stop” with /lokike/ “to hide.” Turner10827 lists Nepali roknu, Bengali rokā, Oriya rokibā and Maithili rokab; Darai -ike is its ordinary infinitive suffix. Turner leaves the remote relation to rudh uncertain, which this link does not assert.'),
    ('pʰar', 'metal plough tip', '9072', 'phā́la',
     'Dhakal p. 51 prints /pʰar/ “metal plough tip.” Turner9072 explicitly gives Bihari, Maithili and Bhojpuri phār “small ploughshare,” alongside Oriya phāra. This agricultural sense selects 9072 rather than the generic blade family 9052.'),
    ('kaṭi', 'cut', '2854', 'kártati',
     'Dhakal p. 52 prints /kaṭi/ “cut.ABS”; the installed gloss abbreviates this to “cut.” Turner2854 lists Maithili kāṭab, Bihari kāṭab and Oriya kāṭibā “to cut”; the Darai -i is the source response’s absolutive inflection of that verbal stem, not a separate etymon.'),
]
assert len(groups) == 6
accepted, audit = [], []
review = ['# Darai: five verbal and agricultural families', '',
          'Six source-backed reflex records across five etyma.', '',
          '| # | Darai form | Proposed etymology | Evidence |', '|---|---|---|---|']
source_dir = Path('data/other/forms/raw_data/dhakal_darai_2011/decoded')
for i, (form, gloss, parent, label, evidence) in enumerate(groups, 1):
    hits = [r for r in rows if r['Form'] == form and r['Gloss'] == gloss]
    assert len(hits) == 1, (form, gloss, len(hits))
    r = hits[0]
    page = int(r['Source'].split('[p. ')[1].split(']')[0])
    source_text = (source_dir / f'p{page:03}.txt').read_text()
    assert gloss in source_text or (form == 'kaṭi' and 'cut. ABS' in source_text), (form, page)
    audit.append(dict(id=r['ID'], form=form, gloss=gloss, source=r['Source'], page=page,
                      sourceText=source_text))
    accepted.append(dict(record=rec(r), parent=parent, kind='reflex',
                         citation=r['Source'] + f';CDIAL[{parent}]', evidence=evidence))
    review.append(f'| {i} | **{form}** ‘{gloss}’ ({r["Source"]}) | **{label}**, CDIAL **{parent}** | {evidence} |')

holds = []
for form, reason in [
    ('ʣʰanike', 'Turner5363 tentatively connects Nepali jhānnu “fry” to an expressive crackling root; the question mark and absence of regional comparanda prevent a secure inherited assignment.'),
    ('lokike', 'Turner11083 groups OAw lokāvai with the *lukka replacement of lūpyate, but the immediate Darai formation and possible contact route remain unresolved.'),
    ('ʦʰegri', 'The closest Bihari forms occur under both 4959 *chagaṭa and 4963 *chaggala; the Darai e and medial g+r do not securely select one branch.'),
    ('pʰar', 'The separate p. 53 and p. 67 generic “blade” responses cannot yet be distinguished between Turner9052 “blade” and 9072 “ploughshare.”'),
    ('kunʣi', 'Turner3225 treats Nepali kuñji as an arrowed loan from Punjabi/Sindhi; Darai’s immediate source is not established.'),
]:
    for r in rows:
        if r['Form'] == form and (form != 'pʰar' or r['Gloss'] == 'blade'):
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 6 and len(holds) == 7, (len(accepted), len(holds))
(P / 'pass081-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass081-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Six affected records and six reflex rows. Held: *ʣʰanike* fry, *lokike* hide, *ʦʰegri* goat, two generic *pʰar* blade records, and two *kunʣi* key records. No source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 51–52, 54, 56 and 68, checked in source PDF images and decoded text; p. 75 supplies infinitive/past morphology. Discovery: darai-search020–022; complete CDIAL articles darai-primary017–020.',
           'Validation: pending.']
(P / 'pass081-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
