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
    ('putəu', 'daughter-in-law', '8270', 'putravadhū',
     'Dhakal p. 47 prints /putəu/ “daughter-in-law.” Turner8270 gives Maithili putahu “son’s wife” and Bhojpuri/Awadhi patoh alongside its putravadhū head; the Darai final diphthong and absent h agree with this eastern family.'),
    ('bʰaim', 'my brother', '9661', 'bhrā́tr̥',
     'Dhakal p. 48 prints /bʰaim/ “my brother.” Turner9661 lists Prakrit bhāi and Nepali, Maithili, Bhojpuri bhāi “brother”; it also documents -m in “my brother” forms from other lects. The Darai possessive response remains one source record.'),
    ('dulʰa', 'husband', '6446', 'durlabha',
     'Dhakal p. 56 prints /dulʰa/ “husband.” Turner6446 lists Bihari and Maithili dulhā “bridegroom” from the durlabha family; the Darai form denotes the married male rather than only his wedding role.'),
    ('dulʰi', 'wife', '6446', 'durlabha',
     'Dhakal p. 56 prints /dulʰi/ “wife.” Turner6446 explicitly gives the feminine Nepali dulahi “bride” and Maithili/OAw dulahini; the Darai reduced feminine form denotes the married female. The source does not establish a separate synchronic derivation.'),
    ('kevari', 'door', '2963', 'kavāṭa',
     'Dhakal pp. 57, 66 and 67 print /kewari/ “door.” Turner2963 lists Bihari kewāṛī, Maithili kewāṛī and Hindi kewār/kiwāṛ “door”; Darai transcription v represents the source /w/. Its remote Dravidian origin is described by Turner as possible, not settled.'),
    ('samdʰi', "son's father in law", '12957', 'saṁbandhin',
     'Dhakal p. 60 prints /samdʰi/ “son’s father-in-law.” Turner12957 gives Maithili samdhī with exactly that meaning and Nepali samdhi “child’s father-in-law”; the Darai cluster and kin relation match the first, masculine branch.'),
]
accepted, audit = [], []
review = ['# Darai: kinship and door', '',
          'Eight source-backed reflex records across five etyma.', '',
          '| # | Darai form | Proposed etymology | Evidence |', '|---|---|---|---|']
source_dir = Path('data/other/forms/raw_data/dhakal_darai_2011/decoded')
i = 0
for form, gloss, parent, label, evidence in groups:
    hits = [r for r in rows if r['Form'] == form and r['Gloss'] == gloss]
    assert len(hits) == (3 if form == 'kevari' else 1), (form, gloss, len(hits))
    for r in hits:
        i += 1
        page = int(r['Source'].split('[p. ')[1].split(']')[0])
        source_text = (source_dir / f'p{page:03}.txt').read_text()
        assert gloss.split()[0] in source_text or (form == 'samdʰi' and 'father in law' in source_text), (form, page)
        audit.append(dict(id=r['ID'], form=form, gloss=gloss, source=r['Source'], page=page,
                          sourceText=source_text))
        accepted.append(dict(record=rec(r), parent=parent, kind='reflex',
                             citation=r['Source'] + f';CDIAL[{parent}]', evidence=evidence))
        review.append(f'| {i} | **{form}** ‘{gloss}’ ({r["Source"]}) | **{label}**, CDIAL **{parent}** | {evidence} |')

holds = []
for form, reason in [
    ('nəmda', 'Turner6946 lists Nepali nanda “husband’s younger sister,” but Dhakal’s /nəmda/ has medial m rather than the regional n; this needs a sound or transcription account.'),
    ('bʰəŋna', 'Turner9433 bhāginēya gives regional bhāṇj-/bhāṇej- for “nephew,” but the two Darai /bʰəŋna/ records lack the diagnostic palatal element; no exact branch is secure.'),
]:
    for r in rows:
        if r['Form'] == form:
            holds.append(dict(record=rec(r), reason=reason))

assert len(accepted) == 8 and len(holds) == 3, (len(accepted), len(holds))
(P / 'pass089-decisions.json').write_text(json.dumps(dict(accepted=accepted, held=holds), ensure_ascii=False, indent=2) + '\n')
(P / 'pass089-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(audit), rows=audit), ensure_ascii=False, indent=2) + '\n')
review += ['', 'Eight affected records and eight reflex rows. Held: *nəmda* husband’s younger sister and two *bʰəŋna* nephew records. The prior Darai holds remain excluded. No lexical-source, identity or build changes.',
           'Primary source: Dhakal 2011 printed pp. 47–48, 56–57, 60, 66–67, checked in source PDF images and decoded text. Discovery: darai-search025–027; complete CDIAL articles darai-primary024–026.',
           'Validation: pending.']
(P / 'pass089-review.md').write_text('\n'.join(review) + '\n')
print(len(accepted), 'accepted', len(holds), 'held')
