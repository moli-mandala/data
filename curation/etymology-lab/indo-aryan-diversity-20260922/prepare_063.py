import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments

P = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
records = json.loads((P / 'grangali-remaining-current.json').read_text())
linked = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank'] == '1')
source_rows = list(csv.reader(open('data/other/forms/20260819-buddruss-grangali.csv')))
audit_rows = list(csv.DictReader(open('data/other/forms/raw_data/20260819-buddruss-grangali-audit.csv')))

groups = [
    (32, 'dog', '12528', 'śuna', 'Turner12528 directly lists Ningalami šũṛə and Shumashti šū̃ṛə. Buddruss explicitly says Grangali sūrə is assimilated from the same šūrə form; the source sibilants and unmarked nasalization are preserved.'),
    (53, 'flour', '1338', '*ārta²', 'Turner1338 directly lists Shumashti āt flour in the *ārta/aṭṭa family. Buddruss gives identical Grangali and Shumashti āt; this is the flour word, not an unrelated homophone.'),
    (68, 'pomegranate', '6254', 'dāḍima', 'Turner6254 section1 directly gives Shumashti dā́ṛim pomegranate. Buddruss pairs it with Grangali darím; the regional Dardic pair belongs with this section, while Turner section2 covers a separate set of regional dārim forms.'),
    (71, 'grass', '4471', 'ghāsa', 'Turner4471 directly lists Shumashti gā̃s grass, and Buddruss gives the same Grangali and Shumashti form. Both retain the source nasalization; no modern donor is inferred.'),
    (134, 'river', '6943', 'nadī', 'Turner6943 directly gives Ningalami nandī and Shumashti nädī within the early-nasalization series. Buddruss pairs these with Grangali nadí. The three source spellings and vowel quantities remain distinct.'),
    (151, 'head', '12694', '*śrāya', 'Turner12694 directly lists Ningalami ṣoũkṛə and Shumashti šāṛe as enlarged forms of the *śrāya head family. Buddruss pairs these with Grangali ṣā̃rə, noting its retroflex sibilant; the extra nasalization and suffixal differences are retained.')
]

accepted = []
audit = []
review = ['# Grangali survey: six basic-vocabulary families', '',
          'Fifteen reflex records across six families. Buddruss printed pp. 27, 29–30, 33–34 were checked against rendered source images; the full Turner articles were read.', '',
          '| # | Forms and meaning | Parent | Evidence |', '|---|---|---|---|']
for i, (item, gloss, parent, label, evidence) in enumerate(groups, 1):
    hits = [r for r in records if r['Source'].endswith(f'item {item}]') and r['Gloss'] == gloss]
    assert len(hits) == {32:3,53:2,68:2,71:2,134:3,151:3}[item], (item, hits)
    for r in hits:
        assert r['ID'] not in linked, r
        key = f'buddruss-grangali1979:item-{item}' + {'Gng':'','Ning':':ningalami','Shum':':shumashti'}[r['Language_ID']]
        imports = [z for z in source_rows if z[0] == r['Language_ID'] and z[3] == gloss and z[10] == key]
        assert len(imports) == 1, (r, imports)
        aud = [z for z in audit_rows if z['Emitted_Key'] == key and z['Atlas_Numbers'] == str(item) and z['Material_Error'] == 'no']
        assert len(aud) == 1, (r, aud)
        assert aud[0]['Printed_Page'] == r['Source'].split('p. ')[1].split(',')[0]
        accepted.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']}, parent=parent, kind='reflex',
                             citation=f'buddruss-grangali1979[p. {aud[0]["Printed_Page"]}, item {item}];CDIAL[{parent}]',
                             evidence=evidence + ' Reviewed import row and source audit matched; no transcription or identity changes.'))
        audit.append(dict(record_id=r['ID'], source_audit=aud[0], import_row=imports[0], image=f'../tmp/pdfs/grangali/render/page-{int(aud[0]["Printed_Page"])-16:02d}.png'))
    review.append(f'| {i} | **{" / ".join(r["Form"] for r in hits)}** ‘{gloss}’ ({len(hits)}) | **{label}**, CDIAL **{parent}** | {evidence} |')

assert len(accepted) == 15
(P/'pass063-decisions.json').write_text(json.dumps({'accepted':accepted,'held':[]}, ensure_ascii=False, indent=2)+'\n')
(P/'pass063-source-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
review += ['', 'Discovery: grangali-search014; full Turner articles in grangali-primary011. No source repairs or rebuild.', '', 'Save and validation pending.']
(P/'pass063-review.md').write_text('\n'.join(review)+'\n')
print(len(accepted), 'accepted')
