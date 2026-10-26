import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments

P = Path('curation/etymology-lab/indo-aryan-diversity-20260922')
records = json.loads((P/'grangali-remaining-current.json').read_text())
linked = {r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
imports = list(csv.reader(open('data/other/forms/20260819-buddruss-grangali.csv')))
source = list(csv.DictReader(open('data/other/forms/raw_data/20260819-buddruss-grangali-audit.csv')))
groups = [
    ('43','twelve','6658','dvādaśa','Turner6658 section1 directly lists Ningalami bas and Shumashti bās twelve. The source Grangali bas matches that regional b-s shape. The distinct duv- section2 is excluded.'),
    ('103','eleven','2485','ekādaśa','Turner2485 directly lists Shumashti zās eleven alongside western Dardic z/ž formations. Buddruss gives Grangali and Ningalami zas with the same z-s frame; all source vowel quantities are retained.'),
    ('123-124','fourteen','4605','caturdaśa','Turner4605 section1 directly lists Ningalami ċaudēs and Shumashti ċäudas fourteen. Grangali ʦōdḗs follows the same c-d-s family, with its source long vowels preserved.'),
    ('141','sixteen','12812','ṣoḍaśa','Turner12812 section1 directly lists Ningalami ṣurós and Shumashti ṣōṛā́s sixteen. Grangali sōrā́s matches the latter Dardic formation; the source sibilant and r spelling remain unchanged.'),
    ('144','six','12803-2','*ṣuvaṭ','Turner12803 section2 directly lists Ningalami ṣō six and neighbouring Ashkun ṣu. Grangali ṣu fits that u-vowel branch; the section1 śeh and section3 cha branches are excluded.'),
    ('154','thirteen','6001-2','*trayēdaśa','Turner6001 section2 directly lists Ningalami ṣlewás and Shumashti λäṷas thirteen. Buddruss Grangali λēvā́s belongs to this lateral-initial Dardic series; its vowel and long ā are retained.')
]
accepted=[];audit=[]
lines=['# Grangali survey: higher numerals','','Seventeen reflex records across six numeral families. Buddruss printed pp. 28, 31–34 visually checked; full Turner articles read.','','| # | Forms and meaning | Parent | Evidence |','|---|---|---|---|']
for i,(item,gloss,parent,label,evidence) in enumerate(groups,1):
    hits=[r for r in records if r['Gloss']==gloss and r['Source'].endswith(f'item {item}]')]
    assert len(hits)=={'43':3,'103':3,'123-124':3,'141':3,'144':2,'154':3}[item],(item,hits)
    for r in hits:
        assert r['ID'] not in linked,r
        key=f'buddruss-grangali1979:item-{item}'+{'Gng':'','Ning':':ningalami','Shum':':shumashti'}[r['Language_ID']]
        ir=[z for z in imports if z[0]==r['Language_ID'] and z[3]==gloss and z[10]==key]
        ar=[z for z in source if z['Emitted_Key']==key and z['Atlas_Numbers']==item and z['Material_Error']=='no']
        assert len(ir)==len(ar)==1,(r,ir,ar)
        assert ar[0]['Printed_Page']==r['Source'].split('p. ')[1].split(',')[0]
        accepted.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=f'buddruss-grangali1979[p. {ar[0]["Printed_Page"]}, item {item}];CDIAL[{parent}]',evidence=evidence+' Reviewed source audit and import row matched; no transcription or identity changes.'))
        audit.append(dict(record_id=r['ID'],source_audit=ar[0],import_row=ir[0],image=f'../tmp/pdfs/grangali/render/page-{int(ar[0]["Printed_Page"])-16:02d}.png'))
    lines.append(f'| {i} | **{" / ".join(r["Form"] for r in hits)}** ‘{gloss}’ ({len(hits)}) | **{label}**, CDIAL **{parent}** | {evidence} |')
held=[r for r in records if r['Gloss']=='nineteen' and r['Source'].endswith('item 41]') and r['ID'] not in linked]
assert len(held)==2
reason='Buddruss flags an irregular loss of the nasal in the Ningalami usí comparison; the Grangali visī́ and Ningalami usí forms need their exact nineteen-versus-twenty branch independently checked.'
h=[dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},reason=reason) for r in held]
assert len(accepted)==17
(P/'pass067-decisions.json').write_text(json.dumps({'accepted':accepted,'held':h},ensure_ascii=False,indent=2)+'\n')
(P/'pass067-source-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
lines+=['','Two nineteen forms held: '+reason,'','Discovery: grangali-search015; full primary articles grangali-primary012–013. No rebuild.','','Save and validation pending.']
(P/'pass067-review.md').write_text('\n'.join(lines)+'\n')
print(len(accepted),'accepted',len(h),'held')
