import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
q0=dict(parent='f_ho2hp2hvmz5oy',citation='Platts[p. 444]',evidence='Platts p. 444 explicitly gives Hindi candr-mā/candar-mā moon and analyses it as Sanskrit candra plus mās. The existing candramas parent preserves this full compound, distinct from candra alone. Retained ndr-m identifies the learned Sanskrit word in these Hindi attestations.')
q1=dict(parent='f_lvqjpfnxdtcla',citation='Platts[p. 444];kannauji[p. 64]',evidence='Platts p. 444 verifies the whole Hindi candr-mā/candar-mā moon compound. The Hindi survey candrama attestation (kannauji p. 64) is linked to Sanskrit candramas in this same validated batch and provides a provisional immediate regional donor for the other survey forms. Independent Sanskrit learning or another Indo-Aryan intermediary remains possible; the attestation locality is not asserted to be the historical source.')
qs=[q0,q1];acc=[];held=[];raw={r['ID']:r for r in json.loads((P/'inventory.json').read_text())}
words={norm(w) for w in ['candramā','cāndramā','candrama','candrmā','candrāma','candəramā','cəndərama','cəndrəma']}
for rr in csv.DictReader(open(P/'unresearched-records.csv')):
 r=raw[rr['ID']]
 if r['Gloss']!='moon' or norm(r['Form']) not in words:continue
 i=0 if r['Language_ID']=='H' else 1;q=qs[i]
 acc.append(dict(record=r,family=i,parent=q['parent'],kind='borrowed',citation=q['citation'],evidence=q['evidence']+' Exact source form '+r['Form']+' is retained.'))
assert q1['parent'] in {x['record']['ID'] for x in acc}
(P/'moon-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'moon-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'moon_save.py').write_text((P/'groundnut_save.py').read_text().replace('groundnut','moon'))
a=[x for x in json.loads((P/'platts-human-moon-research.json').read_text()) if x['word']=='चन्द्रमा'];(P/'moon-primary-articles.json').write_text(json.dumps({q['parent']:a for q in qs},ensure_ascii=False,indent=1));print(len(acc))
