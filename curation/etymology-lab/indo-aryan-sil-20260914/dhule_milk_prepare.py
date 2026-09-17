import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;ledger=json.loads((P/'pass-ledger.json').read_text());done={x['record']['ID'] for f in ledger['decisionFiles'] for x in json.loads((P/f).read_text())['accepted']}
q=dict(parent='6391',citation='CDIAL[6391];watters2013northerndhule[Appendix C, p. 101, item 91]',evidence='CDIAL 6391 dugdha gives Gujarati/Marathi dūdh and unaspirated regional dud/dūd milk. Independent visual inspection of Northern Dhule physical p. 109, printed p. 101, item 91 shows plain/dental d with dental subscript marks, not the retroflex reading stored in the source ledger. The link follows the printed milk word. Stored forms and durable IDs remain unchanged pending source correction; the exact intra-Indo-Aryan transmission is open.')
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in done or 'watters2013northerndhule' not in r.get('survey_sources',[]) or r['Gloss']!='milk':continue
 assert r['Form'] in ['ḍuḍ','duḍ','ḍuḍɦ']
 acc.append(dict(record=r,parent='6391',family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Stored survey form '+r['Form']+' is preserved.'))
assert len(acc)==12
for suffix,obj in [('rules',[q]),('decisions',dict(accepted=acc,held=[]))]:(P/('dhule-milk-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
a=json.loads((P/'noira-dental-primary-articles.json').read_text());(P/'dhule-milk-primary-articles.json').write_text(json.dumps({'6391':a['6391']},ensure_ascii=False,indent=1)+'\n')
pdf=P.parents[3]/'tmp/pdfs/northern_dhule_bhils_2013/silesr2013_004.pdf'
report=dict(source='watters2013northerndhule',physicalPage=109,printedPage=101,item=91,pdf=str(pdf),pdfSha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),records=[x['record']['ID'] for x in acc],finding='Plain/dental d and dental subscript marks in the milk row were stored as retroflex stops. This row was independently inspected at page scale and in a 400-dpi close-up.',scopeLimit='Only the milk row is verified. Other rows and sources must not be globally regularized.',sourceDataEdited=False)
(P/'dhule-milk-source-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=1)+'\n')
(P/'dhule-milk-source-audit.md').write_text('# Northern Dhule milk: source glyph mismatch\n\nPhysical p. 109 (printed p. 101), item 91, was independently inspected as a complete page and in a 400-dpi close-up. Plain/dental d plus dental subscript marks have been stored as retroflex ɖ/ḍ. All twelve survey milk records link to CDIAL 6391 dugdha; source forms and IDs remain unchanged.\n\nThis finding is limited to item 91. Other items require independent glyph checks. A lexical-source correction requires the source-ingestion checklist and identity/graph reconciliation; none was performed during this etymology pass.\n\nExact IDs and PDF identity are in `dhule-milk-source-audit.json`.\n')
(P/'dhule_milk_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth','dhule-milk'))
print('Prepared',len(acc))
