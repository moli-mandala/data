import json
from pathlib import Path
P=Path(__file__).resolve().parent;ledger=json.loads((P/'pass-ledger.json').read_text());done={x['record']['ID'] for f in ledger['decisionFiles'] for x in json.loads((P/f).read_text())['accepted']}
evidence={
'9926':'CDIAL 9926 masta/mastaka gives Gujarati māthũ and Marathi māthā head. The printed matha form matches that dental-th family.',
'5014':'CDIAL 5014 *chātti gives Gujarati/Hindi chātī chest, breast. The printed cati-type form has dental t and matches this anatomical family.',
'14024':'CDIAL 14024 hasta explicitly includes Hindi/Gujarati hāth hand, arm, cubit and Marathi hāt. Thus the elicited arm sense is documented as well as hand.',
'10539':'CDIAL 10539 rakta explicitly means blood as well as red. The rakt-/ragat-/rakat- survey shapes retain this lexical family, with learned or regional circulation left open.',
'10203':'CDIAL 10203 mudrā gives Punjabi mundī, Oriya mudi and Marathi mudī ring. Printed dental-d mundi/mudi forms match these whole-word comparisons.',
'4661':'CDIAL 4661 candra gives Punjabi cand and Gujarati cā̃d/cā̃do moon. The printed cand/cando forms have dental d, not a divergent retroflex stop.',
'6943':'CDIAL 6943 nadī means river; the printed nadi forms retain the dental d of this noun. Learned or regional circulation may underlie the conservative shape and remains open.',
'7733':'CDIAL 7733 pattra gives Hindi pattā/pātā and Gujarati pātũ leaf. The printed dental t/tt forms match this family rather than requiring a retroflex development.',
'2723':'CDIAL 2723 kanda gives Gujarati kā̃do and Marathi kā̃dā onion. The printed kand-/kãd- forms have dental d and the exact onion sense.',
'3277':'CDIAL 3277 *kuttira explicitly gives Gujarati kutro and Marathi kutrā dog. This r-bearing branch fits the printed dental-t forms.',
'11515':'CDIAL 11515 vānara explicitly gives Gujarati vā̃dar/vā̃dro monkey. The printed vandra forms have dental d and the documented nasal-stop insertion.',
'6261':'CDIAL 6261 *dādda explicitly gives Marathi/Hindi dādā elder brother. The printed dadu term belongs to this expressive kinship family with its source vowel retained.'}
rules=[dict(parent=p,citation='CDIAL['+p+']',evidence=ev) for p,ev in evidence.items()];ix={q['parent']:i for i,q in enumerate(rules)};acc=[];cells=[]
for x in json.loads((P/'noira-dental-discovery.json').read_text()):
 r=x['record'];c=x['sourceCell']
 if r['ID'] in done or int(c['PDF_Page']) not in [33,35,37,41,43,46,49,53,56]:continue
 assert len(x['parents'])==1;p=x['parents'][0];i=ix[p]
 cite=rules[i]['citation']+';varghesekumar2015noira[Appendix A3, p. '+c['Printed_Page']+', item '+c['Item']+']'
 ev=evidence[p]+' Visual inspection of the original PDF physical p. '+c['PDF_Page']+' confirms the dental d/t reading underlying stored '+r['Form']+'. The source ledger has misread these dental-stop glyphs as retroflex; this link uses the printed reading while preserving the existing form and durable ID. No universal dental/retroflex replacement or new sound law is asserted. Intra-Indo-Aryan transmission remains open.'
 acc.append(dict(record=r,parent=p,family=i,kind='reflex',citation=cite,evidence=ev));cells.append(dict(recordId=r['ID'],storedForm=r['Form'],sourceCell=c,verifiedDentalStops=True,verification='Visual inspection of the complete relevant printed page at a 2600-pixel render. Only selected stop glyphs are verified.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[])),('verified-cells',cells)]:(P/('noira-dental-second-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'noira_dental_second_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth','noira-dental-second'))
print({'accepted':len(acc),'parents':len({x['parent'] for x in acc})})
