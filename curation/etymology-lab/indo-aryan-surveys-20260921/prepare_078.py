import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
(dict(parent='4989',kind='reflex',citation='CDIAL[4989]',gloss='roof',forms=['cʰān','cʰāni','cʰānā'],languages=['DewasDoneDanuwar','KochariyaEastDanuwar','Majhi','Bote'],evidence='Turner channa covered explicitly includes Nepali chānu roof, Kumauni chāno shed and Garhwali chān cowshed. These adjacent Nepal survey roof forms are compared with that attested roof family. The i-ending forms are provisionally grouped with this regional family; the neighbouring *channi covering family (4992) remains a morphological alternative.')),
(dict(parent='4992',kind='reflex',citation='CDIAL[4992]',gloss='roof',forms=['cānī','canhī','sānhi','cā̃ⁱ','sani','cʰani','cani','cʰanī','cān'],languages=['Rana','Chitwan','KochilaTharu','AdivasiOriya','hal','Bhatri','kaithal'],evidence='Turner *channi covering explicitly records Bihari chānh/chānhī/chānhiyā thatch roof, Old Awadhi chāna and Punjabi chann thatched roof. These central/eastern survey roof responses fit that noun family, with deaspiration and local ch/s spellings retained. The link is a regional comparative assignment; channa 4989 is related but separately represented.')),
(dict(parent='10582',kind='reflex',citation='CDIAL[10582]',gloss='rope',forms=['rāźūlī','rāz','leju'],evidence='Turner rajju explicitly records regional rāž/raž/raz rope, Pahari raj̈ūṛī, and Hindi lej/lejur. The Maiya z/ź forms and Kaithal leju fit those documented developments of the same rope family. Survey responses with ambiguous y transcription are excluded.')),
(dict(parent='f_hgtlpiqxjmpyy',kind='borrowed',citation='CDIAL[10648]',gloss='rope',forms=['ressī','ṛessī','reserī','resurī'],evidence='Turner explicitly identifies Punjabi rassī as the source of the widespread Hindi/Bihari rassī/rasrī and Awadhi rasarī rope family. These Bagheli forms are treated as contact forms; the existing Hindi rassī node is used as a provisional nearby donor proxy, preserving its Punjabi ancestry. The exact regional path of transmission and the local expanded vowels remain uncertain.'))]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss']==g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   a.append(dict(record=r,parent=g['parent'],kind=g['kind'],citation=g['citation'],evidence=g['evidence'],family=i))
# Avoid selecting between inherited channa and channi for i-ending Nepal forms.
held=[];keep=[]
for x in a:
 if x['family']==0 and x['record']['Form']=='cʰāni': held.append(dict(record=x['record'],reason='Roof: channa 4989 vs *channi 4992 is not distinguished by the local i-ending. Hold for a precise morphological history.'))
 else:keep.append(x)
a=keep
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass078-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=1)+'\n')
(P/'pass078-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(held),'held',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
