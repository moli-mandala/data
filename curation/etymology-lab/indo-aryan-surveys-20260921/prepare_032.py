import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='8938a',gloss=['give'],languages=['Kho','Kal'],forms=['prā'],citation='CDIAL[8938a];CDIAL[6141]',evidence='Turner 8938a identifies Khowar prai gave with the aorist prādāt of pradadāti; 6141 independently specifies Kalasha preterite pra from prādāt. These survey prā give responses preserve that suppletive past stem, despite their generic English gloss, rather than the present/imperative de-/di- stem.'),
 dict(parent='12605',gloss=['sleep'],languages=['Dotyali'],forms=['siyo','śeyo','śiyo'],citation='CDIAL[12605,1]',evidence='Turner gives Kumaoni sīṇo/seṇo repose/sleep and Prakrit sēi/sēaï under śete. The Dotyali sī-/śe-/śi- sleep series matches that eastern Pahari e/i-vowel stem with the survey -yo ending. This is a regional comparative assignment; the source s versus ś is preserved, and it is not grouped with t-bearing supta past forms.'),
 dict(parent='1045-2',gloss=['come'],languages=['KochariyaEastDanuwar','KochilaTharu','Vasavi'],forms=['yelo','yɛli','yeleⁱ','ālo'],citation='CDIAL[1045,extension lla]',evidence='Turner explicitly lists Bengali āila came, Old Maithili ayalahũ I came and Marathi ālā in the lla expansion of āgata. The eastern yel-/yāl-type come preterites and Vasavi ālo match that regional expanded past family. Danuwar yelo is explicitly tagged third-person singular preterite; the eastern onset/vowel reduction is a comparative qualification, not a general sound law. v-bearing forms with a potentially different present-stem origin are excluded.'),
 dict(parent='2426',kind='derived',gloss=['stand','to stand'],languages=['Dotyali','Majhi','Bishnupriya'],forms=['ubʰyo','ubi','uba','ubanī'],citation='CDIAL[2426,derivatives]',evidence='Turner gives Bengali/Oriya ubhā upright and explicitly identifies denominative stand verbs, including Nepali ubhinu and Hindi ūbhnā, under ūrdhva. These survey ub(h)- stand forms are grouped as derivatives of the upright adjective, with ordinary past or infinitive endings. The r-expanded ubhār- formation has its own etymon and is not included.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and r['Language_ID'] in g['languages']:a.append(dict(record=r,parent=g['parent'],kind=g.get('kind','reflex'),citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
for r in rs:
 if r['ID'] not in done and r['Gloss'] in ['sleep','(you) sleep'] and r['Language_ID'] in ['awan','poth'] and r['Form'] in ['sɛ','se']:
  h.append(dict(record=r,reason='The short se sleep imperative needs closer dialect evidence to distinguish svapati (13902, including Lahnda sēaṇ and possible Garhwali seṇu) from śete (12605, Kumaoni seṇo/sīṇo). Hold the short forms rather than select a root from vowel shape alone.'))
(P/'pass032-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass032-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
