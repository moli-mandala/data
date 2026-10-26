import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='10452',gloss=['go','to go'],languages=['Hajong','B','MagahiNepal','Or','hal','RathwiBareli'],forms=['ja','java','dʒawa','janāi','jiba','jauk','dzāṇe'],citation='CDIAL[10452]',evidence='Turner explicitly gives Bengali jāoyā, Assamese zāiba, Oriya jibā, Bihari/Maithili jāeb, Old Marwari jāvaï and Gujarati jāvũ under yāti go. These whole survey j-/dz- nonpast or infinitive forms fit those regional stems and endings. Past g- forms are assigned separately to gata; negative prefixes and complex responses are excluded.'),
 dict(parent='10452',gloss=['come'],languages=['Phal'],forms=['yā-'],citation='CDIAL[10452]',evidence='Turner specifically assigns Palula y-/yh-, imperative yah and present yhāndu am coming, to yāti. The Palula survey yā- come is therefore this inherited go/come stem; the English elicitation gloss does not justify assigning it to āpayati.'),
 dict(parent='6141',gloss=['give','to give'],languages=['Kho','Kal','Phal','Hajong','Majhi','Bote','poth','Dang','Kathoriya'],forms=['di','dī','diva','ḍih','ḍeh'],citation='CDIAL[6141]',evidence='Turner groups Khowar diyār, Kalasha/Palula deh, Kashmiri dih, Nepali dinu, Assamese diba and the widespread de-/ḍe- present/imperative series under dadāti, explaining its analogical remodeling with dēti and dēhi. These simple survey di-/de- give forms fit that family. The distinct prā preterite and suffixed past-participial forms are excluded.'),
 dict(parent='4008',gloss=['go'],languages=['Dotyali','Vasavi','Noiri','RathwiBareli','Dang'],forms=['gāⁱyo','ghio','giyo','gaiyo','goyu','goyo','gei'],citation='CDIAL[4008,1]',evidence='Turner identifies Hindi gayā, Nepali/Kumaoni gayo and Old Gujarati gyau/giu as suppletive preterites of go from gata. These survey gV(y)- forms preserve that gone/past stem even though the elicitation gloss is simply go. Forms retaining an l expansion are assigned to the separate *gatailla branch.'),
 dict(parent='4008-2',gloss=['go'],languages=['KochariyaEastDanuwar','KochilaTharu','AdivasiOriya','Bhatri','Vasavi','PauriBareli'],forms=['gelo','gele','galaⁱ','gʸɛle','gala','gələ','gʌla','gəla','gehlo','goilu'],citation='CDIAL[4008,extension illa]',evidence='Turner explicitly separates the illa extension of gata: Prakrit gaïllaya, Assamese gâla, Bengali gela, Oriya gali/gelā, Bihari gel and Bhojpuri gail. These survey g-l past go forms match that extension, with ordinary agreement vowels and local vowel reduction. The Danuwar records are explicitly tagged third-person singular preterite; none is treated as a present yāti stem.'),
 dict(parent='3784',gloss=['to stand'],languages=['Hajong'],forms=['kʰara'],citation='CDIAL[3784,1]',evidence='Turner specifically lists Bengali and Oriya khāṛā standing erect under khaḍaka, beside Hindi khaṛā. The Hajong survey gives the simple khara adjective-shaped response to to stand. It is linked to that standing/erect family, with the source dental-r notation retained; extra ho/huva verbal material is excluded from this single-parent analysis.'),
 dict(parent='13479',gloss=['sleep'],languages=['Vasavi'],forms=['huṭalo'],citation='CDIAL[13479]',evidence='Turner gives Prakrit sutta asleep, Old Gujarati sūtaü and Bhojpuri sūtal under supta, distinguishing this t-bearing past/asleep stem from the present svap-/sup- series. Vasavi huṭalo retains the corresponding t-plus-l form, with western s-to-h and the survey’s retroflex spelling. The link is to the participial sleep family; it does not claim an independently demonstrated history for every local suffix.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and r['Language_ID'] in g['languages']:a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
for r in rs:
 if r['ID'] not in done and r['Language_ID']=='poth' and r['Gloss']=='come' and r['Form'] in ['aś','ac','as']:
  h.append(dict(record=r,reason='Turner 227 and 1044 discuss competing atyeti versus āgacchati sources for regional ac-/aś- come forms and disagreeing scholarly assignments. The Pothwari survey alone does not choose between them, so no categorical parent is saved.'))
(P/'pass031-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass031-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
