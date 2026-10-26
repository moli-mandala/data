import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='2333',gloss=['above'],languages=['Goj','poth','Hajong','Or','Bhatri','B','RathwiBareli','AdivasiOriya','jaun'],forms=['apar','apur','apor','apar / ūpar','upur','upoṛo','upṛe','upore','upoɾe','ūpɔr','ūpɔre','upro'],citation='CDIAL[2333]',evidence='Turner assigns Punjabi uppar/uppur, Nepali upar, Assamese upari/opare, Bengali upar, Oriya upari/upuri/upara, Hindi ūpar and Gujarati upar to *uppari, the emphatically geminated formation of upari. These whole survey above forms match that retained-p regional series, with ordinary oblique/locative final vowels. The single-p Sanskrit upari article instead gives intervocalic-v or zero outcomes. Other stems, extra words and unresolved expanded endings are excluded.'),
 dict(parent='13776',gloss=['big'],languages=['Dotyali','kul'],forms=['ṭʰulo','tʰulo','t̪ʰuḷə'],citation='CDIAL[13776]',evidence='Turner lists Kumaoni ṭhūlo and Nepali ṭhulo large, alongside the wider thūl-/ṭhūl- series under sthūla. Dotyali ṭhulo/thulo and Kullu thulə preserve this th-l large adjective; the source dental versus retroflex spelling is retained. The assignment concerns the regional inherited adjective family and does not normalize the attested vowels.'),
 dict(parent='10187-11',gloss=['big'],languages=['Bhili','Khandesi','mewari_basad','Vasavi','PauriBareli','RathwiBareli'],forms=['moto','moṭhā','mūṭo','mūṭṭā','motho','motu','motto','motā'],citation='CDIAL[10187,11]',evidence='The eleventh *mōṭṭa branch explicitly includes Old Marwari moṭaü big/fat, Gujarati moṭũ and Marathi moṭā, with Jaunsari mōṭō big/fat. These simple western survey moṭ-/mot- adjectives fit that expressive big/fat family, including local vowel and aspiration variation; the relation is qualified family membership rather than a regular sound-law reconstruction. l-expanded adjectives and voiced ḍ forms are not included.'),
 dict(parent='5617',gloss=['sour'],languages=['B'],forms=['tɔk','ʈɔk'],citation='CDIAL[5617,2]',evidence='The second semantic branch of takra explicitly lists Bengali tak sour/acid, ṭak sourness and ṭaka sour, alongside Oriya ṭaka. Both Bengali survey vowel spellings and dental/retroflex source variants belong to that documented sour series. The compiled dictionary keeps both senses on the existing 5617 node, so no nonexistent subsection ID is invented.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and r['Language_ID'] in g['languages']:a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass027-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n')
(P/'pass027-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
