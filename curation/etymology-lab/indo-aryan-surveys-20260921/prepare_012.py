import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
dict(parent='12803-2',gloss=['six'],languages=['Mai','Gaw','Bshk','Chil','Gowro','bhatr'],forms=['śo','śvā'],citation='CDIAL[12803,2]',evidence='Full ṣaṣ separates section 2 *ṣuvaṭ from the simple ṣaṣ and affricate *kṣaṭ branches. The selected northwestern śo/śvā forms match Turner’s specific Maiya šōh/ṣō, Gawri ṣō, Bashkarik ṣɔ and Kohistani Shina ṣvā comparanda in section 2. Their rounded vowel or labial glide selects this branch; plain śe/ṣa is not included.'),
dict(parent='12803-3',gloss=['six'],forms=['coi','cʰɔī̯','cʰɔi̯','ʦ̣hoī','ʦ̣oī','coī','tshō','choū','tṣho','tsh o','co','cāu̯','cay','cʌi','tʃhɔi ̯','cɔ','cɔ̄','ts(h)o','tso','saha','sāū','savā','sav','sao','soa','sɔve','sɔ','so','sa'],languages=['Hajong','Bishnupriya','Kho','RathwiBareli','Bhilali','Bhili','PauriBareli','Noiri','Vasavi','DungraBhili','Khandesi','DewasDoneDanuwar','AdivasiOriya','Bhatri','hal','B','Rathawi','kul'],citation='CDIAL[12803,3]',evidence='Turner’s section 3 *kṣaṭ/*kṣvaṭ gives Khowar c̣hoi, Bengali chay, Kumaoni chai, Western Pahari chɔ, Gujarati cha and Marathi sahā/sā. The selected forms belong to these named regional affricate or western sahā families, including ordinary source c/ts spelling, deaspiration and contracted h/v variants. They are kept separate from the northwestern ṣuvaṭ branch; added classifier material is excluded.'),
dict(parent='13139',gloss=['seven'],forms=['sāṭ','hāṭ','sato','satɔ','sōtʰ','sotʰ','ˈsotə','sotə','ṣāṭ','saṭʰ'],citation='CDIAL[13139]',evidence='Full sapta gives the sāt/satt numeral throughout Indo-Aryan, including Old/modern Oriya sāta and western forms. These are simple seven responses with the source’s dental/retroflex and aspirated variants preserved; the Bhil hāṭ forms continue the corresponding s-to-h regional series. Forms containing a second eight numeral, a classifier, or a malformed bracket are excluded rather than reduced to seven.'),
dict(parent='941',gloss=['eight'],forms=['ātʰ','aaṭh','uśt','æʦ̣','ɛś','aṭʰo','ã.ṭʰ','ath','atʰɔ','ətʰ','otʰ','atʰ','ˈotʰə','otʰə','āṭhe'],citation='CDIAL[941]',evidence='Full aṣṭā/aṣṭau includes Khowar oṣṭ, Bashkarik ac̣, Torwali/Maiya āṭh and the widespread plains āṭh/āṭha series. These selected whole numeral responses fit the named branches, retaining the source’s palatal, retroflex/dental and aspiration notation and final nominal vowel. The n-initial forms filed under eight, which resemble nine, are excluded as gloss/alignment problems; classifier phrases are also excluded.'),
dict(parent='12548',gloss=['dry'],languages=['Hajong','Bishnupriya','Rana'],forms=['huknu','hukna','hukana','hukanī','sūkhanā'],citation='CDIAL[12548]',evidence='The full śuṣka article explicitly includes the expanded Prakrit sukkhāṇa adjective and Assamese xukān/śukān, in addition to simple sukkha. This n-bearing regional dry adjective explains the entire hukna/hukana stem; the initial h corresponds to the eastern x/s series and the ending vowel is retained. The link identifies that expanded dry family and leaves regional contact transmission open; hukase and other additional verbal suffixes are not silently stripped.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   a.append(dict(record=r,parent=g['parent'],kind='reflex',citation=g['citation'],evidence=g['evidence'],family=i))
held=[]
for r in rs:
 if r['Gloss']=='eight' and r['Form'] in ['nõ','nīu','nū̃','nūm','naū','ṇū̃','nō','no','noū̃','no / nao','nāū','nãū̃']:
  held.append(dict(record=r,candidates=[],reason='The n-initial response resembles the local nine numeral rather than eight; the SSNP sequence also has combined seven/eight responses. Do not assign an etymon from the mismatched gloss without checking the original row alignment. Source/identity repairs are outside this no-rebuild etymology pass.'))
 if r['Gloss']=='seven' and ('aṭ' in r['Form'] or 'āṭ' in r['Form'] or 'aṣ' in r['Form'] or 'āṣ' in r['Form']) and len(r['Form'])>4:
  held.append(dict(record=r,candidates=['13139','941'],reason='The recorded string appears to contain both seven and eight; the adjacent SSNP eight records often contain nine. Hold the original source alignment rather than treating this as one numeral or saving a compound without a source check.'))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass012-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass012-decisions.json').write_text(json.dumps({'accepted':a,'held':held},ensure_ascii=False,indent=1)+'\n')
print(len(a),[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)],'held',len(held))
