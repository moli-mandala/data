import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[{'components': ['5943', '10895'], 'gloss': ['he is, he was thirsty'], 'forms': ['tiś lagi', 'tis lagilɛ'], 'citation': 'CDIAL[5943];CDIAL[10895]', 'evidence': 'The complete predicate is thirst + be felt/strike. Turner explicitly lists Jaunsari tīś under *tṛṣyā and the neighbouring Jaunsari expression cīś lāgṇi to be thirsty; lagyati supplies lāg- be felt/begin. The final verbal endings are preserved as inflection; the two lexical constituents are recorded in order.', 'languages': ['jaun']}, {'components': ['5936', '10895'], 'gloss': ['he is, he was thirsty'], 'forms': ['tīs lāgī', 'tīs lagī', 'tīslāgī', 'tis lāge', 'tis lāgī', 'tīs lāgayā', 'tis laga', 'taraslāgī', 'taraslagī', 'taraslāgi', 'tislāgni', 'tīh-lagaṇe'], 'citation': 'CDIAL[5936];CDIAL[10895]', 'evidence': 'The whole predicate consists of thirst + lag- be felt/strike. Turner 5936 documents Hindi tis, Punjabi tīh and Gujarati taras in the tṛṣā family; 10895 documents regional lāg- be felt/touch/begin. The past and infinitival endings belong to the modern lag- paradigm, rather than adding an independent auxiliary. Both ordered lexical constituents are represented; reduced tar/tas forms and clauses with additional copulas are excluded.'}, {'components': ['8199', '10895'], 'gloss': ['he is, he was thirsty'], 'forms': ['pīyas lagī', 'pīyas legī', 'pīyaslagaṭʰ', 'pyāslagī'], 'citation': 'CDIAL[8199];CDIAL[10895]', 'evidence': 'The complete response is thirst + be felt, pipāsā through piās/pyās (explicitly Awadhi and Hindi in Turner) plus regional lag- be felt/strike. The lag-/leg- variation and tense endings are preserved as modern verbal inflection; clauses with independent copulas are excluded.'}, {'components': ['9286', '10895'], 'gloss': ['he is, he was hungry'], 'forms': ['bʰūk lāgī', 'bʰūk lagī', 'bʰūkʰlagī', 'bʰūklagī', 'bʰūklagyāyī', 'bʰuk legī', 'bʰukʰ lagī', 'būk lagla', 'būk laglanī', 'bʰūk laglanī', 'būk lagulī', 'bʰūk lagtaī', 'bhuk lagi', 'bhuk lagɛ', 'bʰūklāgī', 'bhūk-lagaṇe', 'bhūklagni', 'bhūklagnil', 'bhūklāgni', 'bhūklagi'], 'citation': 'CDIAL[9286];CDIAL[10895]', 'evidence': 'The full predicate is hunger + be felt/strike. Turner bubhukṣā lists Hindi bhūk(h), Old Marwari bhūkha, Oriya bhuk(h)a/bhoka and Jaunsari bhukh; lagyati supplies the matching regional lag-/lāg- paradigm, including Oriya lāgibā and western Pahari be felt. These responses contain two lexical constituents, with local tense/person/infinitive endings retained; clauses with additional copulas or progressive auxiliaries are excluded.'}]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   x=dict(record=r,citation=g['citation'],evidence=g['evidence'],family=i)
   if 'components' in g:x['components']=g['components']
   else:x.update(parent=g['parent'],kind=g.get('kind','reflex'))
   a.append(x)
assert len(a)==len({x['record']['ID'] for x in a})
h=[]
(P/'pass065-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass065-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
