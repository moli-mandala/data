import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[{'parent': '5071', 'gloss': ['short', 'small', 'short (thing)'], 'forms': ['cʰot', 'coto', 'cūtā', 'cotoī', 'coṭī', 'coto / cuṭo', 'cota', 'cotia', 'coṭ', 'coṭlū̃', 'chotū', 'tsūṭḷa', 'cʰɔt', 'cɔṭ', 'coṭṭā', 'coṭo', 'coṭā', 'choṭalo', 'coṭi', 'tsot̪o', 'tsuṭla', 'tshuṭli', 'cʰoṭi', 'cʰoṭ', 'cotā', 'cūtī', 'cʰotke', 'tsʰotke', 'cotkā', 'ṭśʰot'], 'citation': 'CDIAL[5071 and addenda]', 'evidence': 'Turner records Lahnda/Punjabi/Western Pahari choṭā/choṭṭā, Kumaoni/Nepali choṭo, Bengali/Oriya choṭa and Gujarati choṭũ small/short under *chōṭṭa. These simple survey forms share its affricate-retroflex stem, with regional aspiration, vowel and diminutive-ending variation retained. Mixed alternatives, small-big echo compounds and Kullui ocə forms are excluded; the comparative family does not by itself settle recent Hindi influence.'}, {'parent': '12732', 'gloss': ['small', 'short'], 'forms': ['nānlo', 'nānlū', 'ṇāṇlo', 'nanīū', 'nanlo', 'nānkɔ', 'nānkū̃', 'nānko', 'nālo', 'nenū', 'nani', 'san', 'nānso', 'ṇāṇlu', 'nānkā', 'nenlū', 'hāṇo', 'hānu', 'nahanā'], 'citation': 'CDIAL[12732]', 'evidence': 'Turner records Gujarati nānũ/nāhnũ, Old Gujarati nānhūṁ, Oriya sāna and Marathi sāna/lahān small under ślakṣṇa. The western survey nān-/hān- forms with k/l diminutive extensions and the eastern san/nani forms fit this established small-size family; hān- reflects the regional s/h correspondence, while nasal or lateral suffixes are preserved. Forms with a medial d cluster are excluded from this rule.'}, {'parent': '3895', 'gloss': ['short'], 'forms': ['kʰātʰā̃õ', 'kʰāṭʰõ', 'kʰāṭʰā̃õ', 'kʰūtʰū', 'kʰāṭõ', 'kʰaṭan', 'kɛṭan', 'kaṭan', 'kaṭané', 'kaṭānõ', 'kaṭaṇ'], 'languages': ['Mai', 'Chil', 'Gowro', 'bhatr', 'Bshk', 'Kalk', 'Gaw', 'Phal', 'Tor'], 'citation': 'CDIAL[3895];zoller2005[p. 120]', 'evidence': 'Zoller explicitly derives Kohistani khaṭā̃v short from *khuṭṭa⁴ (CDIAL 3895), comparing Shina khuṭo and Kalam khäṭän. The survey Kohistani/Gawri forms and neighbouring Palula kaṭāno match this specific adjective family, including the nasal adjectival extension. Deaspirated and front-vowel variants remain qualified comparisons; the less transparent khaṛan and cɛṭan responses are excluded.'}, {'parent': '6935', 'gloss': ['short'], 'forms': ['nāṭo'], 'languages': ['Nimadi'], 'citation': 'CDIAL[6935,1]', 'evidence': 'Turner gives Hindi nāṭā short/dwarfish and Punjabi nāṭā dwarf in the first *naṭṭa branch. Nimadi nāṭo short continues that exact stem with its normal masculine ending, not the aspirated or nasal-cluster branches.'}]
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
(P/'pass049-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass049-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
