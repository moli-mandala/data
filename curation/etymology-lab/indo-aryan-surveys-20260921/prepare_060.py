import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[{'parent': '587', 'gloss': ['this'], 'forms': ['ʔī̃', 'ʔaī̃', 'ʔāī', 'ʔāī̃', 'āī'], 'languages': ['Bshk'], 'citation': 'CDIAL[587]', 'evidence': 'Turner explicitly lists Bashkarik äyĩ this under ayam, alongside the comparable Middle Indo-Aryan ayaṁ/iaṁ forms. These glottal-onset and vowel variants match that directly attested Bashkarik form; they are not assigned simply by similarity to Hindi yah.'}, {'parent': '587', 'gloss': ['this', 'these'], 'forms': ['eju'], 'languages': ['jaun'], 'citation': 'CDIAL[587]', 'evidence': 'Turner specifically places Jaunsari ejo this in the re-emphasized *iyya family under ayam. Survey eju matches that local form; both singular and plural elicitation uses are retained as the source records them.'}, {'parent': '587', 'gloss': ['this'], 'forms': ['jo', 'jā', 'ji', 'jaū', 'jū', 'jɛ'], 'languages': ['bundeli_atarra', 'H'], 'citation': 'CDIAL[587]', 'evidence': 'Turner gives Braj jao and Sodoci jɔ in the strengthened-y demonstrative family under ayam. These Bundeli/Hindi survey j-initial this forms match that central Indo-Aryan series. Turner explicitly acknowledges historical mixing with eṣa; the assignment follows his treatment of strengthened j, not an exclusive unmixed paradigm.'}, {'parent': '5889', 'gloss': ['you'], 'forms': ['t ū', 'tūī'], 'citation': 'CDIAL[5889,1 and addenda]', 'evidence': 'Turner gives Hindi/Marathi tū and Assamese tui among nominative reflexes of tuvam. The source-spaced t ū represents the same single pronoun, not a compound; Tharu tūī is compared to the documented emphatic tui type. Source transcription is preserved.'}]
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
(P/'pass060-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass060-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
