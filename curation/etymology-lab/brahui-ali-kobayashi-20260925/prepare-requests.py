"""Stage exact-form searches only for manually shortlisted Brahui records."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
selection=json.loads((P/'selection.json').read_text())
items=sorted(selection['selected']+selection['held'],key=lambda x:x['surveyIndex'])
for start in range(0,len(items),35):
    rows=items[start:start+35]
    requests=[{'mode':'lexicon','languageId':'Brahui','params':{'form':x['record']['Form'],'relaxed':True}} for x in rows]
    (P/f'discovery-{start+1:03d}-{start+len(rows):03d}-requests.json').write_text(json.dumps(requests,ensure_ascii=False,indent=2)+'\n')
print({'requests':len(items),'chunks':(len(items)+34)//35})
