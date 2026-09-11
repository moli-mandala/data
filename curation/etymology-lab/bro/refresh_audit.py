import json,pickle,re
from pathlib import Path
R=Path(__file__).resolve().parent;d=json.loads((R/'difficult-audit.json').read_text());pages={}
for n,page in enumerate(pickle.load(open(R.parents[2]/'data/cdial/cdial.pickle','rb')),1):
 for k in re.findall('<number>([^<]+)</number>',page):pages.setdefault(k,n)
l=['# Brokskat: unresolved source checks','','These are research holds, not proposed or accepted assignment rows. They remain in the unresolved count. The [primary grammar]('+d['sourceURL']+') was read in searchable text; its unreliable OCR prevents treating uncertain phonetic symbols as verified facsimile readings. CDIAL comparisons were checked in full local dictionary prose.','','| Brokskat | Candidates | Why unresolved |','|---|---|---|']
for q in d['cases']:
 c=q['candidates']
 for k in set(re.findall(r'\b\d{3,5}\b',c)):
  if k in pages:c=c.replace(k,'['+k+'](https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page='+str(pages[k])+')')
 l.append('| **'+', '.join(q['forms'])+'** ‘'+q['gloss']+'’ | '+c+' | '+q['evidence']+' |')
l+=['',f"{len(d['cases'])} unresolved cases cover {len({i for q in d['cases'] for i in q['formIds']})} records; zero assignment rows.",'','Grammar locators for the first seven cases: nominal comparisons, printed pp. 53–54; verbal inventory and historical groupings, pp. 85–86. Later bone and numeral cases cite CDIAL directly. No source or donor ingestion was performed.']
(R/'DIFFICULT-REVIEW.md').write_text('\n'.join(l)+'\n')
