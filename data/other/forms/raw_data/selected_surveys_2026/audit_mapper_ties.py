import json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[5];sys.path.insert(0,str(R));from concepts import sense_candidates,_legacy_senses,NEVER_SPLIT
from pysem.glosses import Matcher,SPLITTER
D=R/'source_checklists/audits';d=json.loads((D/'20260911-selected-surveys-generated-diff.json').read_text());bygloss={r['gloss']:r for r in d['existing_forms_changed_memberships']};records=[]
for gloss,r in bygloss.items():
 candidates=[(t,p,NEVER_SPLIT) for t,p in sense_candidates(gloss)]+[(t,'','|'.join(SPLITTER)) for t in _legacy_senses(gloss)]
 possible=set();ties=[]
 for t,pos,splitter in dict.fromkeys(candidates):
  for text in dict.fromkeys([t,t.casefold(),t.capitalize(),t.upper()]):
   matches=Matcher(splitter=splitter).match(text,pos,100)
   if not matches:continue
   best=(matches[0].similarity,bool(matches[0].pos),matches[0].frequency)
   top=[m for m in matches if (m.similarity,bool(m.pos),m.frequency)==best and m.similarity>=3]
   possible.update(m.concepticon_id for m in top)
   if len({m.concepticon_id for m in top})>1:ties.append({'candidate':text,'pos':pos,'rank':list(best),'ids':sorted({m.concepticon_id for m in top})})
 difference=set(r['before'])^set(r['after']);records.append({'gloss':gloss,'changed_ids':sorted(difference),'not_explained_by_top_rank_ties':sorted(difference-possible),'ties':ties})
out={'pysem_version':'1.3.0','cause':'Matcher.match collects a set and truncates sorted results to one. Match.__lt__ compares similarity, presence of POS and frequency but does not break equal ranks by concept ID. Equal-ranked winners therefore depend on process hash seed.','changed_glosses':len(records),'unexplained':[r for r in records if r['not_explained_by_top_rank_ties']],'records':records}
(D/'20260911-selected-surveys-mapper-ties.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('glosses',len(records),'unexplained',len(out['unexplained']));print(json.dumps(out['unexplained'],ensure_ascii=False,indent=2))
