import csv,json,re,collections
from pathlib import Path
P=Path(__file__).resolve().parent
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
langs={'RathwiBareli','Bhilali','Bhili','PauriBareli','Nimadi','Khandesi','Noiri','DungraBhili','Goj','Vasavi'}
words=set('purio puria puriyo puryā puryo puryũ poryu poryõ poryā poryo pore puriu puryu poriya purai purāi porāi pure pori pora purāy porə porai poreo porei poray porey poiro puiro puyiro poriyo poiri poyaro poirõ poriu poyari'.split())
glosses={'child','boy','girl','son','daughter'}
ev='CDIAL 8399.2 *potara explicitly gives Gujarati porī little girl and poriyo boy, and Marathi por child or young animal. These western child terms are analyzed jointly in that r-bearing family rather than bare pota or putra. Gendered and y-bearing endings are compared with Gujarati porī/poriyo; pur- beside por- and pory-/poir-/poyar- are working vowel/glide correspondences inferred from the survey series, not independently established sound laws. Those variants remain flagged for audit, and intra-Indo-Aryan transmission is open. The article qualifies the deeper non-Aryan origin of the family.'
rule=dict(parent='8399-2',citation='CDIAL[8399.2]',evidence=ev,words=sorted(words),languages=sorted(langs),glosses=sorted(glosses));acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Language_ID'] not in langs or r['Form'] not in words:continue
 gs={s.strip().lower() for s in r['Gloss'].split(';')}
 if gs and gs<=glosses:acc.append(dict(record=r,parent='8399-2',family=0,kind='reflex',citation=rule['citation'],evidence=ev+' Exact survey form '+r['Form']+' and sense '+r['Gloss']+' are preserved.'))
assert acc and len({x['record']['ID'] for x in acc})==len(acc)
(P/'child-r-rules.json').write_text(json.dumps([rule],ensure_ascii=False,indent=1)+'\n');(P/'child-r-decisions.json').write_text(json.dumps(dict(accepted=acc,held=[]),ensure_ascii=False,indent=1)+'\n')
# Preserve the exact location-tagged series so the more difficult alternations can be reviewed.
by=collections.defaultdict(list)
for x in acc:
 r=x['record'];by[r['Language_ID']+'|'+r['Tags']].append({k:r[k] for k in ['ID','Form','Gloss','Source']})
(P/'child-r-locality-series.json').write_text(json.dumps(by,ensure_ascii=False,indent=1)+'\n')
(P/'child-r-comparative-audit.md').write_text('# Western child terms: accepted with comparative qualifications\n\n'+ev+'\n\nThe exact locality-tagged records are preserved in `child-r-locality-series.json`. No claim is made that a single contact route or phonological rule has been demonstrated. Responses containing an additional word or a second alternative were not included.\n\nAffected records: '+str(len(acc))+'\n')
(P/'child_r_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth','child-r'))
print(len(acc),collections.Counter(x['record']['Language_ID'] for x in acc))
