import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[{'parent': '10999-2', 'gloss': ['wind'], 'forms': ['leheṛa'], 'languages': ['Bhatri'], 'citation': 'CDIAL[10999,2]', 'evidence': 'Turner places Oriya lahaṛi/lahaṛā wave in the second *lahaḍi branch, distinguished by retroflex ṛ from plain-r laharī. Bhatri leheṛa wind is a qualified semantic extension from wave/surge to gust; the recorded retroflex consonant supports this branch, while plain-r lehera is excluded.'}, {'parent': '11493', 'gloss': ['wind'], 'forms': ['baṭās'], 'languages': ['Dang'], 'citation': 'CDIAL[11493]', 'evidence': 'Turner explicitly records Nepali/Kumaoni batās and eastern/Hindi cognates wind under *vātatrāsa. Dang baṭās matches the complete regional wind noun; the survey retroflex spelling is retained.'}, {'parent': '5020', 'gloss': ['ash'], 'forms': ['cī'], 'languages': ['Phal'], 'citation': 'CDIAL[5020]', 'evidence': 'Turner explicitly lists Phalura chī and Bashkarik čī ashes under chādi covering, with the ash sense already attested in the Sanskrit compound. Survey Palula cī matches this regional ash noun, with aspiration unmarked in this response.'}]
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
for r in rs:
 if r['ID'] in done:continue
 reason=None
 if r['Gloss']=='river' and r['Form'] in ['khāḍi','kāṛi','khāḍ','khaḍ','kʰāḷi']:
  reason='Turner gives the creek/river family under both *khaḍḍa (3790) and *khāḍa (3860), explicitly cross-listing Marathi khāḍī as an alternative. The survey form does not resolve this competing reconstruction.'
 if r['Gloss']=='mud' and r['Form'] in ['kada','kadə','kado','kadoŋ']:
  reason='The bare kād- mud form does not securely separate karda (2867) from kardama (2869); Turner assigns similar eastern forms to both. No unique parent selected.'
 if r['Gloss']=='short' and r['Form'] in ['mandaro','mandrā','mandarā','maḍro']:
  reason='Turner distinguishes manthara (9754,4) and mandara (9754,6), with regional short adjectives in both. Aspiration loss/contact leaves the survey forms insufficiently diagnostic.'
 if r['Gloss']=='mud' and r['Form'] in ['xaṭa','xata','xatā']:
  reason='Zoller identifies the comparable Kohistani mud word with Pashto xaṭa. Existing Pashto xaṭa survey entries are still status unlinked, which the shared helper does not permit as parents; no new donor node or source ingestion in this no-rebuild task.'
 if reason:h.append(dict(record=r,reason=reason))
(P/'pass051-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass051-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
print(len(a),'accepted',len(h),'held',[(g.get('parent',g.get('components')),sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
