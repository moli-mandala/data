"""Conservative full-article Brahui shortlist; does not assign etymologies automatically."""
import json
import re
from pathlib import Path
P=Path(__file__).resolve().parent
candidates=json.loads((P/'dedr-brahui-form-candidates.json').read_text())
records=json.loads((P/'survey-inventory.json').read_text())
assert len(records)==3366
stop={'the','a','an','to','of','in','on','at','with','for','or','and','is','as','be','one','from','by','that','this','see','which','used','some','very','other','than','its','it','his','her','their','also','etc','into','like','vi','vt','adv','pron','num','int','cop','sg','imp'}
def tok(s):return set(re.findall(r'[a-z]{3,}',s.lower()))-stop
exclude={
    593:'DEDR itself offers an Indo-Aryan alternative for the tear verb; the immediate origin remains unresolved.',
    623:'Two different cached DEDR articles share number 2835 (dripping and pigeon); the parent identity must be resolved.',
    1616:'Ali and Kobayashi explicitly mark keb as Balochi, so a Dravidian reflex link would misstate the immediate donor.',
    1900:'Ali and Kobayashi explicitly mark māmā as Farsi, so a Dravidian reflex link would misstate the immediate donor.',
    2166:'nusxal handmill transparently contains xal stone; the compound needs component analysis.',
    2719:'The user asked that the earlier Gondi meat/DEDR 3373 issue remain untouched; defer this cross-source same-article lead too.',
    3217:'The DEDR wheat article itself offers an Indo-Aryan alternative; the donor pathway is unresolved.',
    749:'The source de has several senses including sun/day, so the pronoun who needs disambiguation.',
    2053:'The source nan has night and we as separate senses; the we comparison needs sense disambiguation.',
    1966:'The source mon has black and front/face senses; the front comparison needs sense disambiguation.',
    68:'The be verb matches several distinct DEDR Brahui articles; resolve the intended parent family.',
    431:'Transitive drop could align with either the descend or throw-down DEDR article; resolve the sense path.',
    1650:'The same Brahui abuse form appears under two DEDR articles (1590 and 1597).',
    3011:'The same Brahui moon form appears under DEDR 5496 and 3338.',
    3029:'The same Brahui home/house/wife form appears under DEDR 752 and 698.',
    3166:'The same Brahui bitter form appears under DEDR 1135 and 1466.',
}
manual_include={68,87,88,187,280,579,720,730,867,869,879,883,1214,1266,1286,1303,1593,2172,2180,2549,2892,3147,3151,3192}
selected=[];held=[]
for item in candidates:
    m=item['matches'][0];i=item['index'];r=records[i-1]
    if m[3]!=1 or len(m[2])!=1 or not m[2][0].startswith('d'):continue
    a=tok(item['gloss']);b=tok(m[1]);overlap=len(a&b)/(min(len(a),len(b)) or 1)
    contact=re.search(r'mark this entry \[[^]]+\]',r['Etymology'])
    if i in exclude or contact:
        reason=exclude.get(i) or 'The source explicitly marks a contact donor; resolve immediate borrowing before linking as a reflex.'
        held.append({'surveyIndex':i,'record':r,'reason':reason,'candidateParent':m[2][0],'dedrForm':m[0],'dedrGloss':m[1]})
    elif overlap>=0.5 or i in manual_include:
        selected.append({'surveyIndex':i,'record':r,'parent':m[2][0],'dedrForm':m[0],'dedrGloss':m[1], 'formScore':m[3], 'overlap':overlap})
assert not {x['surveyIndex'] for x in selected}&{x['surveyIndex'] for x in held}
assert all(records[x['surveyIndex']-1]['ID']==x['record']['ID'] for x in selected+held)
(P/'selection.json').write_text(json.dumps({'selected':selected,'held':held},ensure_ascii=False,indent=2)+'\n')
print({'selected':len(selected),'held':len(held),'unexamined':len(records)-len(selected)-len(held),'parents':len({x['parent'] for x in selected})})
