"""Apply the user's explicit cross-IA uncertainty policy to reviewed family matches."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
read=lambda f:json.loads((P/f).read_text())
excluded=set(read('scope-correction.json')['excludedIds'])
# Exact reason whitelist: no phonological, subsection, compound, or reading holds.
prefixes=('Retained lateral','The mango family','Banana kel','Jaunsari bhēs','Year word','CDIAL 10875','CDIAL 2871','CDIAL 6328','CDIAL 1670','Morning saber','The bhaṇṭā eggplant','CDIAL 8857','CDIAL 142','The rāmro good')
acc=[]
for x in read('audit-records.json'):
 if x['record']['ID'] in excluded or not x['reason'].startswith(prefixes):continue
 assert len(x['candidates'])==1
 q=x['candidates'][0];ev=q['evidence'];ev=' '.join(ev) if isinstance(ev,list) else ev
 # Preserve original reasoning verbatim as historical evidence, not current disposition.
 ev='Previously reviewed comparative evidence: '+ev+' Updated disposition under the user’s explicit cross-Indo-Aryan policy: link this supported etymon despite unresolved inheritance versus intra-Indo-Aryan borrowing. The reflex edge records etymological family affiliation provisionally; it does not establish uninterrupted inheritance or an immediate donor. Prior hold: '+x['reason']
 cite=q['citation'];cite='; '.join(cite) if isinstance(cite,list) else cite
 acc.append(dict(record=x['record'],parent=q['parent'],citation=cite,evidence=ev,family=q['parent'],kind='reflex',transmissionUncertain=True,priorHold=x['reason']))
(P/'contact-decisions.json').write_text(json.dumps(dict(accepted=acc,held=[]),ensure_ascii=False,indent=1)+'\n')
(P/'contact-policy.json').write_text(json.dumps(dict(authorization='in cases where you’re unsure if it’s a cross-IA borrowing, it’s preferred to link it anyways. this can’t be resolved with local data yet usually',application='Supported comparative family links may be saved provisionally with explicit transmission uncertainty. This does not waive etymon/subsection, transcription, morphology, or compound checks.',resolvedRecords=len(acc)),ensure_ascii=False,indent=2)+'\n')
s=(P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','contact-decisions.json').replace('sixth','contact')
(P/'contact_save.py').write_text(s)
print('Policy resolutions',len(acc))
