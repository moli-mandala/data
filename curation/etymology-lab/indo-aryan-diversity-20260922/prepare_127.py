import json
from pathlib import Path

base = Path(__file__).parent
records = {r['Form']: r for r in json.loads((base / 'andersen-records.json').read_text())}
rows = [
    ('pāvate', records['pap']['ID'], 'variant', 'andersen1990[p. 157 (printed p. 159)];CDIAL[8947]',
     "Andersen's Sahasram pāvate entry explicitly says 'see pāp-' and the preceding pāp- head cites Sahasram pāvatave as its dative infinitive variant of pāpotave 'to obtain, get, gain'. Turner 8947 prā́pnōti directly lists Ashokan ru. br. sd. pāpotave inf.; the source parent already has accepted rank-1 8947 ancestry."),
    ('rajuka', records['rajūka']['ID'], 'variant', 'andersen1990[p. 164 (printed p. 166)];CDIAL[10691]',
     "Andersen's Nittur rajuka- line explicitly reads 'see rajūka-'. Turner 10691 *rājjuka 'surveyor' directly lists Ashokan rajuka- and rājūka- among the regional settlement-officer forms; the source parent already has accepted rank-1 10691 ancestry."),
    ('savara', records['savata']['ID'], 'variant', 'andersen1990[p. 172 (printed p. 174)];CDIAL[13278]',
     "Andersen's Rupnath savara entry explicitly reads 'for savata, see savata'. The adjacent savata head means 'everywhere', compares Sanskrit sarvatra and Pali sabbattha, and notes Rupnath savara; Turner 13278 sarvátra is the source parent's accepted rank-1 family."),
]
out = {'accepted':[{'record':records[f], 'parent':p, 'kind':k, 'citation':c, 'evidence':e} for f,p,k,c,e in rows],
       'held':[{'record':records['hevaha'], 'reason':"The printed hevaha cross-reference is dual, 'see hevaṁ and ah-', implying more than a simple spelling variant of hevaṁ; hold until its word formation is represented without collapsing the two elements."}]}
(base/'pass127-decisions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
