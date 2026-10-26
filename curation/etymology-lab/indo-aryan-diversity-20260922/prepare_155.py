import json
from pathlib import Path

base = Path(__file__).parent
records = {r['Form']: r for r in json.loads((base / 'andersen-records.json').read_text())}
accepted = [
    {'record': records['yam'], 'parent': '10391', 'kind': 'reflex',
     'citation': 'andersen1990[p. 162 (printed p. 164), p. 163 (printed p. 165)];CDIAL[10391]',
     'evidence': "Andersen's yam 'since' head compares RV yad and Pali yad and explicitly refers to the following relative ya- pronoun paradigm, which prints -am. Turner 10391 yá relative pronoun lists neuter Sanskrit yád, Pali yaṁ (sandhi yad-), and Ashokan neuter yaṁ/ye/e; this is the source's conjunctive use of the neuter relative form."},
    {'record': records['am'], 'parent': records['yam']['ID'], 'kind': 'variant',
     'citation': 'andersen1990[p. 136 (printed p. 138), p. 162 (printed p. 164)];CDIAL[10391]',
     'evidence': "Andersen's am conjunction is glossed 'as, because, since', compares RV yad and Pali yam, records several inscriptions with am beside yam, and explicitly says 'cf. yam'. Link this source-marked reduced form to the same-function yam head and Turner 10391 relative family."},
    {'record': records['yate'], 'parent': records['yam']['ID'], 'kind': 'variant',
     'citation': 'andersen1990[p. 163 (printed p. 165)];CDIAL[10391]',
     'evidence': "Andersen's independently listed Panguraria yate says 'for yam, see ya-'. The adjacent ya- relative pronoun paradigm and separate yam 'since' head trace to Turner 10391; preserve the printed exceptional form as a source variant of the installed yam record."},
]
out = {'accepted':accepted,
       'held':[{'record':records['yata'], 'reason':"The installed yata record merges two printed heads, 'where' (< yatra) and 'travel, going on, proceeding' (< yātrā). Their etymologies must remain separate until the homograph boundary is corrected."}]}
(base/'pass155-decisions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
