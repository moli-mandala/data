import json
from pathlib import Path

base = Path(__file__).parent
records = {r['Form']: r for r in json.loads((base / 'andersen-records.json').read_text())}
rows = [
    ('adʰa', '644', 'reflex', 'andersen1990[p. 134 (printed p. 136)];CDIAL[644]',
     "Andersen's independently bounded adha- m. 'half' compares RV ardha- and Pali addha-, and points to aḍhatiya- and diyaḍhiya-. Turner 644 ardhá 'half' explicitly lists Ashokan aḍha- and northwestern adha forms."),
    ('adʰātiya', records['aḍʰatiya']['ID'], 'variant', 'andersen1990[p. 134 (printed p. 136)];CDIAL[651]',
     "Andersen's separate ad(h)ātiya- line explicitly says 'cf. aḍhatiya-', the adjacent 'two and a half' head. Turner 651 ardhatr̥tīya lists Ashokan aḍhātiya- and aḍhat°; the source parent already has accepted rank-1 CDIAL 651 ancestry in compiled edges."),
    ('adʰvikāni', records['adʰika']['ID'], 'variant', 'andersen1990[p. 135 (printed p. 137)];CDIAL[250]',
     "Andersen's independently listed Nittur adhvikāni says 'for adhikāni, cf. adhika-'; the adjacent adhika- head is glossed 'exceeding' and compares AVP adhika- and Pali adhika-. The installed source parent already has accepted rank-1 ancestry to Turner 250 adhika-."),
]
out={'accepted':[{'record':records[f], 'parent':p, 'kind':k, 'citation':c, 'evidence':e} for f,p,k,c,e in rows],
     'held':[{'record':records['atiyo'], 'reason':"Andersen compares Sanskrit atīva and Pali atīva/ativi(y)a for 'excessively', but no exact Turner article covering this Ashokan adverb was established; do not infer an etymon from the shared ati- sequence alone."}]}
(base/'pass123-decisions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
