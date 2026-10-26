import json
from pathlib import Path

base = Path(__file__).parent
records = {r['Form']: r for r in json.loads((base / 'andersen-records.json').read_text())}
rows = [
    ('āṇapayati', '1096', 'reflex', 'andersen1990[p. 138 (printed p. 140), p. 139 (printed p. 141)];CDIAL[1096]',
     "Andersen's āṇapayati cross-reference on printed p. 140 points to the following āṇapay- head, glossed 'to give an order, to command' and compared with Sanskrit ājñāpayati and Pali āṇāpeti. Turner 1096 ājñāpayati explicitly lists Ashokan āñapayāmi, aṇapemi, and ānapayisati."),
    ('ārādʰ', '1316', 'reflex', 'andersen1990[p. 139 (printed p. 141)];CDIAL[1316]',
     "Andersen glosses ārādh- 'to please', causative 'to please, win favour', compares Sanskrit ārādhnoti and Pali ārādheti, and prints ārādhetave/ārodhetave/ārādhayitave. Turner 1316 ārādhayati lists Pali ārādheti and Ashokan aradheti, ārādhayaṁtu, ālādhayitu in the causative branch."),
    ('ārodʰetave', records['ārādʰ']['ID'], 'variant', 'andersen1990[p. 139 (printed p. 141)];CDIAL[1316]',
     "Andersen's independently listed Rupnath ārodhetave says 'see ārādh-' and the preceding verb head identifies it as a Rupnath form of ārādhetave, dative infinitive causative. Turner 1316 supports the causative family."),
    ('pavat', '8788', 'reflex', 'andersen1990[p. 156 (printed p. 158)];CDIAL[8788]',
     "Andersen explicitly glosses the causative pavat- 'to practice [dhammaguna]', compares Sanskrit pra + √vart and Pali pavattati/pavatteti, and prints pavatitaviye/pavatitaviyā. Turner 8788 pravartayati, branch 3, gives Ashokan pavatayevū and pavatitaviyā 'to perform'. This is the verb, distinct from the following pavata- 'mountain'."),
    ('pā', records['pi']['ID'], 'variant', 'andersen1990[p. 156 (printed p. 158)];CDIAL[485]',
     "Andersen's Rupnath pā line explicitly reads 'for pi, see pi'. The separate pi head is glossed 'also' and already links to Turner 485 ápi, whose article explicitly lists Ashokan pi."),
]
accepted = [{'record': records[form], 'parent': parent, 'kind': kind, 'citation': citation, 'evidence': evidence}
            for form, parent, kind, citation, evidence in rows]
out = {'accepted': accepted, 'held': []}
(base / 'pass119-decisions.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
