import json
from pathlib import Path

base = Path(__file__).parent
records = {r['Form']: r for r in json.loads((base / 'andersen-records.json').read_text())}
accepted = [
    {'record': records['sādʰika'], 'parent': records['adʰika']['ID'], 'kind': 'derived',
     'citation': 'andersen1990[p. 172 (printed p. 174)];CDIAL[250]',
     'evidence': "Andersen's sādhika- 'having excess or a surplus, excessive' head explicitly brackets its Sanskrit/Pali comparanda as [sa + adhika-] and cross-references adhika-. The installed adhika- 'exceeding' source head has accepted rank-1 Turner 250 ancestry. Preserve the source's synchronically transparent sa- formation as derived, rather than asserting direct reflex of bare adhika-."},
    {'record': records['amisamdevā'], 'parent': records['amisa']['ID'],
     'components': [records['amisa']['ID'], records['deva']['ID']], 'kind': 'component',
     'citation': 'andersen1990[p. 136 (printed p. 138), p. 152 (printed p. 154)];CDIAL[567];CDIAL[6523]',
     'evidence': "Andersen prints amisa- 'unmixed' with the compound aṁmisam-deva- and separately lists amisamdevā 'see amisa-'; the deva- 'god' head on printed p. 154 explicitly cross-references amisamdeva-. Turner 567 gives Ashokan amisā under ámiśra 'unmixed', and Turner 6523 gives Ashokan devā under dēvá 'god'. Both source heads already have accepted rank-1 ancestry; retain both lexical components."},
]
out = {'accepted':accepted,
       'held':[{'record':records['sātireka'], 'reason':"Andersen compares Sanskrit/Pali sātireka 'having excess', while Turner 212 contains bare atirēka 'surplus'. Unlike sādhika-, the source does not explicitly segment sa- + atireka or provide an installed atireka source head. Hold the proposed derivation until its exact status is reviewed."},
               {'record':records['sātirekekāni'], 'reason':"The printed cross-reference points to sātireka, whose own root assignment remains unresolved; hold this variant with its parent to avoid installing a terminal source-to-source link."}]}
(base/'pass160-decisions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
