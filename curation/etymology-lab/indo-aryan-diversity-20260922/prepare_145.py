import json
from pathlib import Path

base = Path(__file__).parent
records = {r['Form']: r for r in json.loads((base / 'andersen-records.json').read_text())}
accepted = [
    {'record': records['ah'], 'parent': '1538', 'kind': 'reflex',
     'citation': 'andersen1990[p. 138 (printed p. 140)];CDIAL[1538]',
     'evidence': "Andersen's ah- 'to say' head explicitly compares RV āha and Pali āha and lists the Ashokan third-person perfect āha/āpa series. Turner 1538 ā́ha 'says' lists the same Vedic and Pali word and Prakrit āhaṁsu/āhu; this is the saying verb, not āp- 'water'."},
    {'record': records['āpa'], 'parent': records['ah']['ID'], 'kind': 'variant',
     'citation': 'andersen1990[p. 138 (printed p. 140), p. 139 (printed p. 141)];CDIAL[1538]',
     'evidence': "Andersen's separate āpa line on printed p. 141 explicitly says 'for āha, see āha'; the preceding ah- 'to say' paradigm on printed p. 140 gives Udegolam āpa as its 3sg perfect variant. Turner 1538 anchors the āha family."},
    {'record': records['misibʰūta'], 'parent': records['misa']['ID'],
     'components': [records['misa']['ID'], records['hu']['ID']], 'kind': 'component',
     'citation': 'andersen1990[p. 162 (printed p. 164)];CDIAL[10135];CDIAL[9416]',
     'evidence': "Andersen prints misibhūta- 'mixed, coupled, united', compares Sanskrit miśrībhūta- and Pali missibhūta-, and explicitly cross-references misa- and hu-. Turner 10135 miśrá includes Ashokan misa- 'mixed'; Turner 9416 bhávati includes Ashokan hoti 'becomes'. Both installed source heads have accepted rank-1 ancestry, so retain the two printed lexical components."},
]
out = {'accepted':accepted, 'held':[]}
(base/'pass145-decisions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
