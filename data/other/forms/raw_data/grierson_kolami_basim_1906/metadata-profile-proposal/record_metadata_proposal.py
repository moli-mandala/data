"""Write source-local lect proposal and pin metadata readiness; no installation."""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(ROOT / 'data'))
from dialects import dialect_tag
from source_meta import SourceMeta, yaml_files
from pybtex.database import parse_file
from pybtex import PybtexEngine

OUT = Path(__file__).parent
def pin(p):
    return {'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
    language_rows = {r['ID']:r for r in csv.DictReader((ROOT/'data/cldf/languages.csv').open())}
    current = list(csv.DictReader((ROOT/'data/cldf/dialects.csv').open()))
    proposed = []
    for base, code, district, name in [
        ('Kolami','Kolami-Wun-LSI1906','Wun','Kolami of Wun (LSI 1906)'),
        ('Naikri','Naikri-Basim-LSI1906','Basim','Bhili of Basim (LSI 1906)')]:
        row = {'ID':f'grierson1906_{base.lower()}_{district.lower()}',
               'Tag':dialect_tag(base,code,name),'Language_ID':base,'Source_Language_ID':code,
               'Name':name,'Glottocode':'','Latitude':'','Longitude':'',
               'Clade':language_rows[base]['Clade'],'Location':f'Historical {district} district, Berar; LSI 1906 survey', 'Quality':''}
        assert not any(r['ID']==row['ID'] or r['Tag']==row['Tag'] or r['Source_Language_ID']==code for r in current)
        assert not any(c.isspace() for c in row['Tag'])
        proposed.append(row)
    with (OUT/'dialects-proposed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(proposed[0]));writer.writeheader();writer.writerows(proposed)
    bib=(ROOT/'data/cldf/sources.bib').read_text()
    new_keys=re.findall(r'@\w+\s*\{\s*([^,]+)',(OUT/'references-proposed.bib').read_text())
    assert all(not re.search(r'@\w+\s*\{\s*'+re.escape(k)+r'\s*,',bib) for k in new_keys)
    yaml_files=list((ROOT/'data/data/other/forms').glob('*.yaml'))
    assert all(not any(re.search(r'^  '+re.escape(k)+r':\s*$',p.read_text(),re.M) for p in yaml_files) for k in new_keys)
    # Validate the draft with the actual schema at its eventual sibling filename.
    with tempfile.TemporaryDirectory(dir=OUT) as scratch:
        fixture=Path(scratch)/'20260927-grierson-kolami-basim.yaml'
        fixture.write_bytes((OUT/'settings-proposed.yaml').read_bytes())
        meta=SourceMeta([*__import__('source_meta').yaml_files(),fixture])
        for language in ('Kolami','Naikri'):
            assert meta.transcription('grierson1906kolamibasim','20260927-grierson-kolami-basim.csv',language)==('grierson-kolami-basim-1906',True)
        assert meta.flag('grierson1906kolamibasim','reference','ocr') is True
        assert meta.flag('haig1897gondi_kolami','reference','ocr') is False
    entries=parse_file(str(OUT/'references-proposed.bib')).entries
    formatted={key:PybtexEngine().format_from_string(entry.to_string('bibtex'),'plain',output_backend='markdown') for key,entry in entries.items()}
    assert all(value.strip() for value in formatted.values())
    result={'status':'metadata/profile proposal; not install-ready',
      'new_keys_no_current_bib_or_settings_collision':new_keys,
      'actual_SourceMeta_schema_and_combined_declaration_check':'pass; route names resolved, profile materialization remains pending',
      'actual_pybtex_plain_formatter':formatted,
      'existing_keys_reused_without_redeclaration':['hislop1866papers','dedr','grierson1906lsi4','grierson1906naiki'],
      'proposed_lects':proposed,
      'routing':[{'scope':'Specimen75 printed564–565 above Basim boundary','base':'Kolami','lect':proposed[0]['ID']},
        {'scope':'General Kolami grammar562–563','base':'Kolami','lect':None,'reason':'Synthesized from multiple supplied materials; not all explicitly Wun'},
        {'scope':'Basim subsection565 lower–569','base':'Naikri','lect':proposed[1]['ID']}],
      'auxiliary_policy':{'hislop1866papers':'Existing citation/settings only; reported source authority, no asserted direct lexical collation',
        'haig1897gondi_kolami':'New reported authority; printed562 plus DEDR section26 corroborates page range185–191; no lexical emission'},
      'ocr_attribution':'Main proposal assumes Nordic keyed OCR contributes transcription, with original owner/independent correction; final flag must reflect actual accepted path. Auxiliary identity was read from original/primary bibliography, not lexically OCR-ingested.',
      'unready_gates':['No complete lexical candidate or Unicode inventory yet','No source profile file yet','No guarded importer exists; no command declared','Bibliographic installed counts/provenance pending','No canonical/parser/full-build/browser claim'],
      'inputs':[pin(ROOT/'data/cldf/languages.csv'),pin(ROOT/'data/cldf/dialects.csv'),pin(ROOT/'data/cldf/sources.bib')],
      'proposal_files':[pin(OUT/n) for n in ['settings-proposed.yaml','references-proposed.bib','dialects-proposed.csv','profile-decisions.json','README.md']]}
    (OUT/'proposal-status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Two historical lect proposals; two collision-free new bibliography/settings keys; no canonical writes.')

if __name__=='__main__': main()
