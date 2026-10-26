"""Pin first-reading detail crops and non-mutating owner observations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
HERE = Path(__file__).resolve().parent
P = ROOT/'tmp/jambu-next-source/ahirwati-native-grammar-20260927'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    source=P/'printed-51-pdf-68-Im68.png'
    specs=[('nominals',(250,335,1990,920),1,'nominal/case spans'),
           ('pronouns',(250,915,1990,1540),1,'whole personal/own/demonstrative/hypothesis span'),
           ('copulas',(250,1535,1990,2180),1,'whole paradigm/past/regional/borrowing/Panjabi spans'),
           ('passive-detail',(1450,650,1810,780),4,'nearest-neighbor enlargement; raised a and underdot n'),
           ('own-detail',(325,1140,1660,1240),1,'MISLOCATED crop; excluded as evidence for own pronouns'),
           ('hypothesis-detail',(300,1410,1950,1540),1,'printed hypothetical obliques and separate kauṇ'),
           ('own-correct-span',(310,1050,1630,1150),1,'correct own-pronoun span')]
    crops=[]
    for name,bbox,scale,scope in specs:
        path=P/f'p51-first-reading-{name}.png'
        crops.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'source_path':str(source.relative_to(ROOT)),
                      'source_sha256':sha(source),'source_bbox_xyxy':bbox,'scale':scale,'scope':scope,
                      'used_as_own_pronoun_evidence':name in ['pronouns','own-correct-span']})
    report={'status':'owner crop provenance and later narrow observations; immutable first reading untouched',
            'crops':crops,'first_reading_sha256':sha(HERE/'child-witnesses-first-reading.jsonl'),
            'parent_coverage_sha256':sha(HERE/'parent-coverage-first-reading.jsonl'),
            'observations':[{'key':'lsi9.2:p51:qualified-oblique-relative-interrogative:w2',
                             'old_form':'kaun','proposed_form':'kauṇ',
                             'evidence_crop':'p51-first-reading-hypothesis-detail.png',
                             'reason':'Distinct dot under final n in native image; independently corroborated by root 2026-09-27. Apply only in subsequent reviewed derivative.'}],
            'field_semantics':{'source_form':'diplomatic Unicode first reading; no silent canonical conversion',
                               'source_quote':'owner evidence excerpt/context summary, not a certified verbatim transcription of all English prose',
                               'source_gloss':'printed translation assigned only where source provides it; paradigmatic analysis kept separately',
                               'qualification':'source caveat or explicit owner transcription review caveat'},
            'root_corroboration':'Root independently supports raised-a karᵃṇō/apᵃṇū/apᵃṇō/apᵃṇā, tū-nē̃ and nasal alternatives aĩh/ā̃h/waĩh/wā̃h/ū̃; root-authored review is separate acceptance evidence.',
            'script_sha256':sha(Path(__file__))}
    path=HERE/'first-reading-crop-provenance.json'
    raw=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode()
    if path.exists() and path.read_bytes()!=raw: raise SystemExit('Refusing historical overwrite')
    path.write_bytes(raw)
    print(json.dumps({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'crops':len(crops)}))
if __name__=='__main__': main()
