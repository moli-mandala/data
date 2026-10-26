"""Pin an independent full-page first-reading review without mutating owner inputs."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=B/'physical-units-first-reading.jsonl';rows=[json.loads(s) for s in source.read_text().splitlines()]
assert len(rows)==32 and sha(source)=='ddcf9dd7a6e6cb53ee5030f94487216a323ab0d3adb2639461c4a6656ef8da7b'
images={n:sha(B/f'original-review/page-{n}.png') for n in (19,20,21)}
assert images=={19:'8e8ff27afdb9a7f5240795d23a0692d9f629809e5841e85c8d321e728e7912bb',20:'87081d4135ac64154d8d6f4272c46d448f69f7ec5cb5c8491ca1540308756c05',21:'f6fbb8ebe586321fd8e68e727f70ff98866bc2ac90f13eba5cb82e493e50ef11'}
findings={
'nair2013cholanaickan:p20:feminine-markers':dict(severity='material-source-scope',field='source_forms/scope',old=['-itti','-i','-e','-ci','-atti','-iti','-ø','-tti','-cci'],proposed_target=['-itti','-i','-e','-ci','-atti','-iti','-ø'],control_only=['-tti','-cci'],evidence='Original p20 first sentence explicitly enumerates target feminine markers only through -ø. Following sentences call -tti reconstructed South Dravidian and -cci retained by Tamil, Kodagu and Kannada; they are comparison claims, not Cholanaikkan forms.'),
'nair2013cholanaickan:p20:vocative-markers':dict(severity='emission-scope',field='source_forms/scope',old=['-ā','-e','-ī','-ē','-annu'],proposed_vocative=['-ā','-e','-ī','-ē'],context_only=['-annu'],evidence='Original p20 says -annu is a noun stem ending conditioning the first vocative variant, not a vocative suffix. The owner note already records this, but the same source_forms array risks treating it as one.'),
'nair2013cholanaickan:p19:instrumental-markers':dict(severity='typed-source-typography-hold',field='source_claim',evidence='Original p19 states free variation while both printed renderings appear indistinguishable -indu at inspected resolution. Do not synthesize a contrasting vowel length or duplicate lexical headword solely from the prose.'),
}
reviews=[]
for r in rows:
    key=r['unit_key'];f=findings.get(key)
    reviews.append(dict(unit_key=key,printed_page=r['printed_page'],pdf_page=r['pdf_page'],form_mentions=len(r['source_forms']),scope=r['scope'],literal_reading='visually-agrees-provisionally',gloss='agrees-with-source-or-not-provided',scope_review='correction-proposed' if f and f['severity'] in ('material-source-scope','emission-scope') else 'agrees-or-context-retained',finding=f))
report=dict(status='first-reading-reviewed-with-two-scope-corrections-and-one-typography-hold',inputs=dict(first_reading_sha256=sha(source),page_image_sha256=images,pdf_sha256='4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af'),coverage=dict(physical_units=32,all_unit_keys_unique=len({r['unit_key'] for r in rows})==32,form_mentions=sum(len(r['source_forms']) for r in rows),page_counts={str(n):sum(r['pdf_page']==n for r in rows) for n in (19,20,21)}),reviews=reviews,important_boundaries=['Three ethnonyms are names of the people; the page does not explicitly attest them as words of the language. Retain documentary status.','Tamil va+nt+aann came is an explicit comparison control, not a Cholanaikkan target.','Bibliography is a cited authority, not lexical evidence.','-annu is a stem-ending condition and -tti/-cci are cross-language comparisons, not independent target emissions.'],literal_typography_limit='The retained page images support the stated ordinary literal readings, but tiny underline/dot-like marks on italic examples need original-font or higher magnification adjudication before final literal acceptance. This report does not convert those marks into phonetic claims.',canonical_changes=False)
out=H/'independent-first-reading-review.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(report['status'],report['coverage'],sha(out))
