"""Persist original-page owner checks, preserving immutable first-reading evidence."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
src=HERE/'texts-first-reading.jsonl'
rows=[json.loads(x) for x in src.read_text().splitlines()]
for r in rows:
    n=int(r['example'])
    r['review_status']='owner_original_page_reviewed'
    # Editorial punctuation belongs to raw evidence, not lexical token spelling.
    raw=' '.join(r['forms'])
    if n==44: raw=raw.replace('=kun ', '=kun, ')+'.'
    elif n==59: raw=raw.replace(' dɑn ', ' “dɑn ')+'.'
    elif n==60: raw+='”.'
    elif n not in (48,55,61): raw+='.'
    r['raw_surface']=raw
    r['raw_surface_policy']='Speech/sentence punctuation preserved here; aligned lexical forms retain all morphological = and - marks, omitting punctuation only.'
    if n==45:
        r['note']='Printed instruction: repeat once. Native text encodes final index as ipl.incl (letter i); visual small-cap character is ambiguous with numeral 1. Preserve encoded analysis and record typography uncertainty; source context and other glosses explicitly identify inclusive first-person plural.'
    r['text_provenance']='Composed by Bishakha Mallik; translated and analysed during the five-day Ranchi workshop. Default Odisha attribution follows p. 266. No village inferred.'
(HERE/'texts-reviewed.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
report=dict(status='owner_original_page_review_passed',scope='All 19 numbered Appendix 2 units (44–62), every aligned word and full translation; original PDF pages 38–40 printed 298–300 viewed separately at 1.8x.',units=len(rows),aligned_words=sum(len(r['forms']) for r in rows),input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),corrections=['Restored exact sentence/quotation punctuation in raw_surface; no lexical changes.'],uncertainties=['Example45 final grammatical glyph i/1 native encoding versus small-cap rendering.','Example46 exclusive independent pronoun and inclusive verb indices retained.','Example54 sleep interlinear omits FIN; source analysis retained.'],independent_review='pending',whole_source_completion=False)
(HERE/'owner-texts-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report))
