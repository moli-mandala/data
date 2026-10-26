"""Pinned compact source-class comparison, without changing candidate metadata."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return [json.loads(s) for s in p.read_text().splitlines()]

def literal(s):
    return s.strip(' .;,?!‘’“”"').lower()

holds_path = HERE / "tense-voice-precision-review-holds.jsonl"
grammar_path = PACKAGE / "grammar/child-witnesses-reviewed.jsonl"
grammar = read(grammar_path)
packet = []
for hold in read(holds_path):
    matches = [g for g in grammar if literal(g["source_form"]) == literal(hold["exact_form"])]
    classes = [g for g in matches if any(w in g["source_analysis"] for w in ["past", "passive", "participle"])]
    packet.append(dict(**hold, exact_grammar_class_matches=classes,
                       recommendation="Retain only the specifically matched printed class, with occurrence scope checked." if classes else "Withhold intrinsic tense/voice tag; exact printed English meaning remains unchanged. No analogical ending/compound inference."))
(HERE / "precision-source-class-packet.jsonl").write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in packet))
with (HERE/"precision-source-class-packet.tsv").open('w', newline='') as stream:
    writer = csv.writer(stream, delimiter='\t')
    writer.writerow(['Key','Form','Gloss','Tags','Exact printed class match'])
    for r in packet:
        writer.writerow([r['key'],r['exact_form'],r['exact_gloss'],r['present_tags'], '; '.join(g['witness_key']+' '+g['source_analysis'] for g in r['exact_grammar_class_matches']) or 'None; no intrinsic class inferred'])
report = dict(rows=len(packet), distinct_keys=len({r['key'] for r in packet}), exact_class_matches=sum(bool(r['exact_grammar_class_matches']) for r in packet),
              inputs={str(p.relative_to(PACKAGE)):sha(p) for p in [holds_path,grammar_path]},
              limitations="Exact literal matching only; no generalization from suffix similarity, translation voice, or diṇū̃/dīṇū̃ typography. p237 explicitly classifies bōlyō, already reviewed separately.",
              outputs={p.name:sha(p) for p in [HERE/'precision-source-class-packet.jsonl',HERE/'precision-source-class-packet.tsv']})
(HERE/'precision-source-class-packet-status.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report))
