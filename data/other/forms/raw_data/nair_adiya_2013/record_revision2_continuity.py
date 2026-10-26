"""Prove revised Adiya freeze changes only structured metadata tags."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE / "revision1"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
old=list(csv.reader((OLD/"proposed.csv").open(newline="")))
new=list(csv.reader((HERE/"proposed.csv").open(newline="")))
assert len(old)==len(new)==171
assert (OLD/"first-reading.jsonl").read_bytes()==(HERE/"first-reading.jsonl").read_bytes()
assert (OLD/"proposal-audit.jsonl").read_bytes()==(HERE/"proposal-audit.jsonl").read_bytes()
changed=[]
for before,after in zip(old,new):
    assert before[10]==after[10]
    delta=[i for i,(a,b) in enumerate(zip(before,after)) if a!=b]
    assert not delta or delta==[14],(before[10],delta)
    if delta: changed.append({"entry_key":before[10],"old_tags":before[14],"new_tags":after[14]})
result={"status":"pass; only source-backed Tags changed between review freezes",
        "old_csv_sha256":sha(OLD/"proposed.csv"),"new_csv_sha256":sha(HERE/"proposed.csv"),
        "first_reading_sha256":sha(HERE/"first-reading.jsonl"),
        "audit_sha256":sha(HERE/"proposal-audit.jsonl"),
        "old_freeze_sha256":sha(OLD/"review-freeze.json"),
        "new_freeze_sha256":sha(HERE/"review-freeze.json"),
        "unchanged_form_gloss_language_notes_citation_key_graph_fields":171,
        "changed_tag_rows":len(changed),"changes":changed,
        "reason":"Remove contextual 2pl from source-defined postposition okka, retain its postp class; preserve source-explicit demonstrative, nominative/oblique/stem, and proximal/distal categories in their attested grammatical cells.",
        "canonical_changes":False}
(HERE/"revision2-continuity.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(len(changed),"tag-only changes; lexical fields unchanged")
