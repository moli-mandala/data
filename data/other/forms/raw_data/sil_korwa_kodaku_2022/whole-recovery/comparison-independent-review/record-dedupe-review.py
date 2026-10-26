"""Read-only static review of Behera's first and final compile dedupe keys."""
import collections,csv,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent; B=H.parent; W=next(p for p in H.parents if (p/'data/make_cldf.py').exists())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
legacy=B/'legacy-review/legacy-repaired-candidate.csv'; additions=B/'integration-proposal/all-additions.csv'
paths=[legacy,additions,B/'routing-proposal/20260828-sil-korwa-kodaku.yaml',B/'routing-proposal/20260927-behera-whole-additions.yaml',W/'data/make_cldf.py',W/'data/source_meta.py',B/'check_transcription_routing.py',B/'check_comparison_parser.py',W/'data/tests/test_behera_whole_recovery.py']
rows=list(csv.reader(legacy.open()))+list(csv.reader(additions.open()))
assert len(rows)==6864 and all(len(r)==15 for r in rows)
keys=[r[10] for r in rows];assert len(set(keys))==6864
citations=collections.Counter(r[7].split(';',1)[0].split('[',1)[0] for r in rows)
assert citations=={'behera2022korwakodaku':6864},citations
base=collections.defaultdict(list)
for r in rows:base[(r[0],r[1],r[2])].append((r[10],r[3]))
collisions={k:v for k,v in base.items() if len(v)>1}
different_gloss=sum(len({g for _,g in v})>1 for v in collisions.values())
report=dict(status='static-and-scoped-parser-policy-pass',inputs={str(p.relative_to(W)):sha(p) for p in paths},counts=dict(rows=6864,legacy_rows=4459,new_rows=2405,unique_source_entry_keys=6864,source_citations=dict(citations),unkeyed_triples=len(base),colliding_triples=len(collisions),rows_in_colliding_triples=sum(map(len,collisions.values())),colliding_triples_with_distinct_glosses=different_gloss),finding='The source citation maps to the explicit true source-level identity override. make_cldf.source_entry_dedupe_key therefore contributes each unique Entry_Key to its first-pass (language, parameter, form, source-key, special-gloss) tuple; all6864 rows remain distinct. The final done tuple includes Entry_Key because row.formatted[1:] spans Entry_Key. Initial merge_redundant_strand_rows is restricted to Strand/Strand3, so this source is unaffected.',scope=['SourceMeta.flag gives the citation-level true value precedence over the new file default false. The new additions YAML has no source declaration, preventing duplicate citation ownership while retaining the source-level override.','check_comparison_parser.py exercises only 2405 additions through actual NFC/NFD parse; test_behera_whole_recovery.py forms Row objects for all6864 and asserts exact initial dedupe tuples. This is complementary coverage, not a compiled end-to-end proof.','Cross-source first-pass collisions with a different citation and no own key are possible under generic compiler policy; Behera rows retain a nonempty unique source key, preventing such merging on their side. However no full multi-source compile or durable-ID/graph parity was run.','Second-pass done uses complete formatted fields except ID, including Entry_Key. A different-source record with an identical Entry_Key and all other fields could collide; globally unique keys are expected, but cross-source global-key closure awaits full build.'],risks=[])
dest=H/'dedupe-policy-review.json';dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report['counts'],indent=2));print(sha(dest))
