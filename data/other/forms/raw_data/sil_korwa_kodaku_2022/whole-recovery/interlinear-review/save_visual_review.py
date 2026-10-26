from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
observations={
93:'C2 n1–4. Source legend checked; n2 and n4 each contain two p/g blocks with one free translation.',
94:'C2 n5–11. n8 ends its own free translation with a comma; n9 remains a separate numbered expression. Source question marks retained.',
95:'C2 n12–18. n15 joins two p/g blocks under one free translation; grade discrepancies in source prose remain uncorrected.',
96:'C2 n19–25. n20 and n23 each have two phonetic blocks; n21 has three. Repeated and contradictory school-grade wording preserved.',
97:'C2 n26 and C3 n1–5. Section boundary and interposed English questions checked; C3 glottal-stop glyph retained.',
98:'C3 n6–11. n10 has a script-g marker, and a wrapped phonetic line; both correctly accounted.',
99:'C3 n12 and C4 n1–5. C4 n3 has p/g but no f; the following n4 translation is not copied backwards.',
100:'C4 n6–12. Wrapped p/g/f lines checked; source internal full stop in sa.b retained.',
101:'C4 n13–18. All six p/g/f groupings and five English question blocks checked.',
102:'C4 n19–24. n20 visibly has a plain g without backslash. Raw scaffold swallows this English line into p; keyed correction separates it in the proposal.',
103:'C4 n25–31. All seven numbered utterances, repeated clauses and English questions checked.',
104:'C4 n32–37. Agricultural narrative, wrapped translations, source phonetic length/nasal signs checked.',
105:'C4 n38–39 then C5 n1–6.1. C5 word-column layout begins here. n5.1/5.2 share translation; n6 continues across page. Question1 and its target answer independently printed.',
106:'C5 n6.2–12.1. Decimal subdivisions share whole translations. Questions2–4 and backslash-enclosed target answers checked. Internal span alignment in n6.1/n12.1 remains unresolved rather than forced.',
107:'C5 n12.2–17.2. Questions5–7, fourfold bidɛ answer, source pause label and all decimal group boundaries checked.',
108:'C5 n17.3–20.2. Cross-page n17 translation scope checked. Questions8–10 and their target answers checked. Explicit pause/emphasis labels retained as analysis, not lexical English glosses.'}
rows=[json.loads(x) for x in (H/'whole-expression-proposal.jsonl').read_text().splitlines()]
assert len(rows)==117
assert next(r for r in rows if r['source_unit_key'].endswith('C.4:expression:20'))['form']=='ar oh sɐmajĩ atʃʰa piʈai bʰi pɐre'
pages=[]
for n,observation in observations.items():
 rec=json.loads((H/f'render-{n}.json').read_text())
 pages.append(dict(pdf_page=n,printed_page=n-10,reviewed_render_sha256=rec['render_sha256'],source_sha256=rec['source_sha256'],observation=observation))
events=json.loads((H/'independent-physical-events.json').read_text())
holds=[]
for e in events:
 if e['marker']=='p' and e['section']!='C.5':
  holds.append(dict(physical_p_key=e['physical_key'],section=e['section'],printed_number=e['printed_number'],disposition='Whole expression retained; continuous-gloss word alignment requires explicit source-scoped reconciliation, not whitespace matching.'))
assert len(holds)==85
(H/'continuous-gloss-alignment-scope.json').write_text(json.dumps(dict(status='Explicit open word-alignment scope; not an exclusion of any target expression',physical_p_blocks=85,records=holds),ensure_ascii=False,indent=2)+'\n')
report=dict(status='Complete 16-page whole-expression and grouping review; word-column candidate prepared; independent review pending',reviewer='/root/asur_finish',scope='Every original page PDF93–108, all 126 phonetic blocks, 97 narrative expressions, ten target questions and ten target answers.',material_findings=[dict(key='behera2022korwakodaku:C.4:expression:20',type='bare-gloss-marker-recovery',before='English g line included in target form',after='Literal target form separated; English analysis retained; raw physical event unchanged',source='PDF102/printed92')],proposal_rows=117,physical_p_blocks=126,target_answers=10,c5_column_rows=192,c5_aligned_pairs=31,c5_unresolved_internal_spans=2,continuous_gloss_blocks_not_automatically_aligned=85,pages=pages,hashes={name:sha(H/name) for name in ['whole-expression-proposal.jsonl','independent-physical-events.json','c5-column-proposal.jsonl','c5-column-coverage.json','continuous-gloss-alignment-scope.json']},image_policy='Original PDF and reviewed-render hashes retained. One overwritten render is used because automatic approval review rejected accumulation of sixteen images under the disk constraint; parent acknowledged. No persistent image collection was generated.',limits=['Not canonical installation or source-stage acceptance.','No narrative sound conversion uses the older wordlist profile.','No word-level alignment is claimed for 85 continuous-gloss blocks.','Two C5 internal spans retain unallocated glosses rather than invented token mappings.'])
(H/'whole-original-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k in ['status','proposal_rows','c5_column_rows','hashes']},indent=2))
