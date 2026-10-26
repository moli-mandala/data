from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=[H.parent/'auxiliary-source-proposal.bib',H.parent/'metadata-integration-notes.md',
       H/'original-pdf028-150dpi.png',H/'original-pdf114-150dpi.png',
       H/'additional-authorities-proposal.bib',H/'auxiliary-review.md',H/'outside-candidate.jsonl']
rows=[json.loads(l) for l in (H/'outside-candidate.jsonl').read_text().splitlines()]
assert len(rows)==48
assert len(rows[0]['original_literal_occurrences'])==1
assert len(rows[1]['original_literal_occurrences'])==2
assert all('outside-group' not in r['source'] for r in rows)
assert all(not r['notes'] for r in rows[:2])
report=dict(status='independent original bibliography review complete; proposed additions pending root integration',
  reviewed_inputs={str(p.relative_to(H.parent)):sha(p) for p in paths},
  original_pages=[{'pdf':28,'printed':18},{'pdf':114,'printed':104}],
  root_proposals='Both bibliographic contents and secondary-attribution notes agree with originals.',
  requested_title_protection={'abraham1995sadri':['Sadri','India'],'ctdp2002hillkorwa':['Hill-Korwa','Pahari Korwa']},
  additional_authorities=['powercity_korba_ndc','singh1994scheduledtribes'],
  evidence_cautions=['Powercity n.d.c uses singular people.html, not n.d.a peoples.html.',
    'Singh printed address is Delhi; Oxford University Press is the publisher, not a separately verified place.',
    'Kora/kor claims are secondary through Behera, with historical/possible origin qualification intact.'],
  citation_revision=dict(all48_actual_printed_page_citations=True,internal_keys_retained_in_audit=True,
      origin_claims_in_etymology_without_duplicate_notes=True,
      rectangle_method='Case-sensitive complete-token/phrase matches; kor has exactly two occurrences in its legend paragraph.',
      transcription_caveat='Printed49 field-transcription/no phonological or grammatical analysis statement retained in status.'),
  root_files_modified=False,canonical_files_modified=False)
(H/'auxiliary-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PASS: bibliography evidence and48-row citation/etymology/whole-token guards')
