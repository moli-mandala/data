"""Independent original review and immutable first-reading closure checks."""
from pathlib import Path
import json,hashlib,collections
P=Path(__file__).resolve().parent;S=P.parent;ROOT=Path.cwd()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(name):return [json.loads(x) for x in (S/name).read_text().splitlines()]
freeze=json.loads((S/'first-reading-freeze.json').read_text())
assert sha(S/'first-reading-freeze.json')=='b6aeabbf7ac9e36172bbb493e6e6c1119352d541881a88364a8ff5233052e60a'
for name,h in freeze['files'].items():assert sha(S/name)==h,name
for page,r in freeze['originals'].items():assert sha(ROOT/r['path'])==r['sha256'],page
r=rows('roman-atoms-first-reading.jsonl');n=rows('native-atoms-first-reading.jsonl')
rl=rows('roman-physical-lines-first-reading.jsonl');nl=rows('native-physical-lines-first-reading.jsonl')
w=rows('whole-utterances-first-reading.jsonl');a=rows('script-alignment-proposal.jsonl');j=rows('physical-break-joins-proposal.jsonl')
assert (len(r),len(n),len(rl),len(nl),len(w),len(a),len(j))==(367,365,42,27,27,362,3)
rr={x['key']:x for x in r};nn={x['key']:x for x in n}
assert len(rr)==367 and len(nn)==365
assert collections.Counter(k for q in a for k in q['roman_keys'])==collections.Counter(rr.keys())
assert collections.Counter(k for q in a for k in q['native_keys'])==collections.Counter(nn.keys())
for q in j:
    assert len(q['physical_keys'])==2 and q['emission_key']==q['physical_keys'][0]
    assert ''.join(rr[k]['raw_form'] for k in q['physical_keys'])==q['source_form']
print('ALIGNMENT SCHEMA',json.dumps(a[0],ensure_ascii=False))
print('WHOLES')
for q in w:
    assert q['first_word_key']==q['roman_word_keys'][0] and q['last_word_key']==q['roman_word_keys'][-1]
    assert all(k in rr for k in q['roman_word_keys']) and all(k in nn for k in q['native_word_keys'])
    assert q['source_free_translation'] is None and q['lexical_free_gloss'] is None
    print(q['key'],q['proposed_whole_form'])
assert set(k for q in w for k in q['roman_word_keys'])==set(rr)
assert set(k for q in w for k in q['native_word_keys'])==set(nn)
corrections=[]
assert rr['lsi9.2:p235:aligned-line:5:word:1']['raw_form']=='hamĩ'
for c in corrections:assert rr[c['key']][c['field']]==c['old']
holds=[{'key':'lsi9.2:p235:aligned-line:2:word:3','field':'raw_form','old':'pīlᵃnē','decision':'retain typed ā/ī source-grapheme hold',
 'observation':'Lower first-vowel body favors a/ā to this reviewer; damage/low native resolution prevents an independently secure replacement. Native पाहने is not permission to normalize Roman.',
 'evidence':'root-p235-near-detail.png'},
 {'key':'lsi9.2:p234:aligned-line:15:word:8','field':'raw_gloss','old':'evil-aped','decision':'retain typed unresolved middle-cluster hold',
 'observation':'Visible terminal ed and evil- are secure; the fused middle cluster cannot be certified as ap or act from this witness. Do not repair semantically.',
 'evidence':'root-p234-evil-gloss-detail.png'}]
review=[]
for row in r+n:
    key=row['key'];review.append({'key':key,'source_script':row['source_script'],
      'exact_first_reading_guard':row['raw_form'] if 'raw_form' in row else row['raw_native'],
      'exact_gloss_guard':row.get('raw_gloss'),'source_image':row['source_image'],
      'source_image_sha256':row['source_image_sha256'],
      'decision':'keyed literal correction proposed' if any(c['key']==key for c in corrections) else
       'retain literal with typed source hold' if row.get('typed_uncertainties') else 'original literal/gloss scope agrees',
      'existing_typed_uncertainties_preserved':row.get('typed_uncertainties',[])})
(P/'all-732-physical-key-review.jsonl').write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in review))
(P/'keyed-corrections-proposal.json').write_text(json.dumps(corrections,ensure_ascii=False,indent=2)+'\n')
(P/'keyed-dispute-decisions.json').write_text(json.dumps(holds,ensure_ascii=False,indent=2)+'\n')
report={'reviewer':'/root/asur_finish','status':'independent original review complete; no material literal error established; explicitly unresolved source glyphs retained',
 'first_reading_freeze_sha256':sha(S/'first-reading-freeze.json'),'frozen_inputs':freeze['files'],'originals':freeze['originals'],
 'counts':{'Roman_atoms_compared':367,'Native_atoms_compared':365,'Roman_physical_lines':42,'Native_physical_lines':27,
 'whole_span_proposals_checked':27,'physical_break_joins_checked':3,'script_alignment_spans':362,'material_literal_corrections_proposed':len(corrections)},
 'method':'All native p233 and Roman p234–236 physical source lines were read against immutable source crops; every atom/form/gloss was compared. Narrow native-pixel controls checked disputed quantities and consonants. Original pages/crops, not owner text, are authority.',
 'corrections':corrections,'disputes':holds,
 'reviewer_diagnostic_history':'An interim report misread serialized TSV tab immediately before hamĩ as an initial t. Exact before-value guard rejected this; actual frozen atom/whole forms already hamĩ and original confirms it. No input or correction proposal was changed.',
 'other_source_decisions':['dīṇū̃ has a short horizontal quantity stroke over initial i here; preserve this specimen reading without harmonizing to grammar p51 diṇū̃.',
 'tarah-hī has a horizontal stroke over final i in the narrow control; no short-i correction.',
 'Roman khūryō and native खूयो remain separate source spellings; no cross-script normalization.',
 'Raised a and nasal signs, all three line-continuation joins, and the three native/Roman segmentation differences retain their distinct physical witnesses.',
 'Native सगको (two occurrences), क्यूंके (two), and तकी-नैं remain typed font/reading holds; no correction inferred from the aligned Roman strings.',
 'Native जिमीदारा-के explicitly differs from Roman jimīdār-kē; preserve source witnesses.',
 '27 whole spans are source punctuation/speech proposals, with explicit nested overlap; no author-supplied free translation or invented smooth English gloss. Three line-break continuations are not new attestations.'],
 'identity_recommendation':'Before final emission, use immutable first-physical-atom anchored expression identities and retain ordinal display sequence separately, as in the reviewed Kolami package; do not derive stable identity from punctuation or row order.',
 'scope_limits':['No installed/final-output acceptance, phonetic interpretation, canonical mutation or full-source completion claim.','Source ambiguities remain reviewed typed uncertainties, not certified glyphs.'],
 'review_outputs':{name:sha(P/name) for name in ['all-732-physical-key-review.jsonl','keyed-corrections-proposal.json','keyed-dispute-decisions.json','control-manifest.json']}}
(P/'independent-full-original-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('REPORT',sha(P/'independent-full-original-review.json'))
