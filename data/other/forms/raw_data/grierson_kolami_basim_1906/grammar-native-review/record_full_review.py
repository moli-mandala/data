"""Reproduce independent keyed review; does not alter owner source/candidates."""
from pathlib import Path
from collections import Counter
import hashlib,json
HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/'grammar-review/reviewed-child-witnesses.jsonl'
PIN='ce78a459c1102d2b6e6376ea953f23dae4ebf9a5888daa63c9c72eae38c59ed2'
assert hashlib.sha256(INPUT.read_bytes()).hexdigest()==PIN
rows=[json.loads(s) for s in INPUT.read_text().splitlines()]
assert len(rows)==224 and len({r['source_unit_key'] for r in rows})==224
prefix='grierson1906kolamibasim:p'
changes={
'562:genitive-locative:1:independent:w8':('kīt','kit'),
'562:personal-pronouns:2:independent:w1':('annu(ṅg)','anu(ṅg)'),
'562:personal-pronouns:7:independent:w1':('īn','ïm'),
'562:present-copula:1:independent:w2':('āṇḍātīv','āṇḍātīvo'),
'562:past-copula:2:independent:w1':('āṇḍō','āṇḍīvo'),
'563:negative:1:independent:w5':('si-tōtīv','si-tōtīn'),
'566:pronunciation-l-n:1:independent:w7':('bāpam maṭṭen','bāpam muṭṭen'),
'566:pronunciation-n-nd-d:1:independent:w12':('evāñju','eañju'),
'566:nouns:1:independent:w8':('gaḍgākūn','gadyākūn'),
'566:personal-pronouns:2:independent:w1':('annuṅg','anuṅg'),
'566:personal-pronouns:3:independent:w2':('anned','amned'),
'566:personal-pronouns:5:independent:w1':('anmed','amned'),
'566:personal-pronouns:10:independent:w1':('īnmed','imned'),
'566:personal-pronouns:12:independent:w1':('avannuṅg','avanuṅg'),
'566:possessive-agreement:1:independent:w1':('anned','amned'),
'566:possessive-agreement:1:independent:w2':('inned','imned'),
'566:possessive-agreement:1:independent:w4':('innen bālā','imnen bālā'),
'566:other-pronouns:1:independent:w2':('adāv','adān'),
'566:other-pronouns:1:independent:w3':('inand','ivand'),
'566:present-copula:2:independent:w1':('āṇḍātīv','āṇḍātīvo'),
'567:beat-paradigm:1:independent:w1':('gōḷsātan','gōḷsātūn'),
'567:beat-paradigm:2:independent:w1':('gōḷsātiv','gōḷsātīvo'),
'567:beat-paradigm:4:independent:w1':('gōḷsātam','gōḷsātūm'),
'567:beat-paradigm:7:independent:w1':('goṭṭan','goṭṭān'),
'567:beat-paradigm:8:independent:w1':('goṭṭō','goṭṭīvo'),
'567:beat-paradigm:10:independent:w1':('goṭṭam','goṭṭām'),
'567:beat-paradigm:12:independent:w1':('goṭṭār','goṭṭēr'),
'567:beat-paradigm:13:independent:w1':('gōḷdātān','gōḷdātūn'),
'567:beat-paradigm:16:independent:w1':('gōḷdātvo','gōḷdātūm'),
'567:negative:1:independent:w1':('tōdā','tōdī'),
'567:negative:1:independent:w2':('masīten','murken'),
'567:negative:1:independent:w5':('sīyōṅg-tōdā','siyaṅg-tōdī'),
'567:conjunctive-participle:1:independent:w3':('vāttān','sāhādūn'),
'567:conjunctive-participle:1:independent:w4':('vaddan','saddan'),
'567:conjunctive-participle:1:independent:w5':('innekādān','innekādūn'),
}
holds={
'567:beat-paradigm:1:independent:w1':'u body certain; compact upper mark ū/ü/plain-u interpretation still requires root native comparison',
'567:beat-paradigm:4:independent:w1':'final um certain; upper u quantity/diacritic interpretation still requires native comparison',
'567:beat-paradigm:13:independent:w1':'final un certain, not ān; upper u quantity/diacritic interpretation pending',
'567:beat-paradigm:16:independent:w1':'final um certain, not vo; upper u quantity/diacritic interpretation pending',
'567:conjunctive-participle:1:independent:w3':'first source example begins sāhād, not vātt; final u upper mark pending narrow source typography adjudication',
'567:conjunctive-participle:1:independent:w5':'final u body replaces earlier ā proposal; upper u mark and medial consonant need final native corroboration',
'566:numerals:1:independent:w1':'nasal/quantity marks on five retained literally; no phonological interpretation inferred',
}
changes={prefix+k:v for k,v in changes.items()};holds={prefix+k:v for k,v in holds.items()}
lookup={r['source_unit_key']:r for r in rows}
assert set(changes)<=set(lookup) and set(holds)<=set(lookup)
for k,(old,new) in changes.items():assert lookup[k]['form']==old,(k,old,lookup[k]['form'])
manifest=json.loads((HERE/'selected-original-manifest.json').read_text())
for p in manifest['pages']:assert hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()==p['sha256']
originals={p['printed_page']:p for p in manifest['pages']}
labelkey=prefix+'566:present-copula:3:independent:w1'
assert lookup[labelkey]['printed_label']=='Sing. 2. m. present verb substantive'
reviews=[]
for r in rows:
 k=r['source_unit_key'];change=changes.get(k);uncertainty=holds.get(k)
 reviews.append({'source_unit_key':k,'physical_parent_key':r['physical_parent_key'],'language_label':r['language_label'],'old_form':r['form'],'proposed_form':change[1] if change else r['form'],'gloss':r['gloss'],'printed_label_before':r['printed_label'],'printed_label_proposed':'Sing. 3. m. present verb substantive' if k==labelkey else r['printed_label'],'disposition':r['proposed_disposition'],'independent_sharp_review':'literal_correction_proposed' if change else 'current_literal_supported','typed_review_uncertainty':uncertainty,'original_page':originals[r['printed_page']],'all_glosses_checked':'Printed lexical or functional gloss unchanged; source classifications not inferred from external morphology.'})
(HERE/'full224-sharp-review-proposal.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reviews))
corrections=[{'source_unit_key':k,'field':'form','old':old,'new':new,'status':'provisional_exact_literal_pending_quantity' if k in holds else 'independent_source_correction_proposal','typed_uncertainty':holds.get(k),'page':lookup[k]['printed_page']} for k,(old,new) in changes.items()]
corrections.append({'source_unit_key':labelkey,'field':'printed_label','old':lookup[labelkey]['printed_label'],'new':'Sing. 3. m. present verb substantive','status':'root_native_label_accepted','history':'Earlier low-resolution 2.m source-numbering conflict retained as superseded reading, not asserted sharp-original conflict.'})
status={'reviewer':'/root/asur_finish','status':'complete224_original_comparison_with_keyed_corrections_and_typed_quantity_holds','input_sha256':PIN,'counts':{'children_reviewed':224,'per_page':dict(Counter(r['printed_page'] for r in rows)),'form_correction_proposals':len(changes),'label_corrections':1,'remaining_scoped_review_uncertainties':len(holds)},'corrections':corrections,'all_current15_uncertain_reviewed':[k for k,r in lookup.items() if r.get('typed_uncertainties')],'hypothetical_avnd':'Remains audit_only_hypothesized_source_form; original must be derived claim retained, no emitted reconstructed form or donor/derived edge invented.','p566_amd':'Sharp pronunciation native explicitly prints amd; previous canonical and correction independently corroborated, one physical occurrence only.','rejected_provisional_diaeresis_readings':[{'key':prefix+'566:pronunciation-a-u:1:independent:w2','current_form':'ū','decision':'Preserved macron after narrower native control; earlier resized ü impression rejected by root.'},{'key':prefix+'563:conjunctive-participle:1:independent:w4','current_form':'sūlt-nā','decision':'Preserved contiguous macron after narrower native control; earlier resized ü impression rejected by root.'}],'acceptance_limits':['This is a separate source-keyed proposal. Owner first readings, canonical rows, tags/profile and graphs remain unchanged.','Compact final-u upper marks listed as typed holds require independent root corroboration, not harmonization to related forms.','No final combined-output acceptance or installation claim.'],'canonical_changes':False}
(HERE/'full224-sharp-review.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(status['counts']))
