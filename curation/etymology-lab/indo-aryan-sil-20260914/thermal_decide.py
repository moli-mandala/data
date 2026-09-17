import csv,json,re
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914')
E={
'5679-2':('CDIAL[5679]','CDIAL 5679 explicitly identifies the ll-extension in Bengali tātal, Oriya tātalā/tātilā and Bhojpuri/Hindi tātal hot. The Bhil tātlo/tātlũ/tataḷo variants fit that extension rather than unextended tapta; local shortening and lateral realization are retained.'),
'5679':('CDIAL[5679]','CDIAL 5679 tapta explicitly gives Hindi tāta/tātā and Kashmiri totu hot. These t-bearing adjectives fit the unextended hot family, with Kullui vowel colouring retained.'),
'2389':('CDIAL[2389]','CDIAL 2389 uṣṇa explicitly includes Prakrit usiṇa, Kashmiri wuśun hot and western Gujarati ūnũ/Marathi ūn. The Bashkarik ūṣūn matches the attested Jambu úṣu᷄́n hot (gawri, f_i33dpbqp24rpe); Bhil unːu matches the western n-series. This does not resolve possible regional transmission.'),
'6811':('CDIAL[6811]','CDIAL 6811 *dhipp explicitly lists Maithili dhīpal warm and dhipāeb to warm. Magahi dhīpal hot matches that documented extended adjective; the expressive origin and possible influence of dīp are retained.'),
'5684':('CDIAL[5684]','Tāpyati offers a plausible base for tapla/təpla hot, but neither *ātapala nor simple tapta establishes this l-form. Verify the local participial morphology before assigning the complete response.'),
'6809':('CDIAL[6809]','Dhikṣate gives Hindi dhiknā be hot, but dhikal hot has an additional participial ending needing local morphological support.')}
sets={
'5679-2':{'tātlo','tātlũ','tataḷo','tatola'},
'5679':{'tatːi','t̪ot̪ə','t̪ʊt̪ə'},
'2389':{'ūṣūn','unːu'},
'6811':{'dʰīpal'},
'5684':{'təpla','tʌpla','təplo','topla','toplata','tāpālā','tāpīrī'},
'6809':{'dʰikʌl','dʰikala'}}
rs=list(csv.DictReader((P/'unresearched-records.csv').open()));acc=[];held=[];rules=[];ix={}
for r in rs:
 if not r['Gloss'].startswith('hot'):continue
 ks=[k for k,ws in sets.items() if r['Form'] in ws]
 if not ks:continue
 k=ks[0];cite,ev=E[k]
 if k not in ix:ix[k]=len(rules);rules.append(dict(parent=k,citation=cite,evidence=ev))
 i=ix[k]
 if k in {'5684','6809'}:held.append(dict(record=r,families=[i],reason=ev,passNumber=55));continue
 acc.append(dict(record=r,parent=k,family=i,citation=cite,evidence=ev+' Exact source form '+r['Form']+' is preserved.'))
a=json.loads((P/'thermal-primary-articles.json').read_text());a.update(json.loads((P/'thermal-followup-primary-articles.json').read_text()));(P/'thermal-primary-articles.json').write_text(json.dumps(a,ensure_ascii=False,indent=1))
(P/'thermal-rules.json').write_text(json.dumps(rules,ensure_ascii=False,indent=1));(P/'thermal-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'thermal_save.py').write_text((P/'week_mother_save.py').read_text().replace('week-mother','thermal'));(P/'thermal_decide.py').write_text(Path(__file__).read_text());print(len(acc),len(held))
