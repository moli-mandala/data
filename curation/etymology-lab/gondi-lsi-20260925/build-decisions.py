"""Stage Gondi LSI research decisions. No accepted-overlay writes."""
import csv,gzip,json,re
from pathlib import Path
P=Path(__file__).resolve().parent
A=json.load(open(P/'survey-inventory.json'))
# Source-ordered residual number -> (whole DEDR article, primary-evidence explanation).
M={
1:('d2826','DEDR quotes Gondi (Tr. W. Ph.) saiyung “five” beside SR sīyung; the LSI spelling sayuŋg is the same non-masculine numeral.'),
2:('d910','DEDR quotes Gondi (Tr. W.) yēṛung “seven”; LSI yēṛuŋg preserves the retroflex and final nasal.'),
3:('d3918','DEDR quotes Gondi (Tr. W. ChD.) pad “ten”, identical to this survey form.'),
4:('d3655','DEDR quotes Gondi (Tr.) nālung “four”, matching the LSI non-masculine numeral nāluŋg.'),
16:('d4756','DEDR lists Gondi (W. Ph.) māṛsāl “man” and Tr. mai-mansāl “man and wife”; LSI mansāl matches the article’s man family, though its initial nasal differs from the W. Ph. form.'),
17:('d400','DEDR quotes Gondi (Tr.) ār “woman”, identical to the survey form.'),
19:('d4764','DEDR quotes Gondi (Tr. W. Ph.) marrī “son”, identical to the survey form.'),
20:('d4616','DEDR quotes Gondi mīār (plural mīāhk) “daughter”; LSI miāhk is its plural response.'),
25:('d2216','DEDR quotes Gondi kōnḍā “bullock, ox”; LSI koṇḍā gives the same animal term.'),
26:('d2216','The LSI plural konḍāŋ retains the stem of DEDR Gondi kōnḍā “bullock, ox”; -ŋ is plural morphology, not a separate derivation.'),
29:('d3650','DEDR quotes Gondi nai/nāi/nāī “dog”; LSI naik adds the regular survey plural ending to that stem.'),
32:('d5152','DEDR quotes Gondi yēṭī “she-goat”, identical to the survey form.'),
33:('d5152','The LSI plural yēṭiŋg retains DEDR Gondi yēṭī “she-goat” with plural morphology.'),
44:('d4567','DEDR quotes Gondi parrō “on top”, identical to the LSI “up” response.'),
45:('d2584','DEDR quotes Gondi sir “under, beneath”; LSI sir “down” is the same spatial stem.'),
47:('d4205','DEDR quotes Gondi pijjā “after, behind (time or place)”, identical to the survey form and sense.'),
48:('d5151','DEDR quotes Gondi bōl/bōr “who, which man”; LSI bōl is exact.'),
49:('d5151','DEDR quotes Gondi bad as a feminine interrogative “who” and Mu. bad “what”; LSI bad “what?” belongs to this whole interrogative article.'),
50:('d5151','DEDR explicitly quotes Gondi (Tr.) bārī “why”, identical to the survey form.'),
55:('d2559','DEDR quotes Gondi hille “not, no”, identical to the survey response.'),
56:('d2781','DEDR quotes Gondi (Mu.) han- and (Tr. W.) handānā “to go”; LSI han is the same verbal stem.'),
59:('d5160','DEDR quotes Gondi (Tr.) anā “I”, identical to the survey pronoun.'),
60:('d5160','DEDR quotes emphatic Gondi annā “I”, identical to the survey alternate.'),
61:('d5160','DEDR quotes Gondi anā and oblique nā- “I”; LSI nāvōr “my” uses that possessive stem with an inflectional ending.'),
63:('d5154','DEDR quotes Gondi (Tr.) ammaṭ “we” (exclusive), identical to the survey response; the source does not specify clusivity.'),
65:('d5154','DEDR quotes oblique mā- beside Gondi ammaṭ “we”; LSI māvōr “our” is a possessive form of that stem.'),
66:('d3684','DEDR quotes Gondi immā with oblique nī- “thou”; LSI nīvōr “thy” preserves the possessive stem.'),
67:('d3688','DEDR quotes Gondi (Tr.) immaṭ “you” plural, identical to the survey response.'),
68:('d3675','DEDR quotes Gondi nillānā and nillā “stand”; LSI nill is the same verbal stem.'),
69:('d3675','DEDR quotes Gondi nittānā and nit- “stand”; LSI nitt is the same verbal stem.'),
70:('d2598','DEDR quotes Gondi sīānā “to give, allow”; LSI sī is the same verbal stem.'),
84:('d2602','DEDR quotes Gondi jīānā and Ch. jī- “to strike, beat”; LSI jī matches the attested root.'),
115:('d3688','DEDR quotes Gondi immaṭ with oblique mī- “you” plural; LSI mīvōr “your” preserves the possessive stem.'),
116:('d1','DEDR quotes Gondi (Ch.) ōl “he, that man”, identical to the LSI pronoun.'),
117:('d1','DEDR quotes the oblique ōn- of Gondi ōl “he”; LSI ōnōr “his” retains that stem with possessive morphology.'),
118:('d1','DEDR quotes Gondi plural ōṛ “those men”, identical to the LSI “they” response.'),
119:('d1','DEDR quotes Gondi plural ōṛk “those men”, identical to the LSI alternate “they”.'),
120:('d1','DEDR quotes Gondi plural ōṛ/ōṛk “they”; LSI ōṛōr “their” is its possessive form.'),
121:('d1','DEDR quotes Gondi plural ōṛk “they”; LSI ōṛknōr “their” is its possessive form.'),
122:('d4955','DEDR quotes Gondi (Tr.) muṛcul “the back”, identical to the LSI form.'),
129:('d4438','DEDR quotes Gondi (Tr.) pēn “god”, identical to the LSI form.'),
131:('d5159','DEDR quotes Gondi (Tr.) yēr “water”, identical to the survey form.'),
132:('d698','DEDR quotes Gondi (Tr. W. Ph. A.) rōn “house” under this whole article; it need not resemble the displayed head “inside”.'),
}
H={
**{i:'The father term is not the āpō/ta- family quoted in DEDR 156; the oblique and plural responses require a verified stem before whole-form ancestry.' for i in range(5,14)},
14:'The mother form dai is not directly represented in the shortlisted DEDR mother articles; possible Indic contact needs an immediate donor.',
15:'The mother form dāhṛī has no secure same-form Gondi DEDR parent or verified donor.',
18:'The child form cʰavvā lacks a direct DEDR comparison; do not equate it with the distinct koko/pilla children terms on gloss alone.',
**{i:'This horse/mare form or plural resembles regional Indo-Aryan vocabulary, but no supported immediate donor or Gondi DEDR parent has been identified.' for i in range(21,25)},
27:'DEDR 3136 explicitly puts Gondi ṭālī “cow” after a question mark; preserve the uncertainty, consistent with the earlier Gondi cow hold.',
28:'The “bitch” response siṭṭī has no supported same-form DEDR comparison or immediate donor.',
30:'The plural “bitches” response siṭṭiŋg cannot be assigned until the siṭṭī stem is established.',
31:'The he-goat response bokrāl may reflect an Indic goat term, but its immediate donor has not been established.',
**{i:'The deer form resembles Indo-Aryan hiran-; the direct donor and possible Gondi inflection remain unverified.' for i in range(34,37)},
**{i:'The good-person response contains cokkō plus a person stem, but the adjective parent is unresolved; a whole-response reflex would skip its component analysis.' for i in range(37,41)},
**{i:'The bad-boy/girl response is multiword, and būrtāl/tūṛ- components lack verified parents.' for i in (41,42)},
43:'uñcō “high” resembles Indo-Aryan ūñcā, but an immediate donor attestation has not been established.',
46:'The lak “far” search finds same-gloss Gondi forms but no linked primary parent or stable derivational analysis.',
51:'ānī “and” may be a regional conjunction loan; immediate donor and source pathway are unverified.',
52:'pē “but” has no secure primary parent in the database.',
53:'iŋgē “yes” has no secure primary parent in the database.',
54:'iŋgō “yes” has no secure primary parent; unrelated ṭiŋgō bells must not be used.',
57:'bīs “twenty” resembles an Indo-Aryan numeral, but the immediate donor is not verified.',
58:'pacās “fifty” resembles an Indo-Aryan numeral, but the immediate donor is not verified.',
62:'aplō “we” is potentially a distinct inclusive pronoun, but the DEDR inclusive/exclusive parent is not established.',
64:'aplōtōr “our” depends on unresolved aplō “we”.',
71:'vit “run” has no same-form DEDR Gondi run entry; unrelated vit- “sow” is explicitly a different sense.',
**{i:'This is a complete subject–predicate response. Its pronominal and verbal elements are attested separately, but the whole clause is not an inherited lexical stem; component assignment needs a supported scheme.' for i in range(72,84)},
**{i:'This is a complete inflected subject–verb clause. The jī- “beat” root is secure, but a whole-clause reflex would erase person/tense morphology and the separate subject.' for i in range(85,103)},
**{i:'This is a complete inflected subject–verb clause. The han- “go” root is secure, but the whole response requires component and inflection analysis.' for i in range(103,115)},
123:'DEDR 1096 quotes Gondi kaccī “iron” but explicitly questions Indo-Aryan origin; do not assign inherited reflex or skip to a remote donor.',
124:'sunnō “gold” lacks a same-form DEDR Gondi parent; possible Indic donor remains unverified.',
125:'cāndi “silver” likely belongs to regional Indo-Aryan vocabulary, but no immediate donor was verified.',
126:'gulām “slave” is a contact term; the immediate Gondi donor has not been verified.',
127:'kisān “cultivator” is a contact term; the immediate Gondi donor has not been verified.',
128:'gaṛriyāl “shepherd” is a contact term; the immediate Gondi donor has not been verified.',
130:'śaitān “devil” is a contact term; the immediate Gondi donor has not been verified.',
133:'kōṛā “horse” has no secure same-form primary parent and its possible Indic donor is unverified.',
134:'gadʰāl “ass” has no verified immediate donor or Gondi DEDR parent.',
135:'uṭṭum “camel” has no verified immediate donor or Gondi DEDR parent.',
136:'bōkā “cat” matches an unrelated Gondi bōkā “bone” under DEDR 4528; the cat ancestry remains unresolved.',
137:'gōgūṭal “cock” has no verified Gondi DEDR parent; any regional loan donor remains unresolved.',
138:'yētā kōrr “duck” is a multi-part water-bird phrase. DEDR 5159 supports oblique yēt- “water”, but kōrr is not securely analyzed.',
139:'suryāl “sun” resembles an Indo-Aryan sun term, but the immediate donor remains unverified.',
}
assert set(M).isdisjoint(H)
assert set(M)|set(H)==set(range(1,len(A)+1)),(set(range(1,len(A)+1))-(set(M)|set(H)))
ids={d for d,_ in M.values()};articles={}
with gzip.open('data/data/dedr/entry-texts-audit.csv.gz','rt') as f:
 for x in csv.DictReader(f):
  if x['Entry_ID'] in ids:articles[x['Entry_ID']]={k:x[k] for k in ['Entry_ID','Page','Citation','Raw_Article','Status']}
assert set(articles)==ids
accepted=[];held=[]
for i,r in enumerate(A,1):
 if i in M:
  parent,evidence=M[i]
  accepted.append(dict(record=r,parent=parent,kind='reflex',citation=r['Source']+';dedr['+parent[1:]+']',evidence=evidence,comparison='Full DEDR article, homepage Gondi comparanda and exact LSI source record checked.',status='pending-review'))
 else:held.append(dict(record=r,candidates=[],reason=H[i]))
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
(P/'proposed-decisions.json').write_text(json.dumps(dict(status='pending-review',accepted=accepted,held=held),ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
(P/'externally-assigned-records.json').write_text('[]\n')
(P/'unexamined-inventory.json').write_text('[]\n')
prior=json.load(open(P.parent/'dravidian-surveys-20260921'/'proposed-decisions.json'))
priorA=[x for x in prior['accepted'] if x['record']['Language_ID']=='Gondi' and 'grierson-lsi1928' in x['record']['Source']]
priorH=[x for x in prior['held'] if x['record']['Language_ID']=='Gondi' and 'grierson-lsi1928' in x['record']['Source']]
assert len(priorA)==39 and len(priorH)==4
allids={x['record']['ID'] for x in accepted+held+priorA+priorH};assert len(allids)==182
with open(P.parents[2]/'cldf/forms.csv') as f:
 compiledids={r['ID'] for r in csv.DictReader(f) if r['Language_ID']=='Gondi' and 'grierson-lsi1928' in r['Source']}
assert allids==compiledids,(len(allids),len(compiledids),len(allids^compiledids))
hold_ids={x['record']['ID'] for x in priorH};assert not hold_ids.intersection(r['ID'] for r in A)
coverage=dict(source='grierson-lsi1928',language='Gondi',totalCompiledSourceRecords=182,priorPendingProposals=len(priorA),priorResearchHolds=len(priorH),newPendingProposals=len(accepted),newProposedAssignmentRows=len(accepted),newResearchHolds=len(held),unexamined=0,exactRecordPreservation=True,uniquePassOwnership=True,alreadyRank1AssignedAmongPassRecords=0,preservedEarlierQuestionedHolds=['ant4169','meat3373','southbroom4794'])
(P/'inventory-counts.json').write_text(json.dumps(coverage,indent=2)+'\n')
(P/'record-coverage-001.json').write_text(json.dumps(coverage,indent=2)+'\n')
print(json.dumps({'proposed':len(accepted),'holds':len(held),'articles':len(articles),'prior':len(priorA)+len(priorH)}))
