"""Review-only Census Tamil Nadu Kota residual research; no accepted writes."""
import csv,gzip,json
from pathlib import Path
P=Path(__file__).resolve().parent;D=P.parents[2]
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==412
search=[]
for start in range(0,412,35):
 end=min(start+35,412)
 search+=json.loads((P/f'discovery-{start+1:03}-{end:03}-results.json').read_text())
assert len(search)==412 and all(x['count']==len(x['rows']) for x in search)
# Exact inventory index, parent whole DEDR article, and source/primary comparison.
groups=[
('d2929',[6],'DEDR 2929 gives Tamil neruppu “fire” and Kota nep “live coal”; Census nep “fire” is the Kota heat/fire member with broader elicitation gloss.'),
('d5474',[9,13],'DEDR 5474 explicitly gives Kota veṭm “mountain”; Census viṭəm is a close Kota mountain form entered separately under hill and mountain glosses.'),
('d4616',[23],'DEDR 4616 gives Kota mog “child” and Malayalam makavu “infant”; Census mug “baby” matches the child stem.'),
('d4813',[30],'DEDR 4813 directly gives Kota ma·mn, compared with Tamil māmaṉ “maternal uncle”; Census mamən is the same kin term.'),
('d4355',[51],'DEDR 4355 gives Kota pīc and Tamil pūcai “cat”; Census pis is a local cat form with short vowel transcription.'),
('d334',[54],'DEDR 334 gives Kota āv “cow” and Kannada āvu; Census ao retains the long-vowel cow term with a reduced glide.'),
('d1711',[61],'DEDR 1711 gives Kota kudyr and Kannada kudire “horse”; Census kudire matches this horse family.'),
('d1977',[70],'DEDR 1977 gives Kota keyv/kev and Kannada kivi “ear”; Census kiv belongs to that ear family.'),
('d2591',[66],'DEDR 2591 directly gives Kota rek “wing, feather”; Census rek “wing” is an exact form-and-sense match.'),
('d546',[77],'DEDR 546 gives Kota i·ruv and Malayalam īral “liver”; Census iru keeps the Kota liver stem without the final consonant.'),
('d4682',[75],'DEDR 4682 explicitly gives Kota maṇḍ and Kannada maṇḍe “head”; Census məṇḍu matches the head stem.'),
('d3633',[81],'DEDR 3633 gives Kota na·lg “tongue” and Kannada nālige; Census nalug preserves the tongue stem.'),
('d3948',[93],'DEDR 3948 directly gives Kota pay “house”; Census pai is the diphthongal house form.'),
('d4867',[99],'DEDR 4867 gives Kota meḷg “chilli” and Kodagu maḷu “chillies”; Census meluh retains the pepper/chilli stem.'),
('d486',[151],'DEDR 486 explicitly gives Kota ib “iron”; Census ibə is that iron noun with a final vowel.'),
('d990',[158],'DEDR 990 gives Kota oḏ “one”; Census oḍə preserves the dental numeral.'),
('d474',[159],'DEDR 474 gives Kota eyḏ “two”; Census iḍə is a source-vowel variant of the same two numeral.'),
('d5052',[160],'DEDR 5052 gives Kota mūnḏ “three”; Census munḍə retains the nasal-retroflex numeral.'),
('d3655',[161],'DEDR 3655 gives Kota nāng “four”; Census naŋgə retains the nasal-velar numeral.'),
('d2485',[162],'DEDR 2485 gives Kota ār “six”; Census aṛə is the rhotic six numeral.'),
('d910',[163],'DEDR 910 gives Kota ēy “seven”; Census ei is a source-vowel variant of the same numeral.'),
('d784',[164],'DEDR 784 gives Kota eṭ “eight”; Census eiṭ retains the dental eight numeral.'),
('d3918',[165],'DEDR 3918 gives Kota pat “ten”; Census pətte retains the ten stem.'),
('d3140',[202],'DEDR 3140 explicitly gives Kota kati·r- “to cut”; Census katir is the same cutting verb without the dictionary infinitival notation.'),
('d347',[219],'DEDR 347 gives Kota a·ṛ- “dance” and a·ṭ “dance, game”; Census āttum is a dance response with source verbal morphology.'),
('d2053',[226],'DEDR 2053 gives Kota koṛ- with past koṭ- “give”; Census kɔttu vutu retains the giving stem with an auxiliary in the elicited response.'),
('d3582',[238],'DEDR 3582 gives Kota naṛv-/naṛd- “walk”; Census nətutu vutu is an inflected walking response with a corresponding nasal-dental stem.'),
('d1628',[270],'DEDR 1628 gives Kota kuki·r-/kukrc- “sit down”; Census kukkursu vutu retains that reduplicated sitting stem with an auxiliary.'),
('d2017',[283],'DEDR 2017 gives Kota ke·ḷ- with past ke·ṭ- “hear”; Census kēttu vutu retains the past-stem dental before an auxiliary.'),
('d5160',[372],'DEDR 5160 gives Kota a·n “I”; Census ānu is the first-person singular stem.'),
('d410',[369],'DEDR 410 gives Kota i- proximal and i· “this place”; Census ivulə “here” preserves the proximal deictic base with locative material.'),
('d410',[374],'DEDR 410 gives Kota id “this thing”; Census id “to it/to this” is an oblique/case response of the proximal pronoun.'),
('d2559',[380],'DEDR 2559 gives Kota il- “not to exist” and Kannada illa “not”; Census ilā continues the negative base.'),
('d1',[382],'DEDR 1 gives Kota avr “those persons”; Census əvurukku “to them” is a case-marked distal plural pronoun.'),
('d1',[385],'DEDR 1 gives Kota ay “that place, there”; Census a i is its source-spaced locative form.'),
('d5154',[389],'DEDR 5154 gives Kota ām with em-/am- oblique “we”; Census əmmut “to us” preserves the first-person plural oblique.'),
('d5154',[390],'DEDR 5154 directly gives Kota a·m “we”; Census am matches the plural pronoun, without forcing inclusive/exclusive interpretation.'),
('d5151',[391],'DEDR 5151 gives Kota ed/edn/en for “what”; Census edin is an interrogative variant.'),
('d5151',[392],'DEDR 5151 contains temporal e- interrogatives including “when”; Census ēppɔduttul retains the epp- temporal base with extra source morphology.'),
('d5151',[394],'DEDR 5151 explicitly lists Kota dār “who”; Census dar is the matching person interrogative.'),
('d5151',[395],'DEDR 5151 gives Kota dār “who”; Census dard “whose” retains this interrogative under case marking.'),
('d5151',[396],'DEDR 5151 gives Kota dār “who”; Census darə “whom” is a case-marked person interrogative.'),
('d3684',[397],'DEDR 3684 gives Kota nī with nin- oblique “you” singular; Census nintul “to you” retains nin- with a case ending.'),
('d3688',[398],'DEDR 3688 gives Kota nīm with nim- oblique “you” plural; Census nimmut “your” retains nim- with possessive morphology.'),
]
proposals={}
for parent,idxs,evidence in groups:
 for n in idxs:
  assert n not in proposals;proposals[n]=([parent],evidence)
for n,parts,evidence in [
 (31,['d4813','d4616'],'Census mamən mol “maternal uncle’s daughter” combines Kota ma·mn “maternal uncle” in DEDR 4813 and Kota mo·ḷ “daughter” in DEDR 4616; both constituents are retained in source order.'),
 (74,['d4682','d4707'],'Census məṇdumir “hair” combines Kota maṇḍ “head” in DEDR 4682 and mi·r “hair” in DEDR 4707; this is a head-hair compound, not a simplex inherited stem.'),
]:
 assert n not in proposals;proposals[n]=(parts,evidence)

ids={p for parts,_ in proposals.values() for p in parts};articles={}
with gzip.open(D/'data/dedr/entry-texts-audit.csv.gz','rt') as f:
 for r in csv.DictReader(f):
  if r['Entry_ID'] in ids:articles[r['Entry_ID']]=r
assert set(articles)==ids
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')

accepted=[];held=[]
recent={'bus','school','student','teacher','bullet','mason','brass','money','silver','iron','cabbage','potato','tomato','ink','frock','sari','banian','loom','festival','divorce','heaven','hell','ancester'}
special_holds={
 156:'The same Census iḍavət is entered for both “second” and “third”; ordinal form and source identity need verification.',
 157:'The same Census iḍavət is entered for both “second” and “third”; the conflicting ordinal gloss must be resolved.',
 366:'On “he/she” lacks the consonants and gender contrast in DEDR 1 Kota avn/avḷ; a reduced-pronoun analysis is unverified.',
 367:'Ol “he/she” lacks the consonants and gender contrast in DEDR 1 Kota avn/avḷ; a reduced-pronoun analysis is unverified.',
 368:'Əvuḷukku resembles DEDR 1 Kota avḷ “that woman”, but the source gloss includes both “to her” and “to him”; gender/sense conflict remains.',
 370:'Əvunutu resembles DEDR 1 Kota avn “that man”, but the source gloss “his/her” merges gender; exact pronominal analysis remains open.',
 373:'Əddu is glossed “it/this”, mixing distal a- with proximal “this”; DEDR 1 and 410 cannot be chosen securely.',
 375:'Ətuna “its/of this” mixes distal-looking a- with proximal “this”; DEDR 1 versus 410 remains unresolved.',
 387:'Or “they” might be a reduced distal plural, but DEDR 1 Kota avr is not directly attested in this form.',
 393:'Evəḷ is glossed “where”, while DEDR 5151 Kota evḷ is “which woman”; the exact sense conflict blocks an automatic assignment.',
}
for n,(r,q) in enumerate(zip(records,search),1):
 if n in proposals:
  parts,evidence=proposals[n]
  x={'record':r,'parent':parts[0],'kind':'component' if len(parts)>1 else 'reflex',
     'citation':r['Source']+';'+','.join('dedr['+p[1:]+']' for p in parts),
     'evidence':evidence,'comparison':'Homepage exact search, full DEDR article and exact Census record checked; whole-article family identification.',
     'status':'pending-review'}
  if len(parts)>1:x['components']=parts
  accepted.append(x)
 else:
  form=r['Form'];gloss=r['Gloss'];g=gloss.lower()
  if n in special_holds:detail=special_holds[n]
  elif any(k in g for k in recent):detail='Cultural or contact vocabulary needs an immediate-donor and source-transcription check.'
  elif ' ' in form or any(s in form for s in ('vutu','butu','kuttu')):detail='The elicited response is multi-part or auxiliary-bearing; its lexical components and inflection need separate verification.'
  elif '/' in gloss or '?' in gloss:detail='The source gloss is disjunctive or qualified; a single parent sense cannot be fixed yet.'
  else:detail='No same-sense primary DEDR article has been verified for this exact transcription in this pass.'
  held.append({'record':r,'candidates':[],'reason':f'{form} “{gloss}”: {detail} Exact-form homepage search returned {q["count"]} row(s); follow-up comparison remains open.'})
(P/'proposed-decisions.json').write_text(json.dumps({'status':'pending-review','language':'Kota','source':'census2023tamilnadu','accepted':accepted,'held':held},ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(held,ensure_ascii=False,indent=2)+'\n')
print('proposed',len(accepted),'rows',sum(len(x.get('components',[x['parent']])) for x in accepted),'held',len(held),'articles',len(articles))
