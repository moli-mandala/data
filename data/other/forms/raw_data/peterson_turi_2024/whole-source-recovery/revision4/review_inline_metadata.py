"""Keyed source-scoped metadata, never grammatical features mined from prose."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
decisions={}
def put(keys,tags,basis):
 for key in keys.split():
  assert key not in decisions,key
  decisions[key]=(tags.split(),basis)
put('dual','affix du','p267 explicitly labels =kin dual.')
put('plural plural-marker','affix pl','pp267–268 explicitly identify plural markers.')
put('both','quantifier du','p268 printed both=du.')
put('we-both','pron first-person du exclusive multiword-expression','p268 explicitly we both (excl), with dual form.')
put('we-pl','pron first-person pl 1pl exclusive','p268 explicit we (pl.excl).')
put('historical-four four five grierson-numerals-1 grierson-numerals-2','num','Explicit numeral discussion/table4 and historical numeral quotation.')
put('one-article','num determiner indef','p268 explicitly one used as an indefinite article.')
put('count-marker','affix uncertain','p268 calls this a counting morpheme, tentatively glossed count; rejects established classifier analysis.')
put('objective objective-obligation','affix obj','pp269/276 explicitly objective case marker, not person/number of its host.')
put('genitive-pronoun-1 genitive-pronoun-2 genitive-other','affix gen','p269 explicit genitive markers; pronoun-host conditioning is not marker POS.')
put('possessive','affix poss','p269 inalienable possessive marker.')
put('father','noun kinship','p269 possessive father example and lexical kinship relation.')
put('locative','affix loc','p270 Locative heading labels the case marker.')
put('instrumental','affix instr','p270 Instrumental heading labels the case marker.')
put('to-monkeys','pl obl','p270 explicitly prints monkey=pl=obl, distinct from obj analysis in example4.')
put('of-children','pl gen','p270 explicitly prints child=pl=gen.')
put('finite finite-underlying finite-surface finite-raised jh-finite shared-finite hortative-finite','affix','Finite marker only. Following subject-index person/number is conditioning context, not a feature of this vowel.')
put('third-index-1','affix third-person sg 3sg obj','p271 explicitly Odisha =i object index.')
put('third-index-2 third-subject','affix third-person sg 3sg subj','p271 explicitly Odisha =ɛ subject index.')
put('third-index-jh','affix third-person sg 3sg subj obj','p271 explicitly Jharkhandi =i indexes both subject and object.')
put('first-indices-1 first-indices-2','affix first-person sg 1sg subj obj','p271 table5 and prose identify the first-singular free-variant argument indices.')
put('jh-object','affix third-person sg 3sg animate obj','p271 example12 explicitly 3sg.anim.obj in hit predicate; the finite marker is separate.')
put('past-perfect-1 past-perfect-2','affix pret perfect','p272 identifies past-perfect markers.')
put('past-middle-1 past-middle-2 past-middle-following past-middle-alone jashpur-past','affix pret middle','p272 explicitly middle-voice past marker, including footnote Jashpur witness.')
put('middle-1 middle-2 middle-final','affix middle','p272 explicitly middle marker; adjacent past morphology is separate.')
put('middle-surface-short','affix middle pret','p272 explicitly combined surface form after elision of ɛ from middle+past sequence.')
put('jashpur-fruitful','verb middle pret','p272 footnote9 explicitly fruit-y-mid-pst.mid=fin. The full inflected word is not an affix.')
put('past-imperfective past-imperfective-repeat','affix pret ipfv','p273 explicitly past imperfective; not a generic perfective.')
put('simple-past simple-past-repeat','affix pret','p273 explicitly pure past, not perfective past.')
put('negative negative-underlying negative-surface','neg negator','p273 preverbal negator. Third-person singular animate describes following =ɛ, not ka/kɑ/kə.')
put('third-person','affix third-person sg 3sg animate subj','p273 explicitly third-singular animate subject index on the negator.')
put('give-inline','verb pret third-person animate obj pl 3pl','p274 exact printed give-pst.act=3anim.obj=fin=3pl; no unprinted singular object inference.')
put('recipient-index','affix third-person animate obj','p274 =i indexes animate G at object position, explicitly illustrated by 3anim.obj.')
put('children','pl','p274 explicitly ʧhaʋa=kun the children; source number marker =kun.')
put('bellies','obj','pp274–275 exact lahiʔɲ=kɛ with belly=obj. English plural does not establish plural morphology.')
put('sar-index','affix third-person obj uncertain','p275 source says =i refers to plural sons despite apparently singular form; no definite sg/pl tag assigned.')
put('sar-give','verb pret third-person obj sg 3sg subj','p275 explicit give-appl.pst=3.obj-y-fin=3sg.subj. Object number remains unspecified in the printed analysis.')
put('loc-copula copula-derivation shared-copula locative-copula','verb copula stem','pp275–276 hɛn- is the locative copula stem taking argument/TAM morphology; the trailing hyphen alone does not make it an affix.')
put('ident-copula identity','affix copula','pp275/277 explicitly enclitic identity copula =naŋ.')
put('stay ranchi-stay','verb','p275 explicitly stay/remain as full lexical verb.')
put('jh-stay','verb copula','p275 explicitly related Jharkhandi copula tai.')
put('ranchi-past-cop','verb copula stem pret','p275 explicitly Ranchi past-tense copula stem tahi-.')
put('ranchi-present-cop','verb copula stem pres','p275 explicitly Ranchi present-tense copula stem hen-.')
put('inanimate-index','affix third-person sg 3sg inanimate obj','p276 example28 explicitly 3sg.inan.obj; semantic subject at the copula object-index slot, not a claim that the finite vowel has these features.')
put('negative-copula','verb copula neg pres','p276 full suppletive negative-present copula kanɔʔɔ, not an affix.')
put('imperative-components-1 hortative-middle','affix middle','pp277–278 middle marker component only; following person index separate.')
put('imperative-components-2','affix second-person sg 2sg','p277 explicitly second-person singular marker -m.')
put('imperative-combined-1 imperative-combined-2','affix impv middle second-person sg 2sg','p277 combined imperative-middle-second-singular endings.')
put('prohibitive','prohibitive part','p278 preverbal prohibitive alu hosts a subject index; no affix classification asserted.')
put('sequential','affix conjunctive-participle','p278 sequential converb marker =kɛtɛ.')
put('simultaneous','affix conjunctive-participle ipfv','pp278–279 imperfective converb marker -tɛ; reduplication belongs to preceding stem.')
put('hiatus','epenthetic','p290 general hiatus-breaking notation, no locality or inflectional affix claim.')
put('sit-1 sit-2 sit-3','verb','p267 sitting predicate examples, preserved phonetic variants.')
put('hats','noun','p267 explicitly calls the unmarked target a noun; no plural morphology inferred.')

def main():
 p=HERE/'inline-reviewed.jsonl'
 units=list(map(json.loads,p.read_text().splitlines()))
 emitted=[];seen=set()
 for u in units:
  if u['kind'] not in ('target','morpheme','epenthetic'):continue
  tail=u['source_unit_key'].split(':inline:')[-1]
  ts,basis=decisions.get(tail,([], 'No additional explicit grammatical classification asserted for this lexical quotation; retain its literal/gloss/lect and contextual Notes.'))
  seen.add(tail)
  u['structured_tags']=ts
  u['structured_metadata_basis']=basis
  if tail=='to-monkeys':u['printed_analysis']='monkey=pl=obl'
  if tail=='of-children':u['printed_analysis']='child=pl=gen'
  if tail=='jashpur-fruitful':u['printed_analysis']='fruit-y-mid-pst.mid=fin'
  emitted.append({'source_unit_key':u['source_unit_key'],'form':u['form'],'tags':ts,'basis':basis,'printed_analysis':u.get('printed_analysis','')})
 assert set(decisions)<=seen,set(decisions)-seen
 p.write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in units))
 report={'status':'owner_source_scoped_review','all_inline_units':len(units),'emitted_units_reviewed':len(emitted),'metadata_policy':'No tags inferred from unrestricted English gloss/Notes tokens. Explicit per-key analysis controls inline metadata; context describing the following index cannot transfer to finite/negative markers.','units':emitted}
 (HERE/'inline-metadata-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(len(emitted),'inline emitted units scoped')
if __name__=='__main__':main()
