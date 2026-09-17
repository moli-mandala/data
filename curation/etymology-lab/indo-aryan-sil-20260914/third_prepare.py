"""Explicit third-pass candidate rules. Research only; no overlay writes."""
import csv,json,re,unicodedata,collections
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
src=(P/'prepare.py').read_text();exec('def norm'+src.split('def norm',1)[1].split('families=json.loads')[0])
# Additional glyph equivalences; quantities and nasalization are discovery-only folds.
oldnorm=norm
def norm(s):return oldnorm(s.replace('ɡ','g').replace('ǰ','j').replace('č','c'))
qs=[]
def add(parent,gloss,words,ev,locator=None):qs.append(dict(parent=parent,gloss=gloss,words=words.split('|'),evidence=ev,citation='CDIAL['+(locator or parent.replace('-','.',1))+']'))
add('700','different|separate','alag|alagg|alaga|alago|alga|algā|aḷag|aḷgu','CDIAL 700 alagna gives Pali alagga “not joined”, Hindi alag/algā, Old Marwari alago and regional alag forms “separate”. This supplies the modest separate/different semantic change; the link is to this negative formation, not to a superficially similar word for another.')
add('9875-2','chili|chilli|chillies|pepper','mirc|mirci|mirca|mirce|mircu|miric|mirica|mirac|marc|marca|marci|marcu|marac|maric|marica|marici|moric|moricā','CDIAL 9875.2 *maricca explicitly gives Bhojpuri maricā “chillies”, Awadhi mircā, Punjabi mirc/marc and Gujarati marcī/marcũ “red pepper”. Retained c selects this strengthened branch, distinct from marīca > miri and the marucca branch. Capsicum is a later semantic extension, not an ancient plant identification.')
add('6459','door','duar|duara|duari|duvar|duvari|duwar|duwara|duwari|duor','CDIAL 6459 *duvāra gives Prakrit duāra/duvāra, Punjabi duār, Nepali duwār and eastern duār/duāra “door”. The explicit du-vowel sequence selects the expanded branch; bare dar/dor and door-frame compounds are excluded because their precise history requires separate analysis.')
add('4992','roof','chan|chana|chani|chanhi|chānhi|chann|channi','CDIAL 4992 *channi gives Punjabi chann “thatched roof”, Bihari chānh/chānhī/chānhiyā and Old Awadhi chāna. This supports the nasal roof family; the stop-bearing chat family and compound responses are kept separate.')
add('13720-2','few|a few|little','thoṛa|thoṛo|thoṛi|thoṛe|thoḍa|thoḍo|thoḍu|thora|thoro|thori|thore|thoḷa','CDIAL 13720 explicitly lists the historical -ḍ- extension: Hindi thoṛā, Marwari thoṛo, Gujarati thoṛũ, Marathi thoḍā, Nepali thor and eastern thor/thoṛā “few, a little”. Jambu’s *stōkaḍ- node represents this extension rather than bare stōka; it is not the unrelated tree-trunk word *thuḍa.', '13720, -ḍ- extension')
add('138-2x','ring|finger ring','aŋguṭhi|aṅguṭhi|aŋguthi|aṅguthi|aŋguṭi|aṅguṭi|aŋṭhi|aṅṭhi|aŋthi|aṅthi|auṭhi|authi','CDIAL 138.2 *aṅguṣṭhiya gives Punjabi aṅgūṭhī, Nepali aũṭhi, Hindi ãgūṭhī and Marathi ãgṭhī “ring”. This selects the -iya formation rather than aṅguṣṭha “thumb”. Dental-t spellings and nasal loss are reviewed separately; Persian aṅguštarī-shaped responses are not collapsed into this branch.','138.2')
add('6446','wife|bride','dulhin|dulhan|dulahini|dulahi|dulhani','CDIAL 6446 derives feminine dulahini/dūlhin/dulhan “bride” in its durlabha > dullaha family. The bride-to-wife semantic use is compatible, but the survey-specific meaning is retained and the relation identifies this historical feminine formation; it does not assign masculine bridegroom forms to the same elicited meaning.')
add('9963','woman|wife|woman; wife','mehararu|mehraru|mahiraru|mihiraru|mihraru|mahraru|meheraru|meharua|mehrua','CDIAL 9963 *mahilārūpa explicitly lists Bihari mehrārū, Maithili/Bhojpuri meharārū and Awadhi meharuā/meheruā “woman, wife”. These extended forms select this compound, rather than the shorter mahilā family. The reconstruction of its first member remains qualified by the broader dictionary discussion.')
add('3515','millet','kodo|kodõ|kodon|kodua|kodawa|kodaw|kodrā|kodri','CDIAL 3515 kodrava gives Hindi/Bengali/Bihari kodo, Nepali kodo and Oriya kodua as grain names. The article applies them to different millet species, so the survey’s broad “millet” gloss is retained. Punjabi kodo and Hindi/Western Pahari kodrā routes explicitly marked as loans require separate donor analysis.')
add('9201','millet','bajra|bajri|bajṛa|bajṛi|bajara|bajari|bajaro|bajuro','CDIAL 9201 *bājjara lists Hindi bājrā/bājṛā, Punjabi bājrā, Gujarati bājrī and Marathi bājrā for millet. The link identifies this comparative grain-name family; the addendum criticizes a proposed deeper reconstruction, and no particular botanical species is imposed on the survey gloss.')
add('11503','eggplant|brinjal','baigan|baigon|baigaṇ|baingaṇ|baiṅgaṇ|baingan|baiŋgan|baīgan|baiŋgaṇ|baigun|baigana|baiŋgaṇa|bāiṅgaṇa','CDIAL 11503.1 vātiṅgaṇa lists Prakrit vāiṃgaṇa, Hindi baigan/baĩgun, Nepali baigan and Oriya bāiṅgaṇa “eggplant”. The diphthong-bearing forms select section 1; bare baṅga and bāgun belong to distinct branches, and the article explicitly identifies an Iranian route for some northwestern forms.','11503.1')
add('10434','millet','janer|janera|jonhri|jonhri|jondri|jondari|jondṛi|jondṛa|jondara|junhār|junri|juneri','CDIAL 10434 yavanāla lists Bihari janer/jonhrī/jõdhrī, Hindi junhār/jundrī and Marathi jõdhḷā for regional grain names. The broad “millet” elicitation is retained; the source also covers maize and other grains, so these links do not resolve the botanical identity.')
add('9369','eggplant|brinjal','bhaṇṭa|bhanṭa|bhaṭa|bhaṭṭa|bhaṭṭā|bhāṭā','CDIAL 9369.1 bhaṇṭākī gives Bihari bhaṇṭā and Bengali/Maithili/Awadhi bhā̃ṭā “eggplant”. The selected aspirated retroflex forms identify this plant-name family, distinct from vātiṅgaṇa; source nasalization/gemination must be retained, and a deeper substrate origin is not settled.','9369.1')
add('4749','rice|rice (uncooked)|uncooked rice','caval|cavaḷ|cavar|cawal|cawar|camal|camar|cavul|cavulo|caul|caur|caura|cauḷa|cau|cal','CDIAL 4749 offers *cāmala OR *cāvala, with Prakrit cāulā/cavala, Punjabi cāval, Nepali cāmal and Bhojpuri cāur “husked rice”. Both reconstructed alternatives remain open under this existing family node; the semantic match does not include cooked-rice responses, and short cau/cal forms need local comparison.')
(P/'third-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1))
# Current graph + overlay, rather than the browser database alone, govern eligibility.
linked={r['Child_ID'] for r in csv.DictReader((ROOT/'cldf/edges.csv').open()) if r['Rank']=='1'}
linked.update(r['Form_ID'] for r in csv.DictReader((ROOT/'data/etymology-assignments.csv').open()) if r['Rank']=='1' and r['Status']=='accepted')
held={x['record']['ID'] for f in ['decisions.json','second-decisions.json'] for x in json.loads((P/f).read_text())['held']}
ws=[{norm(w) for w in q['words']} for q in qs];senses=[set(q['gloss'].split('|')) for q in qs];cs=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in linked or r['ID'] in held:continue
 w=norm(r['Form']);gs={s.strip().lower() for s in r['Gloss'].split(';')}
 if re.search(r'[,;/ ()]',w):continue
 matches=[i for i in range(len(qs)) if w in ws[i] and gs<=senses[i]]
 if matches:cs.append(dict(record=r,families=matches))
(P/'third-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1))
with (P/'third-review.txt').open('w') as f:
 for i,q in enumerate(qs):
  rs=[x['record'] for x in cs if i in x['families']];f.write(f"\n{i}. {q['parent']} {q['gloss']} ({len(rs)})\n")
  f.write('; '.join(l+': '+', '.join(sorted({r['Form'] for r in rs if r['Language_ID']==l})) for l in sorted({r['Language_ID'] for r in rs}))+'\n')
print('Candidates',len(cs))
