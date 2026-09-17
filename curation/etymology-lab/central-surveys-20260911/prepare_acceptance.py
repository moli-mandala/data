"""Prepare the explicitly approved donor correction; does not save the overlay."""
import csv, json, copy, sys, unicodedata
from pathlib import Path
P = Path(__file__).resolve().parent
ROOT = P.parents[2]
sys.path.insert(0, str(ROOT))
from assign_form_ids import assign_ids
RAW = ROOT / 'data/other/params/raw_data'
def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
specs = [
 ('badan','Ar','badan','body','بدن','badan','A بدن badan, s.m. Body; privities.', 'Arabic body head; exclude the Sanskrit-derived mouth/face homonym.'),
 ('aurat','Pers','ʻaurat','private parts','عورت','aurat','P عورت ʻaurat (for A. عورة); originally private parts; in Urdu a woman or wife.', 'Persian-labelled etymological head. Woman/wife is explicitly an Urdu semantic development in Platts, not projected into Persian.'),
 ('zaban','Pers','zabān','tongue; language','زبان','zabaan','P زبان zabān, or zubān: the tongue; speech, language, dialect.', 'Exclude Arabic jabān coward. Preserve Persian z and long ā.'),
 ('kam','Pers','kam','less; deficient; little','کم','kam','P کم kam: deficient, less, short; little, small.', 'Exclude kām work/desire and Sanskrit √kam.'),
 ('sabut','Ar','s̤ubūt','permanence; firmness; proof','ثبوت','ثبوت','A ثبوت s̤ubūt, vulg. s̤abūt: permanence, firmness, proof; adjectivally firm, proved, entire.', 'Arabic etymological head uses primary u-vocalism. Regional a-vocalism and the entire adjective sense are retained in survey evidence; no direct Arabic contact is asserted.'),
 ('makan','Ar','makān','place; dwelling; house','مکان','مکان','A مکان makān: a place; a habitation, dwelling, abode, house, home, room.', 'Arabic noun of place; excludes unrelated makka and makkhan.'),
 ('wazni','Ar','waznī','weighty; heavy','وزني','vaznii','A وزني waznī, vulg. wazanī (relative noun from wazn): weighty, heavy; grave, important.', 'Preserve the complete adjective and primary vocalism; do not replace it with the weight noun.'),
 ('wazndar','Pers','wazn-dār','weighty; heavy','وزن دار','vazn','Platts s.v. wazn lists wazn-dār weighty. Dehkhoda: وزن دار. [ وَ ] (نف مرکب ) وزن دارنده . دارای وزن . سنگین .', 'Persian compound adjective, independently attested in Dehkhoda; Arabic wazn plus Persian -dār. No claim about which language transmitted it to these surveys.'),
 ('phulgobi','H','pʰūlgobī','cauliflower','फूलगोबी','', 'Cent Saral Bhasha, PDF page 11, vegetable item 4: फू लगोबी — Cauliflower.', 'Preserve printed unaspirated b; join layout-induced space in फूल. Hindi lexical grouping head; existing Kannauji attestation remains distinct. No reconstruction of the whole compound.'),
]
bank='https://www.centralbankofindia.co.in/sites/default/files/%E0%A4%95%E0%A4%A8%E0%A5%8D%E0%A4%A8%E0%A4%A1.pdf#page=11'
deh='https://vajehyab.com/?f=amid&q=%D9%BE%D9%88%D9%84+%D8%AF%D8%A7%D8%B1&s=text'
audit=[]
for key,lang,form,gloss,native,query,excerpt,note in specs:
    source=f'platts1884[s.v. {query}]'
    urls=['https://www.rekhta.org/urdudictionary?keyword='+query]
    if key=='wazndar':source='platts1884[s.v. wazn, wazn-dar];dehkhoda-vajehyab2026[s.v. وزن دار]';urls.append(deh)
    if key=='phulgobi':source='centralbank-saral-kannada[PDF p. 11, vegetables, item 4]';urls=[bank]
    if key=='wazni':source='platts1884[s.v. wazni]'
    audit.append(dict(ID='central-surveys-donor-'+key,Entry_Key='central-surveys-donor-'+key,Language_ID=lang,Form=unicodedata.normalize('NFC',form),Original=form,Native=native,Gloss=gloss,Source=source,SourceExcerpt=excerpt,URLs=urls,Transcription=note,Status='install',Evidence='Selected dictionary etymological head, with transmission deliberately unspecified by user approval. '+note))
forms=[dict(ID=r['ID'],Language_ID=r['Language_ID'],Form=r['Form'],Original=r['Form'],Gloss=r['Gloss'],Source=r['Source'],Status='entry') for r in audit]
registry=list(csv.DictReader((ROOT/'data/form-identities.csv').open()))
mapping,updated=assign_ids(copy.deepcopy(forms),copy.deepcopy(registry))
oldids={r['Form_ID'] for r in registry}
new=[r for r in updated if r['Form_ID'] not in oldids]
assert len(new)==9
for r in audit:r['Persistent_ID']=mapping[r['ID']]
dump(RAW/'20260911-central-surveys-donors-audit.json',audit)
dump(P/'new-donor-identities.json',new)
dump(P/'new-donor-forms.json',[dict(r,ID=mapping[r['ID']]) for r in forms])
print('Prepared nine independently identified donor heads; overlay untouched.')
