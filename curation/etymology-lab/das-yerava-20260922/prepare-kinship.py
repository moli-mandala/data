import csv,gzip,json,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/das-yerava-20260922';old=root/'data/curation/etymology-lab/dravidian-surveys-20260921';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
inv=[r for r in csv.DictReader(open(root/'data/cldf/forms.csv')) if r['Source'].startswith('das1987yerava[')]
assert len(inv)==90
(p/'survey-inventory.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2)+'\n')
(p/'inventory-counts.json').write_text(json.dumps({'source':'das1987yerava','records':90,'Ravula':48,'Paniya':42},indent=2)+'\n')
mapping={
('Amma','Mother'):('d183','DEDR183 explicitly gives Malayalam amma and Tamil ammā mother, with the same nursery kin term in Kannada and Tulu. Das preserves Amma in both Yerava lects. The Sanskrit/Pali comparisons in the article preclude using nursery resemblance alone as proof of a unique remote origin.'),
('Appa','Father'):('d156','DEDR156(a) gives Kannada appa, Kodagu appë and Tamil/Malayalam appan father. Das Appa preserves the same labial-geminate kin stem and paternal referent; differing mother/sister senses in other branches are not imposed on these records.'),
('Avva','Mother'):('d273','DEDR273 gives Kannada avva/avve mother and grandmother and Kodagu avvë mother. Das Ravula Avva mother fits this labial-geminate family. The article also compares Prakrit avvā; no immediate donor is established.'),
('Avva',"Father's Mother"):('d273','DEDR273 explicitly gives Parji avva father’s mother, Kannada avva grandmother and Kodagu avvë mother. Das has a separate Ravula Avva father’s mother response; its distinct referent and record are retained. The Prakrit comparison does not specify a borrowing route.'),
('Anna','Elder Brother'):('d131','DEDR131 gives Kannada aṇṇa, Kodagu aṇṇë, Malayalam/Tamil aṇṇan elder brother. Das Anna has the same kin role and doubled nasal spelling; its popular orthography does not independently establish retroflex articulation.'),
('Akka','Elder Sister'):('d23','DEDR23 explicitly gives Kannada/Tulu/Telugu akka elder sister and Kodagu akkë. Paniya Akka matches the geminate stop and kin role; the Sanskrit/Prakrit parallels remain comparisons rather than proven donor paths.'),
('Akki','Elder Sister'):('d23','DEDR23 gives Kannada akka, Kodagu akkë and Tulu akkè elder sister. Ravula Akki preserves the akk- kin stem with a high final vowel; the source gives no phonemic key, so the correspondence is a comparative proposal, not an orthographic normalization.'),
('Thamma','Younger Brother'):('d3085','DEDR3085 gives Kannada tamma and Kodagu tammaṇë younger brother beside Malayalam/Tamil tampi. Das Paniya Thamma is compared with this southern younger-brother family. Initial th is retained as popular spelling, not interpreted as evidence of aspiration.'),
('Thangi','Younger Sister'):('d3015','DEDR3015(a) gives Kannada taṅgi and Kodagu taŋge younger sister beside Tamil taṅkai. Das Paniya Thangi preserves the nasal-plus-velar kin stem and exact younger-sister role; th is not converted into an inferred phoneme.'),
('Mama',"Mother's Brother"):('d4813','DEDR4813 gives Tamil māmaṉ/māmā, Malayalam māman and Kannada māma/māva maternal uncle. Das Mama is kept with the mother’s-brother role. This widespread kin article also compares Sanskrit māma(ka); local transmission and nursery formation are unresolved.'),
('Mama',"Wife's Father"):('d4813','DEDR4813 explicitly gives Tamil māmaṉ father-in-law, Kannada māva spouse’s father and Parji māma wife’s father. Das Mama wife’s father is retained separately from the maternal-uncle response. The Indo-Aryan comparison does not establish an immediate donor.'),
('Mami',"Wife's Mother"):('d4813','DEDR4813 explicitly gives Tamil māmi wife’s or husband’s mother, Tulu māmi mother-in-law and Kodagu māviⁿ with the Tamil referents. Das Mami wife’s mother matches the feminine kin formation within the whole article, not an inferred transformation of the male response.')}
heldforms={'Achcha':'The printed source groups three grandparent roles. A plausible acca-family comparison needs contact and kin-role research before linking; keep the entire printed grouped gloss.', 'Chacha':'Possible contact kin term; chacha often has other roles regionally. The source explicitly groups three grandparent roles, so do not substitute a familiar uncle gloss or infer an immediate donor.', 'Chachi':'The source prints father’s mother; familiar regional aunt terms do not establish the correct historical route here.', 'Ithawa':'The full Ithawa formation remains unexplained; do not attach it merely because co-equivalent Avva has a supported family.', 'Ileya':'DEDR513 young/younger family is plausible, but the full Ravula kin formation and relation type need checking against same-language evidence.', 'Ilevu':'DEDR513 young/younger family is plausible, but final -vu and feminine kin formation need same-language evidence before a whole-word relation.'}
assigned={r['Form_ID'] for r in read_assignments()};props=[];held=[]
for r in inv:
 if r['ID'] in assigned:continue
 if (r['Form'],r['Gloss']) in mapping:
  d,ev=mapping[r['Form'],r['Gloss']];props.append(dict(record=r,parent=d,kind='reflex',citation=f'dedr[entry {d[1:]}];das1987yerava[p. 65]',evidence=ev+' Das p.66 describes Kannada/Malayalam/Kodagu multilingual contact; inheritance versus local transmission remains for review. DEDR node scope is the whole article per user clarification.',status='pending-review'))
 elif r['Form'] in heldforms:held.append(dict(record=r,candidate='kinship research',reason=heldforms[r['Form']]))
(p/'proposed-decisions.json').write_text(json.dumps(dict(status='pending-review',accepted=props,held=held),ensure_ascii=False,indent=2)+'\n')
(p/'externally-assigned-records.json').write_text(json.dumps([r for r in inv if r['ID'] in assigned],ensure_ascii=False,indent=2)+'\n')
used={a['parent'] for a in props};articles={r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in used and r['Status']!='duplicate-excluded'}
(p/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
s=(old/'render.py').read_text().replace('Dravidian survey research','Das (1987) Yerava research').replace('Dravidian surveys, 2026-09-21','Das (1987) Yerava, 2026-09-22').replace('2026-09-21','2026-09-22').replace('../dravidian-surveys-20260921/research-holds.json','../das-yerava-20260922/research-holds.json').replace('Review is deferred until the end of the user’s autonomous research window.','Pending user review; autonomous research continues.');(p/'render.py').write_text(s)
(p/'verify-checkpoint.py').write_text((old/'verify-checkpoint.py').read_text())
print('proposed',len(props),'held',len(held),'unexamined',90-len(props)-len(held))
