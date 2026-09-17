"""Apply the user's donor-nesting correction to pending research manifests."""
import csv,json,datetime
from pathlib import Path
P=Path(__file__).resolve().parent; ROOT=P.parents[2]
forms={r['ID']:r for r in csv.DictReader((ROOT/'cldf/forms.csv').open())}
new=json.loads((ROOT/'data/other/params/raw_data/20260911-central-surveys-donors-audit.json').read_text())
for r in new:forms[r['Persistent_ID']]=dict(r,ID=r['Persistent_ID'])
ids={r['ID'].removeprefix('central-surveys-donor-'):r['Persistent_ID'] for r in new}
mapping={
 'f_alir5ifyzfzvq':'f_lmwk7knn4svus','f_ra6uvjvpsrlbe':'f_rlqlstrekfzvc','f_44kuz3f4pxo5e':'f_74vcu23lscgak','f_thgcjhj7vr5i2':'f_7a3ccguwm7lam',
 'f_xitegx7mbquyi':'f_odkzxsetllmtk','f_7tuydfvqt7ici':ids['badan'],'f_gsynajbspnc2y':'f_honuos2suexey','f_vhi3eao4uvyjm':ids['aurat'],
 'f_zqbwnodbqwsho':ids['zaban'],'f_prmeiidvi4gto':'f_2h46v2qxjhcaq','f_snjuyk7aztjyu':'f_jl7b6dgwhptmm','f_nomhmgwrljcl4':'f_tusbqtxogmrni',
 'f_n4lkgn2ws3cvw':'f_vbtodh3gxxoq2','f_cdko5w5xxg5qg':'f_khzburdg3bkvo','f_lwa4hsrbk5gee':'f_tfolnhjem4qz6','f_dbu2w7q6m7sh2':'f_4txbqir4uazlw',
 'f_nc5i5pyxldlxg':'f_fqt722mnhxvpu','f_7u6j2wcwpjlti':ids['wazni'],'f_47fi53dmujrya':ids['wazndar'],'f_cdzupphnjmodm':'f_edny5jsgvz5fk',
 'f_bk76lqk32ood6':ids['makan'],'f_qbahea6yfb6qm':'f_fnwdoz5wjhgai','f_5azv73u3n6upa':'f_3smlfjflb2ukg','f_xqxbqhxafwpsu':'f_f5yhi6kl6jazu',
 'f_ymjgts524mota':ids['kam'],'f_bgpkwqir4pwla':'f_xrwhducfi7pgc','f_22utxbhjyd5lu':'f_ppgmg52jy2qtw','f_dbuiao2qkdgv4':ids['sabut'],
 'f_wl2o55yco7mxk':ids['phulgobi'],
}
special={
 ids['aurat']: 'Platts labels ʻaurat Persian, ultimately Arabic, and explicitly marks woman/wife as its Urdu meaning; that semantic development is retained without projecting it into Persian.',
 ids['sabut']: 'Platts gives Arabic s̤ubūt, regional s̤abūt, and the adjective entire; the donor head retains the primary noun meaning and vocalism.',
 ids['wazni']: 'Platts explicitly labels the complete adjective waznī Arabic and gives colloquial wazanī; survey b/v and j adapt w and z.',
 ids['wazndar']: 'Platts lists the full adjective wazn-dār and Dehkhoda independently records Persian وزن دار heavy; the Arabic base and Persian -dār remain a whole adjective, with local b/v, j and final-r variation.',
 ids['badan']: 'Platts explicitly distinguishes Arabic badan body from the Sanskrit-derived mouth/face homonym.',
 ids['zaban']: 'Platts gives Persian zabān for physical tongue as well as language; survey j adapts z. Arabic jabān coward is excluded.',
 ids['makan']: 'Platts identifies Arabic makān place, dwelling or house; the survey forms retain this lexical sense.',
 ids['kam']: 'Platts identifies Persian kam less/deficient, distinct from kām and the Sanskrit verbal root.',
 'f_3smlfjflb2ukg': 'The Persian murɣ chicken family underlies the Hindustani feminine murgī cited by Platts; the regional feminine ending is retained without claiming it was borrowed as such directly from Persian.',
 'f_xrwhducfi7pgc': 'Platts confirms the nāxun fingernail family; local kh/k/h and vowel adaptation remain qualified.',
 'f_odkzxsetllmtk': 'Platts and the installed Persian head support barābar equal; local reduction and reduplicated-looking syllables do not establish independent local derivation.',
}
paths=[]
for lid in ['mewari_basad','Nimadi','bagheli_lakshman']:
 for path in sorted((P.parent/lid).glob('batch-*.json')):
  m=json.loads(path.read_text())
  if m.get('researchDirectory')==str(P):paths.append((path,m))
archive=P/'before-approval-manifests.json'
assert not archive.exists(),'Correction already prepared; do not replay.'
archive.write_text(json.dumps({str(path.relative_to(ROOT)):m for path,m in paths},ensure_ascii=False))
audit=[]; rows=[]
for path,m in paths:
 for q in m['proposals']:
  old=q['parentId']
  if old in mapping:
   nid=mapping[old];n=forms[nid];lang={'Pers':'Persian','Ar':'Arabic','H':'Hindi-Urdu'}[n['Language_ID']]
   q['evidenceBeforeApproval']=q['evidence'];q['parentBeforeApproval']={'id':old,'form':q['parentForm'],'language':q.get('parentLanguage')}
   if n['Language_ID']=='H':
    evidence='Central Bank of India, Cent Saral Bhasha, PDF p. 11, vegetable item 4, pairs फूलगोबी with cauliflower. This independent Hindi lexical head replaces the unlinked survey proxy; aspiration/frication and vowel length in regional responses remain qualified. Reversed gobiphul and bare gobi are excluded.'
   else:
    basis=special.get(nid,f'The installed {lang} {n["Form"]} ‘{n["Gloss"]} head and the cited primary lexical evidence identify this loan family. The selected survey forms retain local consonant, vowel and ending variants; these are not silently normalized.')
    evidence=basis+' Nested under the Persian/Arabic etymological entry at the user’s request; this records the loan family, not direct contact or a demonstrated Hindi-Urdu intermediary.'
   q.update(parentId=nid,parentForm=n['Form'],parentLanguage=lang,evidence=evidence)
   q['citation']=';'.join(dict.fromkeys(q['citation'].replace('Platts[','platts1884[').split(';')+n['Source'].split(';')))
   q.pop('acceptanceDependency',None)
   for a in q['assignments']:a.update(Etymon_ID=nid,Source=q['citation'],Notes=evidence)
   audit.append(dict(survey=m['survey'],number=q['number'],oldParent=old,newParent=nid,language=n['Language_ID'],form=n['Form'],rows=len(q['assignments']),source=n['Source']))
  for a in q['assignments']:
   a['Notes']=a['Notes'].replace('Pending central-survey proposal','Approved central-survey proposal')
  q['status']='approved-pending-validation';rows+=q['assignments']
 m['status']='approved-pending-validation'
 m['approval']='User: i approve all; nest Perso-Arabic loans under Persian/Arabic entries, without asserting Hindi-Urdu mediation.'
 path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
(P/'approved-assignments.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
(P/'donor-nesting-corrections.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(proposals=sum(len(m['proposals']) for _,m in paths),rows=len(rows),correctedProposals=len(audit),correctedRows=sum(x['rows'] for x in audit),persoArabicRows=sum(x['rows'] for x in audit if x['language']!='H'))))
