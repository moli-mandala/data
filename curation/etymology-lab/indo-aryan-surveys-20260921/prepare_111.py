import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[];h=[]
for r in rs:
 if r['ID'] in done or r['Language_ID']!='Kho':continue
 f,g=r['Form'],r['Gloss'];x=None;reason=None
 if f=='kār' and g=='ear':x=dict(parent='3056',kind='reflex',citation='CDIAL[3056 and addenda]',evidence='Turner *kāra⁶ explicitly includes Khowar kār ear, and the addendum rules out borrowing from Waigali. The entry leaves its connection with karṇa unclear; this analysis retains the specific *kāra ear head, without collapsing it into karṇa or the unrelated tax homonym.')
 if f=='dūdɛrī' and g=='far':x=dict(parent='6495',kind='reflex',citation='CDIAL[6495 and addenda];bashir2023[s.v. dudéri]',evidence='Turner explicitly gives Khowar dudéri distant/distance within dūra, comparing the extended dūratara form and dissimilation of r–δ–r. The addendum repeats duderi under that derivation; Bashir independently cites T6495. The link is to the attested dūra family, with the precise comparative development qualified as in Turner.')
 if f in ['drūŋ','drūŋg'] and g=='long':x=dict(parent='6616',kind='reflex',citation='CDIAL[6616];bashir2023[s.v. drung]',evidence='Turner *draṅgha long explicitly gives Khowar druṅg/drūṅg long, tall, and Bashir cites this same entry. This uses the specific *draṅgha head rather than substituting dīrgha merely because of the shared meaning; Turner’s deeper Indo-European comparison remains tentative.')
 if f in ['īspīsār','īsʦār','ispsār','īspesār','ispisār'] and g=='sister':x=dict(parent='13913',kind='reflex',citation='CDIAL[13913];bashir2023[s.v. ispusár]',evidence='Turner derives Khowar ispusār from the accusative svasāram, with retention influenced by brār brother. Bashir explicitly records reduced variants ispsár, istsár and isptsár as well as isprár; the survey forms fit that documented sister paradigm.')
 if f in ['kosī-','kosi-'] and g=='walk':x=dict(parent='2981',kind='reflex',citation='CDIAL[2981]',evidence='Turner kasati explicitly cites Khowar kosik to walk about, with first-person kasim. The elicited kosī-/kosi- verbal forms belong to that documented stem; ordinary verbal inflection does not require a separate derivational edge.')
 if f in ['pośī-','pośi-'] and g=='see':x=dict(parent='8012',kind='reflex',citation='CDIAL[8012]',evidence='Turner paśyati explicitly includes Khowar pošik see. The survey pośī-/pośi- forms preserve this stem with the source’s sibilant spelling and ordinary verbal endings.')
 if f=='ūx' and g=='water':x=dict(parent='1921',kind='reflex',citation='CDIAL[1921,1]',evidence='Turner udaka explicitly includes Khowar uγ water, fruit juice and sap in its first branch, beside Kalasha ūk. The survey ūx has the same velar-fricative ending with recorded voicing variation; it is not assigned to the separate daka branch.')
 if f=='gāmɛṣ' and g=='buffalo':x=dict(components=['4147-2','9964'],citation='CDIAL[4147,2];CDIAL[9964];bashir2023[s.v. gaméṣ]',evidence='Both Turner *gāvā cow and mahiṣa buffalo explicitly cite Khowar ga-mεṣ; Bashir likewise cites T4147 and T9964 for gaméṣ. These support the ordered cow + buffalo compound analysis, rather than a resemblance-based assumption of direct Persian borrowing.')
 if f=='istārī' and g=='star':reason='Turner 13713 lists Khowar istari under str̥-, but Bashir explicitly preserves Iranian (Morgenstierne 1936) versus Sanskrit (1973) alternatives. Hold pending examination of the competing arguments.'
 if f=='yor' and g=='sun':reason='Turner 612 aru marks this comparison doubtful and questions initial y-. Bashir instead identifies Iranian origin and compares Wakhi yír. No supported immediate donor route is resolved.'
 if f=='ṣoī' and g=='near':reason='Turner 12684 śraya labels the proposed borrowing through a dialect with ṣ < śr very doubtful. Bashir repeats that number; a firm inherited edge would erase the primary qualification.'
 if f in ['īskūrdī','uskūrdī','iskūr'] and g=='short':reason='Bashir iskúrdi short is explicitly Iranian, unlike adjacent iskuṭík shrink < CDIAL 13650. Do not transfer the neighbouring entry’s etymology to the adjective; immediate donor unresolved.'
 if f=='śūny' and g=='needle':reason='Turner 13551-2 compares Kho šunǰ, whereas Bashir explicitly labels šunǰ a Yidgha loan. The survey’s final y also requires original-transcription checking. Hold rather than assert inheritance.'
 if f=='āyūkūn' and g=='egg':reason='Bashir ayúkun egg is explicitly Iranian (Morgenstierne 1936). No exact supported immediate donor node was resolved.'
 if f=='duxūr' and g=='fingernail':reason='Bashir doγúr/duγuúr is explicitly borrowed from Wakhi. A supported Wakhi donor node must be resolved before saving; no Sanskrit similarity substitute.'
 if x:a.append(dict(record=r,**x))
 if reason:h.append(dict(record=r,reason=reason))
(P/'pass111-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n');print('accepted',len(a),'held',len(h))
p=P/'lexical-reference-links.json';links=json.loads(p.read_text());
for x in a:
 for c in x['citation'].split(';'):
  if c.startswith('bashir2023'):links[c]='https://escholarship.org/uc/item/955239w9'
p.write_text(json.dumps(links,ensure_ascii=False,indent=1)+'\n')
