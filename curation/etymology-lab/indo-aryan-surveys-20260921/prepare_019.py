import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
groups=[
 dict(parent='13264',gloss=['small'],languages=['Hajong','Bhatri'],forms=['huru','suṛu'],citation='CDIAL[13264]',evidence='Turner gives Assamese xaru “small, fine, younger, inferior” and Bengali/Oriya saru “fine, thin, slender” under saru. These eastern huru/suṛu forms match the whole regional small adjective, with h for the eastern s/x series and source flap notation retained. The link leaves regional contact open; Bishnupriya hurkaŋ has additional material and is not reduced to this stem.'),
 dict(parent='10930-8',gloss=['bad'],languages=['Bshk','Tor'],forms=['laɬ','lalɬ','lāṣ'],citation='CDIAL[10930,8]',evidence='Section 8 *latra, compared with litra, explicitly gives Bashkarik lāλ and Torwali lāš “bad”. These survey lateral-fricative/sibilant forms match the named local comparanda, preserving the source spelling lalɬ as a variant rather than normalizing it. This is a qualified member of Turner’s expressive defective family; it does not assert a regular derivation from the unrelated cloth sense in section 1.'),
 dict(parent='13355',gloss=['good'],languages=['Vasavi'],forms=['hāro','hārõ'],citation='CDIAL[13355]',evidence='The full sāra article explicitly gives Gujarati sārũ “good, whole” and explains the semantic relation to the best/essential part. Vasavi hāro/hārõ has the matching whole adjective with western s-to-h development and ordinary final agreement. Forms with dz/j in place of r are not automatically included.'),
 dict(parent='f_oz4z6vzzul6kw',kind='borrowed',gloss=['few'],languages=['H'],forms=['kam'],citation='platts1884[s.v. kam]',evidence='Platts labels kam Persian and records “less, little, small, scanty”. The existing Persian kam donor node supports these Hindi few responses as loans, rather than inherited continuations of a similar Sanskrit form. Non-Hindi regional attestations are not assigned directly to Persian by this rule.'),
 dict(parent='f_dbu2w7q6m7sh2',kind='borrowed',gloss=['bad'],languages=['poth'],forms=['xarāv'],citation='platts1884[s.v. ḵẖarāb, p. 488];liljegren[entry LX002408]',evidence='The complete cached Platts article gives ḵẖarāb “spoiled, bad” and its Arabic origin; Liljegren LX002408 explicitly identifies Urdu xaraab as the donor of Palula xaraap. The existing Hindi/Urdu xaraab node therefore provides an attested regional donor for Pothwari xarāv with b/v adaptation. The borrowed link is a provisional regional transmission hypothesis, not a claim of direct Arabic contact.'),
 dict(parent='f_ravane4ki3ewe',kind='borrowed',gloss=['meat'],forms=['sīkār','śīkār','sikār','śīkar','sīkar','sikhār','śikhār','sikʰar','çikar'],citation='platts1884[s.v. shikār];zoller2023[p. 772, footnote 327]',evidence='Platts labels shikār Persian and includes hunting, prey and game; Zoller’s full footnote explicitly compares West Pahari śikār “game; meat”, supporting the regional semantic extension. The existing Hindi sikār donor proxy has a direct-source Hindi comparison (Bhattacharya) and is used here as a provisional Hindi/Urdu route. Both the donor proxy and local transmission remain provisional; the analysis does not skip directly to a remote Indo-Iranian root. Syllable-reduced and śakkar-like ambiguous responses are excluded.'),
 dict(parent='9932',gloss=['mother'],forms=['mahatārī','maht ārī','mahat ārī','mahat āri','mɛhɛt ārī'],citation='CDIAL[9932];platts1884[s.v. mahtārī, p. 1099]',evidence='Platts explicitly derives mahtārī “mother” from mahat + tara + ikā. This supplies the feminine formation for Turner’s mahattara elder/great family, whose article also has Marathi mahātārā old and Konkani mhāntāro/mhāntāri old man/woman. The h-bearing survey forms fit this full formation, with internal extraction spaces retained. Short matāī/matārī forms which could mix with mātr are held instead of generalized to this rule.'),
]
a=[]
for i,g in enumerate(groups):
 for r in rs:
  if r['ID'] not in done and r['Gloss'] in g['gloss'] and r['Form'] in g['forms'] and ('languages' not in g or r['Language_ID'] in g['languages']):
   a.append(dict(record=r,parent=g['parent'],kind=g.get('kind','reflex'),citation=g['citation'],evidence=g['evidence'],family=i))
assert len(a)==len({x['record']['ID'] for x in a})
(P/'pass019-rules.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1)+'\n')
(P/'pass019-decisions.json').write_text(json.dumps({'accepted':a,'held':[]},ensure_ascii=False,indent=1)+'\n')
f=P/'lexical-reference-links.json';links=json.loads(f.read_text());links.update({'s.v. kam':'https://www.rekhta.org/urdudictionary?keyword=kam','s.v. shikār':'https://www.rekhta.org/urdudictionary?keyword=shikaar'});f.write_text(json.dumps(links,ensure_ascii=False,indent=1)+'\n')
old=P.parent/'indo-aryan-sil-20260914'
e=[]
for n,word in [('platts-additional-research.json','ḵẖarāb'),('platts-week-mother-followup.json','महतारी')]:
 e.extend(x for x in json.loads((old/n).read_text()) if x.get('word')==word)
(P/'pass019-reused-primary.json').write_text(json.dumps(e,ensure_ascii=False,indent=1)+'\n')
print(len(a),[(g['parent'],sum(x['family']==i for x in a)) for i,g in enumerate(groups)])
