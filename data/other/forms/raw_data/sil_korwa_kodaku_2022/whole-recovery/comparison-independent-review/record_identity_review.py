"""Independent identity acceptance recommendation; no registry mutations."""
import csv, hashlib, json
from pathlib import Path
H=Path(__file__).resolve().parent
W=H.parent
D=next(p for p in H.parents if (p/'make_cldf.py').exists())
P=W/'identity-review/all-source-routing-proposal.json'
proposal=json.loads(P.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
registry={r['ID']:r for r in csv.DictReader((D/'cldf/dialects.csv').open())}
existing=[r for r in proposal['wordlists'] if 'existing_dialect_record' in r]
assert len(existing)==18
for r in existing:
    assert registry[r['dialect_id']]==r['existing_dialect_record']
    assert registry[r['dialect_id']]['Language_ID']==r['language']
pages=[21,22,23,24,27,28,60,61,63,65]
evidence=[dict(pdf_page=n,printed_page=n-10,path=str((W/f'scope/page-{n:03d}-native.txt').relative_to(D.parent)),sha256=sha(W/f'scope/page-{n:03d}-native.txt')) for n in pages]
decisions=[
dict(code='A',language='Asuri',dialect='sil-asuri-2005-husambu',status='recommend acceptance',evidence='PDF60/printed50 explicitly gives Asuri mother tongue/home/parents and Husambu current residence; 22 September2005. Matrix PDF23 gives Gumla/Jharkhand.'),
dict(code='B',language='Sadri',dialect='sil-sadri-2005-dumortoli',status='recommend acceptance',evidence='PDF60/printed50 identifies Sadri mother tongue/home/family and Dumortoli;15 October2005. PDF21 identifies local Sadri; PDF27 locates text at same named village without asserting same speaker.'),
dict(code='J',language='Birjia',dialect='sil-birjia-2005-arahas',status='recommend acceptance with new source-qualified base',evidence='PDF61/printed51 explicitly names Birjia-Arahas-Gumla, all language fields Birjia,23 September2005. PDF22 treats Birjia separately from Asuri/Mundari and Korwa/Kodaku; PDF23 groups it under Other Munda varieties.'),
dict(code='U',language='mu',dialect='sil-mundari-2005-mahuabathan',status='recommend acceptance',evidence='PDF63/printed53 explicit Mundari mother tongue/home/family, current residence Mahuabathan,5 September2005. Simdega is birthplace/upbringing; keep distinction. PDF23 supplies Palamau/Jharkhand.'),
dict(code='V',language='Sadri',dialect='sil-sadri-2005-kotgohna',status='recommend acceptance',evidence='PDF63/printed53 explicitly Sadri mother tongue/home/family,13 October2005. Dihari Korwa is community label; source says these families speak Sadri. Preserve Kotgohna/Kotgahna/Kotagohna source spellings as locality aliases.'),
dict(code='k',language='Sadri',dialect='sil-sadri-kesramal-standard',status='recommend acceptance',evidence='PDF21/23 explicitly standard Sadri, Kesramal. PDF65/printed55 has Kesrelmal-Sadri-Orissa and footnote10 says biodata/list information were lost. Do not add date or speaker facts. PDF27/28 anchors Kesramal Brave Hunter text, separately sourced through Abraham1995.'),
dict(code='T',language=None,dialect=None,status='retain typed identity hold',evidence='PDF21 mixture of Birjia/Mundari/Sadri; PDF24§3.1.4 separates Tanmai from Korwa/Kodaku/Sadri but §3.2 calls it Kodaku locally named Korwa; PDF63 calls respondent language Korwa. No unique source route can be accepted from these conflicting statements.')]
birjia=dict(status='recommend separate Birjia/bijo1238 source-qualified base',
    primary_glottolog=dict(url='https://glottolog.org/resource/languoid/id/bijo1238',independently_opened='2026-09-27',facts=['Spoken L1 language Bijori','Glottocode bijo1238','ISO639-3 bix','Birjia explicitly listed among alternate names']),
    primary_ciil=dict(url='https://sanchika.ciil.org/communities/0f3f7338-e32e-4cca-9918-9db996199b68',checked='Official indexed repository community text independently retrieved2026-09-27',claim='Munda affiliation described as commonly assumed; preserve qualification.',sketch_url='https://sanchika.ciil.org/server/api/core/bitstreams/f56fac29-02f8-4c31-83d1-4d4f2aed663d/content',sketch_limit='Indexed original first-page text corroborates limited Gumla fielddata; direct fetch failed. No full sketch collation or publication date claim.'),
    recommendation='Munda broad display clade is supported by Behera matrix and qualified CIIL statement. Do not infer finer subgroup or merge Birjia with Birhor/Asuri/Korwa. Modern catalog alias supports this named language route, not all historical community-name occurrences.',
    coordinates='Keep blank; Arahas/Gumla belongs in locality metadata, not an inferred point.',
    required_metadata_adjustment='Change proposed base Quality C to blank when Latitude/Longitude are blank, consistently with the six new dialect proposals; no coordinate-quality assessment has been established.')
texts=[dict(section=r['section'],language=r['language'],current_id=r['dialect_id'],recommended_id='sil-kodaku-kanda' if r['section']=='C.3' else r['dialect_id'],status='accept locality route with identifier correction' if r['section']=='C.3' else 'accept locality route',evidence='PDF27 Table7 independently viewed: Job=PahadiKorwa/Rakkaya;Marriage=Kodaku/Kanda;Life=LocalSadri/Dumortoli;BraveHunter=StandardSadri/Kesramal. PDF28 prose independently viewed clarifies test location versus source locale and Abraham1995 provenance.',constraint='Text role/title is provenance, not a dialect. Do not infer identical respondent/date from shared site; no Kanda coordinates or replacement with another Kodaku site.') for r in proposal['texts']]
report=dict(status='recommend source-stage identity acceptance after two narrow metadata fixes',reviewer='rohru_finish',input_sha256=sha(P),original_pdf_sha256=proposal['source_sha256'],
    method='Read original native-PDF page evidence for all requested sheets and source matrix/context. Independently viewed existing PDF27/28 original images. Reopened official Glottolog and CIIL indexed primary repository evidence. No new images or downloads; no canonical writes.',
    original_page_evidence=evidence,wordlist_decisions=decisions,birjia=birjia,text_decisions=texts,
    preserved_existing_routes=[dict(code=r['source_code'],language=r['language'],dialect_id=r['dialect_id'],exact_registry_match=True) for r in existing],
    registry_sha256=sha(D/'cldf/dialects.csv'),
    required_fixes=[dict(field='texts[C.3].dialect_id',old='sil-kodaku-kanda-story',new='sil-kodaku-kanda',reason='Kanda is the source locality; story is a record role. Update all proposed route/tag references consistently before registration.'),dict(field='proposed_language_registry_row.Quality',old='C',new='',reason='Coordinates absent and no coordinate-quality estimate established; align with all new dialect records.')],
    scope_limits=['Existing18 routes are preserved byte-field exactly; their historical coordinates were not independently re-geocoded in this review.','Six comparison dialects retain blank coordinates/quality and parent clades checked in independent-routed-review-pass2.json.','Tanmai246lexical candidates remain fully preserved as unresolved identity evidence, not silently lost.','This is identity recommendation, not complete whole-source installation approval.'])
(H/'identity-acceptance-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],preserved=len(existing),resolved_new_comparison_routes=6,texts=4,required_fixes=2)))
