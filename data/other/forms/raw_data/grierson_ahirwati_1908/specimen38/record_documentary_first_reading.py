"""Preserve complete No38 documentary prose and free translation as first readings.

These are documentary witnesses, not additional target-language lexical rows.
Latin quantities in source quotations still need narrow occurrence review.
"""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=Path(__file__).resolve().parents[7]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest_path=BASE/'scope/native-specimen-image-manifest.json'
manifest=json.loads(manifest_path.read_text())
by_page={r['printed_page_expected']:r for r in manifest['pages']}
texts=[
('lsi9.2:p237:folktale-provenance',237,'documentary-provenance',
 'The other specimen of Ahīrwāṭī comes from the Jhajjar Tahsīl in the south of Rohtak. It is a folktale illustrating the reputed avarice of people of the Ahīr (or as they are often locally called Hīr) caste. An Ahīr promises to give his son-in-law whatever he asks for. When the son-in-law asks for a very petty present, the Ahīr invents all kinds of excuses to get off giving it.'),
('lsi9.2:p237:mixed-Ahirwati-Bangaru-quoted-comparisons',237,'documentary-linguistic-analysis-with-quoted-witnesses',
 'The specimen is written in the Persian character, as received from Rohtak. It well shows the mixed character of the Ahīrwāṭī of this district. The first sentence, ēk Ahīr dukhāḷō paṛō thō, is good Ahīrwāṭī, and the second, us-kā jamāī bērē-mai āyā, is equally good Bāṅgarū. So throughout the whole specimen Bāṅgarū and Ahīrwāṭī forms are found side by side, often in the same sentence. Sometimes we have words like bōlō, for the Ahīrwāṭī bōlyō, and sometimes words like the Bāṅgarū bōlā. The only local peculiarity is the dropping of the y in the past participle (bōlō for bōlyō) which is general over the whole of the Rohtak district. In one place the oblique form aĩh is used for the nominative yō, this.'),
('lsi9.2:p238:free-translation-opening',238,'whole-text-free-English-translation',
 'There was a certain Ahīr, who had fallen sick. His son-in-law came to visit him. It happened that that day he was a little better. He asked his brother who the young'),
('lsi9.2:p239:free-translation-completion',239,'whole-text-free-English-translation',
 'fellow with the red turban was. ‘A guest,’ said he. ‘Who is he?’ ‘Its Jai-kaḷi’s husband.’ Said the Ahīr, ‘are you Jai-kaḷi’s husband?’ ‘Yes, sir.’ ‘Then, brother, to-day I feel better. Ask me for a present.’ ‘Sir,’ said the son-in-law, ‘your soul is tightened now. If I ask for anything, you will not give it.’ ‘Why should I not give it? What you have heard are the truthful words of a dying man.’ ‘Very well, if you give anything, give me that four-pronged corn-rake that used to hang there.’ ‘That,’ said the Ahīr, ‘is just what I can’t give you, my fine fellow. It has three rings on each joint. I have had it by me for one-and-twenty years. It is the one which my dear old uncle Hukamlō used to work with. It’s a regular bit of my liver. I have broken off three marriages for each (joint) rather than part with it. How on earth can I give it you?’'),
]
parents={json.loads(l)['key'] for l in (BASE/'scope/physical-source-unit-census.jsonl').read_text().splitlines()}
records=[]
for key,page,kind,text in texts:
    assert key in parents
    witness=by_page[page]
    assert sha(ROOT/witness['lossless_png_path'])==witness['lossless_png_sha256']
    records.append({'source_unit_key':key,'printed_page':page,'kind':kind,'raw_documentary_first_reading':text,
                    'original_witness':witness,'review_status':'root whole-block first reading; independent literal/quantity review pending',
                    'typed_uncertainties':[{'type':'transcription','reason':'Quoted Latin quantities, personal-name spelling and small punctuation require narrow original comparison before certified transcription.'}],
                    'emission_policy':'Documentary only. Free English translation is not token-aligned lexical Gloss. Historical social characterization remains an attributed source statement, not an editorial claim.'})
target=HERE/'documentary-first-reading.jsonl'
assert not target.exists()
target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
status={'status':'four complete documentary first-reading blocks; not accepted lexical emission','records':4,
        'source_manifest_sha256':sha(manifest_path),'output_sha256':sha(target),
        'free_translation_continuity':['lsi9.2:p238:free-translation-opening','lsi9.2:p239:free-translation-completion'],
        'pending':['Independent original-wording check','Four quoted-comparison groups require fine child accounting/routing and narrative-span reconciliation','Native237 Persian whole-text reading and conservative alignment'],
        'canonical_changes':False}
(HERE/'documentary-first-reading-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'records':4,'sha256':sha(target)}))
