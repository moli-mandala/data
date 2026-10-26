"""Source-only owner first reading of every p49–51 census parent.

No canonical emission, automatic variant links, or phonemic interpretation.
Tuple fields: source form, printed gloss, explicit grammatical analysis,
language scope, disposition, source qualification. Source symbols are literal.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
HERE = Path(__file__).resolve().parent
SCOPE = HERE.parent / 'scope'
GROUPS = {}
CONTEXT = {}

def group(page, name, quote, rows):
    key = f'lsi9.2:p{page}:{name}'
    CONTEXT[key] = quote
    GROUPS[key] = rows

def w(form, gloss='', analysis='', language='Ahirwati', disposition='eligible', note=''):
    return (form, gloss, analysis, language, disposition, note)

D = 'documentary'
H = 'unattested_source_hypothesis'

group(49, 'distribution-1', 'Ahīrwāṭī, also called Hīrwāṭī and Ahīrwāl; Delhi and Rohtak much mixed with Bāṅgarū.', [
    w('Ahīrwāṭī', '', 'language name', '', D), w('Hīrwāṭī', '', 'language name', '', D),
    w('Ahīrwāl', '', 'language name', '', D), w('Mēwātī', '', 'Delhi source return label', '', D),
    w('Bāṅgarū', '', 'neighbouring dialect name', '', D)])
group(49, 'mewati-boundary-2', 'Ahīrwāṭī described as merely a variety of Mēwātī; centre Rewari in West Gurgaon.', [])
group(49, 'nabha-narnaul-mixture-3', 'South Nabha/Narnaul distribution; surrounding Bāgrī, Mārwāṛī, Jaipuri, Mēwātī; Narnaul mixture qualified.', [])
group(49, 'classification-4', 'Connecting link between Mēwātī, Bāṅgarū, Bāgrī, Shēkhāwātī; backbone throughout Mēwātī.', [])
group(49, 'Abhira-history-5', 'Ābhīras / Āhīrs / Hīrs historical tribal names; Samudra-gupta and Kāṭṭis historical commentary.', [
    w('Ābhīras', '', 'historical ethnonym', '', D), w('Āhīrs', '', 'historical ethnonym', '', D), w('Hīrs', '', 'historical ethnonym', '', D)])
group(49, 'Ahir-history-6', 'Historical Ahīr rulers and Asa/Asirgarh; Ptolemy ethnonym quotation; continues p50.', [
    w('Ἀβίρια', '', 'quoted Ptolemy name', '', D, 'Greek accent/letter typography requires independent review; not target-language attestation.')])
group(49, 'footnote-name-hyphens', 'Properly speaking there should be a hyphen in each of these three names. Thus, Ahīr-wāṭī, Hīr-wāṭī, and Ahīr-wāl.', [
    w('Ahīr-wāṭī', '', 'source preferred segmentation of name', '', D),
    w('Hīr-wāṭī', '', 'source preferred segmentation of name', '', D),
    w('Ahīr-wāl', '', 'source preferred segmentation of name', '', D)])
group(49, 'footnote-Dajana-unreturned', 'Dujana language return not received; left out of account.', [])
group(49, 'footnote-Narnaul-list-classification', 'Original Rough List called Narnaul dialect Bāgrī-Mēwātī; author treats it as a form of Ahīrwāṭī.', [
    w('Bāgrī-Mēwātī', '', 'historical survey label', '', D)])
group(50, 'history-continuation-and-Khandeshi-comparison', 'Distinct Mālvī/Ahīrī and Khāndēśī/Ahīrāṇī names; hypothetical Bhilla/Bhīl from Ābhīra. Most important agreement is sū̃ “I am”, typical of Ahīrwāṭī and connected dialects, also common in Khāndēśī.', [
    w('Ahīrī', '', 'Malvi alias', '', D), w('Ahīrāṇī', '', 'Khandeshi alias', '', D),
    w('Bhilla', '', 'hypothetical ethnonym derivation', '', D), w('Bhīl', '', 'hypothetical ethnonym derivation', '', D),
    w('Ābhīra', '', 'hypothesized historical etymon', '', D),
    w('sū̃', 'I am', 'first person singular copula', 'Ahirwati'),
    w('sū̃', 'I am', 'first person singular copula', 'Khandeshi', 'eligible_comparator', 'Same printed token explicitly shared across the two language attributions; no invented second printed token.')])
group(50, 'population-table-introduction', 'Reported speakers table; total 448,945.', [])
for n, label, value in [(1,'Gurgaon','159,900'),(2,'Pataudi','19,000'),(3,'Delhi (returned as Mēwātī)','18,694'),(4,'Rohtak (Jhajjar)','71,470'),(5,'South Nabha','43,881'),(6,'Narnaul of Patiala (returned as Bāgrī-Mēwātī)','136,000')]:
    group(50, f'population-cell:{n}', f'{label}: {value}. Documentary population, not lexical evidence.', [])
group(50, 'literature-and-authorities', 'Author knows no works written in Ahīrwāṭī and no previous account of their language.', [])
group(50, 'script-provenance-unprinted-Gurmukhi', 'Deva-nāgarī/Gurmukhī/Persian script provenance; Sikh Nabha Gurmukhi specimens need not be printed; Gurgaon Devanagari and Rohtak Persian specimens supplied.', [])
group(50, 'Gurmukhi-not-printed', 'Those in the Gurmukhī character need not be printed.', [])
group(50, 'grammar-standard-and-copula-comparison', 'Gurgaon is the grammatical standard. Uses sū̃ instead of Mēwātī hū̃ “I am”; connections with Bāṅgarū qualified by regional mixture.', [
    w('sū̃', 'I am', 'first person singular copula'),
    w('hū̃', 'I am', 'first person singular copula', 'Mewati', 'eligible_comparator')])
group(50, 'nominal-ending-comparison-continuation', 'Strong masculine a-base: nominative singular ō, oblique ē; Bagri nominative ā, oblique ē. Same rule in adjectives and genitive suffixes, continuing p51.', [
    w('a', '', 'base-class notation', '', D), w('ō', '', 'strong masculine noun nominative singular ending'),
    w('ē', '', 'strong masculine noun oblique singular ending'),
    w('ā', '', 'strong masculine noun nominative singular ending', 'Bagri', 'eligible_comparator'),
    w('ē', '', 'strong masculine noun oblique singular ending', 'Bagri', 'eligible_comparator')])
group(50, 'population-footnote', 'Nabha figure reported 63,881; 20,000 shown under Mēwātī (p45).', [])
group(51, 'locative-genitive-whole-example', 'Locative agreement ends in ē, not ō. Thus mhārē (not mhārō) ghari, in my house.', [
    w('ē', '', 'locative agreement ending'), w('ō', '', 'contrastive non-locative agreement ending'),
    w('mhārē ghari', 'in my house', 'whole example', note='Printed parenthesis contrasts mhārō inside the quoted span; full physical quotation remains in source_quote.'),
    w('mhārē', '', 'possessive locative agreement', note='Same physical token as whole example; no separately supplied lexical gloss.'),
    w('mhārō', '', 'contrasted possessive form', note='Printed “not mhārō”; not licensed here as locative.'),
    w('ghari', '', 'locative noun', note='Same physical token as whole example; full translation supplied only for phrase.')])
group(51, 'locative-noun-alternatives', 'Nouns of this class form locative singular in ē or ai, as ghōṛē, or ghōṛai, in a horse.', [
    w('ē', '', 'locative singular ending'), w('ai', '', 'locative singular ending'),
    w('ghōṛē', 'in a horse', 'masculine locative singular'), w('ghōṛai', 'in a horse', 'masculine locative singular')])
group(51, 'consonant-noun-locative', 'Masculine nouns ending in consonants form locative in i, as in ghari, in a house.', [
    w('i', '', 'masculine consonant noun locative ending'), w('ghari', 'in a house', 'masculine locative noun')])
group(51, 'dative-agent-suffix-alternatives', 'The usual suffix of the dative is nai or nē̃; same suffix also used for case of agent.', [
    w('nai', '', 'dative and agent case suffix'), w('nē̃', '', 'dative and agent case suffix')])
group(51, 'genitive-suffix', 'The genitive suffix is kō, as in Mēwātī.', [w('kō', '', 'genitive suffix', note='Mewati comparison shares this printed form; no automatic variant/borrowing relation.')])
group(51, 'agent-passive-whole-example', 'Agent case before future passive participle, same in form as infinitive: tū-nē̃ karᵃṇō thō, it was to be done by thee.', [
    w('tū-nē̃ karᵃṇō thō', 'it was to be done by thee', 'whole sentence; future passive participle construction'),
    w('tū-nē̃', '', 'pronoun in agent case', note='Analysis only; no separately printed word gloss.'),
    w('karᵃṇō', '', 'future passive participle; same form as infinitive', note='Raised a retained literally; no automatic weak-vowel substitution.'),
    w('thō', '', 'past copula', note='Shared physical token with whole sentence; separate general past gloss occurs below.')])
group(51, 'genitive-locative-dative-use', 'Locative of genitive often used for dative, e.g. mērai, to me.', [w('mērai', 'to me', 'genitive locative used for dative')])
group(51, 'occasional-neuter-example', 'Occasional instances of neuter gender, as in diṇū̃, that which is to be given.', [w('diṇū̃', 'that which is to be given', 'neuter', note='Qualified occasional-neuter claim; initial i quantity requires independent corroboration.')])
group(51, 'personal-pronoun-me-alternatives', 'ma-nē̃ as well as mū̃-nē̃, by me. Note use of nē̃ to form agent case of these pronouns.', [
    w('ma-nē̃', 'by me', 'first person singular agent pronoun'),
    w('mū̃-nē̃', 'by me', 'first person singular agent pronoun'), w('nē̃', '', 'agent case marker')])
group(51, 'possessive-pronoun-your', 'Tumhārō is “your”.', [w('Tumhārō', 'your', 'possessive pronoun', note='Source capitalization preserved; person/number not independently supplied here.')])
group(51, 'own-pronoun-alternatives', 'Own is apᵃṇū or apᵃṇō, masculine oblique apᵃṇā.', [
    w('apᵃṇū', 'own', 'possessive/reflexive meaning'), w('apᵃṇō', 'own', 'possessive/reflexive meaning'),
    w('apᵃṇā', 'own', 'masculine oblique', note='Raised a literal in all three forms.')])
group(51, 'demonstrative-this-nominative', 'yō or yoh (feminine yā), this.', [
    w('yō', 'this', 'demonstrative'), w('yoh', 'this', 'demonstrative'), w('yā', 'this', 'feminine demonstrative')])
group(51, 'demonstrative-this-oblique', 'Oblique singular aĩh or ā̃h; oblique plural in often used in sense of singular.', [
    w('aĩh', 'this', 'demonstrative oblique singular', note='Nasal attachment/first vowel quantity awaiting independent review.'),
    w('ā̃h', 'this', 'demonstrative oblique singular'), w('in', 'this', 'demonstrative oblique plural', note='Often used in sense of singular; do not erase printed plural analysis.')])
group(51, 'demonstrative-that-nominative', 'wō or woh (feminine wā), that.', [
    w('wō', 'that', 'demonstrative'), w('woh', 'that', 'demonstrative'), w('wā', 'that', 'feminine demonstrative')])
group(51, 'demonstrative-that-oblique', 'Oblique singular waĩh or wā̃h or ū̃; plural un also often used as singular.', [
    w('waĩh', 'that', 'demonstrative oblique singular', note='Nasal attachment/first vowel quantity awaiting independent review.'),
    w('wā̃h', 'that', 'demonstrative oblique singular'), w('ū̃', 'that', 'demonstrative oblique singular'),
    w('un', 'that', 'demonstrative oblique plural', note='Also often used as singular.')])
group(51, 'relative-when-example', 'Relative pronoun often used in sense of demonstrative. So jab, then, as well as “when”.', [w('jab', 'then; when', 'relative/demonstrative contextual use')])
group(51, 'qualified-oblique-relative-interrogative', 'Probably oblique singular of jō and kaun are jaĩh or jā̃h and kaĩh or kā̃h respectively, but I have not come across examples.', [
    w('jō', '', 'relative pronoun'), w('kaun', '', 'interrogative pronoun'),
    w('jaĩh', '', 'proposed relative pronoun oblique singular', 'Ahirwati', H, 'Author explicitly has not encountered examples; quantity/nasal attachment provisional.'),
    w('jā̃h', '', 'proposed relative pronoun oblique singular', 'Ahirwati', H),
    w('kaĩh', '', 'proposed interrogative pronoun oblique singular', 'Ahirwati', H, 'Author explicitly has not encountered examples; quantity/nasal attachment provisional.'),
    w('kā̃h', '', 'proposed interrogative pronoun oblique singular', 'Ahirwati', H)])
for person, number, forms in [(1,'sg',['sū̃']),(1,'pl',['sā̃']),(2,'sg',['sā','sai']),(2,'pl',['sō','saĩ']),(3,'sg',['sai']),(3,'pl',['saĩ'])]:
    group(51, f'present-copula:p{person}:{number}', 'Present verb substantive; physical '+str(person)+' '+number+' cell: '+', or '.join(forms)+'.',
          [w(f, '', f'present copula; person {person}; {number}', note='Paradigm has grammatical labels only, no printed independent English translation.') for f in forms])
group(51, 'past-copula-three-gender-number-forms', 'Past thō, was; feminine thī, masculine plural thā.', [
    w('thō', 'was', 'past copula'), w('thī', 'was', 'past feminine copula'), w('thā', 'was', 'past masculine plural copula')])
group(51, 'Bawal-Kot-Kasam-regional-copulas', 'Country round Bawal of Nabha and Kot Kasam of Jaipur: sō (sī, sā) also used as well as thō.', [
    w('sō', 'was', 'past copula', note='Explicit Bawal/Kot Kasam regional scope; not automatically Gurgaon lect.'),
    w('sī', 'was', 'parenthetical regional past copula alternative', note='Parenthesis preserved; gender pairing not explicitly stated here.'),
    w('sā', 'was', 'parenthetical regional past copula alternative', note='Parenthesis preserved; gender pairing not explicitly stated here.'),
    w('thō', 'was', 'past copula', note='Regional comparison repeated witness.')])
group(51, 'Bagri-second-plural-borrowing-claim', 'The saĩ of the second plural present is borrowed from Bāgrī.', [
    w('saĩ', '', 'second person plural present copula; source-attributed borrowing', note='Explicit Bagri donor claim. Exact donor endpoint must be checked against Sirsa/Hissar comparators; no invented graph edge.')])
group(51, 'Panjabi-comparison-whole-copulas', 'Syllable gō sometimes added to present like gā of Panjabi. sai-gō, he is, corresponding to Panjabi hai-gā.', [
    w('gō', '', 'added copular syllable'), w('gā', '', 'compared added copular syllable', 'Panjabi', 'eligible_comparator'),
    w('sai-gō', 'he is', 'present copular expression'), w('hai-gā', 'he is', 'compared copular expression', 'Panjabi', 'eligible_comparator')])
group(51, 'inherited-conjugation-closing', 'In other respects verbs are conjugated as in Mēwātī.', [])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, data):
    raw = (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode() if not path.suffix == '.jsonl' else ''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data).encode()
    if path.exists() and path.read_bytes() != raw:
        raise SystemExit(f'Refusing overwrite: {path}')
    path.write_bytes(raw)

def main():
    census = [json.loads(l) for l in (SCOPE/'physical-source-unit-census.jsonl').read_text().splitlines()]
    parents = [x for x in census if x['printed_page'] in [49,50,51]]
    assert set(GROUPS) == {x['key'] for x in parents}, (set(GROUPS)-{x['key'] for x in parents}, {x['key'] for x in parents}-set(GROUPS))
    records = []
    for parent in parents:
        for rank, row in enumerate(GROUPS[parent['key']],1):
            form,gloss,analysis,language,disposition,note = row
            records.append({'witness_key': parent['key']+f':w{rank}', 'parent_key':parent['key'],
                            'printed_page':parent['printed_page'],'source_form':form,'source_gloss':gloss,
                            'source_analysis':analysis,'language_label':language,'disposition':disposition,
                            'source_quote':CONTEXT[parent['key']], 'qualification':note,
                            'typed_uncertainties': [note] if any(s in note for s in ['awaiting','requires independent','provisional','requires independent corroboration']) else [],
                            'stage':'owner source-only first reading; not accepted output'})
    assert len({r['witness_key'] for r in records}) == len(records)
    coverage = [{**p,'source_context':CONTEXT[p['key']],'child_keys':[r['witness_key'] for r in records if r['parent_key']==p['key']],
                 'disposition':'quoted_witnesses_accounted' if GROUPS[p['key']] else 'documentary_context_no_quoted_target_word'} for p in parents]
    write(HERE/'child-witnesses-first-reading.jsonl',records)
    write(HERE/'parent-coverage-first-reading.jsonl',coverage)
    manifest=json.loads((SCOPE/'native-grammar-image-manifest.json').read_text())
    for page in manifest['pages']:
        assert sha(ROOT/page['lossless_png_path'])==page['lossless_png_sha256']
    report={'status':'complete owner first reading of all grammar census parents; independent review pending',
            'parents':len(parents),'witness_interpretations':len(records),'counts_by_disposition':{k:sum(r['disposition']==k for r in records) for k in sorted({r['disposition'] for r in records})},
            'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [SCOPE/'physical-source-unit-census.jsonl',SCOPE/'native-grammar-image-manifest.json',SCOPE/'ROOT_IDENTITY_DECISION.md',SCOPE/'ROOT_NATIVE_GRAMMAR_REVIEW.md']},
            'original_images':manifest['pages'],'script_sha256':sha(Path(__file__)),
            'output_sha256':{p.name:sha(p) for p in [HERE/'child-witnesses-first-reading.jsonl',HERE/'parent-coverage-first-reading.jsonl']},
            'cautions':['Counts are witness interpretations, not distinct physical printed tokens: whole expressions/words and explicit shared language scopes overlap.',
                        'No inferred English word alignment inside whole translations; atoms retain blank lexical gloss when source supplies only a whole translation.',
                        'No phonemic normalization or automatic suffix hyphens. Raised a encoded ᵃ; source quantity/nasal marks retained.',
                        'Four proposed oblique forms remain audit-only because author explicitly has not encountered examples.',
                        'Regional copulas are not silently assigned to Gurgaon; source Bagri borrowing claim awaits exact donor endpoint review.',
                        'Ethnonyms/language names are documentary; historical hypotheses are not target lexical attestations.',
                        'Native first-reading nasal attachment/quantity holds require independent original review.'],
            'installation':'not requested or attempted; no canonical changes'}
    write(HERE/'first-reading-status.json',report)
    print(json.dumps({'parents':len(parents),'witnesses':len(records),'counts':report['counts_by_disposition']}))

if __name__=='__main__':
    main()
