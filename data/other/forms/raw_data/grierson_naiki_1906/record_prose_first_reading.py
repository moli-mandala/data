"""Literal first-reading inventory; not an importer or installation approval.

Groups retain the source's scope. Separators are explicit alternatives only
where stated in prose; full phrases are not automatically split into words.
Original pages were inspected alongside the Norwegian keyed transcription.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# page | group | printed form(s) | source gloss/analysis | disposition | caveat
READINGS = '''570|noun-plural-rational|kōr|plural suffix for rational beings|target-affix|Source says seems to be added.
570|noun-plural-other|l|plural suffix|target-affix|
570|children|pōrākōr|children|target|
570|swine|turrēl|swine|target|
570|buffaloes|sirlā|buffaloes|target|
570|me-to-me|ānun|me; to me|target|Accusative and dative do not appear distinguished.
570|case-suffixes|n / un / kun|dative and accusative suffixes|target-affix|Preserve separately printed alternatives; no reconstructed word.
570|daughter|pōrīn|to the daughter|target|
570|father|bānun|to the father|target|
570|field|vāvart-un|to the field|target|
570|servants|chākarkun|to the servants|target|
570|belly|pōṭāl|to the belly|target|
570|towards-father|bānēkil|towards the father|target|
570|genitive|nē|genitive suffix|target-affix|
570|locative|lōpul|locative suffix|target-affix|
570|heaven|akās-nē|of heaven|target|
570|country|dēsam-lōpul|in the country|target|
570|one|oko|one|target|Source explicitly cannot distinguish long and short e/o.
570|two|iroṭēr|two|target|Source explicitly cannot distinguish long and short e/o.
570|two-neuter|yeraṇḍi|two|target|Explicit neuter label.
570|four|nāli|four|target|
570|one-proposed|ōkō|one|qualified-source-reading|Author proposes possible vowel-length reading; not an independently attested alternate.
570|two-proposed|irōṭēr|two|qualified-source-reading|Author proposes possible vowel-length reading; not an independently attested alternate.
570|telugu-one|oka|one|control-Telugu|
570|tamil-two|iraṇḍu|two|control-Tamil|
570|kui-I|ānu|I|control-Kui|
570|kui-we|āmu|we|control-Kui|
570|kui-thou|īnu|thou|control-Kui|
570|she-it|ād|she; it|target|One form explicitly serves both genders; do not select one sense silently.
570|she-it-genitive|aunē|genitive of she/it|target|Separate printed scope from masculine pronoun table witness.
570|they|Add|they|target|Source capitalized at sentence start; says occurs in one place.
570|this|Ivun|this|target|Source capitalized; inflected as aun, but no inflected forms printed here.
570|relative|Aun|relative pronoun|target|Source says seems to be used as a relative pronoun.
570|relative-clause|aun hissā ānun varlēn ād sī|which share me-to comes that give|target-expression|Explicit aligned prose example; component segmentation requires separate review.
570|who|ēn|who?|target|
570|what|tā|what?|target|
570|indefinite-suffix|ī / nī|indefinite pronoun suffixes|target-affix|
570|anyone|yēnī|anyone|target|
570|anything|tānī|anything|target|
570|am-art-is|anlēn|I am; thou art; it is|target|Source provisionally states verbs do not change for person; keep all three readings.
570|shall-do|kaknām|we shall do|target|
570|were|aṇḍēr|they were|target|
570|fell|paṭṭul / paṭṭun|it fell|target|Two explicitly printed forms; shared gloss.
571|present-marker|l|characteristic consonant of present tense|target-affix|Source says seems; not an asserted independent lexical root.
571|past-marker|t|characteristic consonant of past tense|target-affix|Combined with preceding consonant in various ways; also used present/future below.
571|gives|siylēn|it gives|target|
571|gave|sitēn|he gave|target|
571|went|seddēn|he went|target|
571|said|yeṇḍēn|he said|target|
571|doing|kākēntān|I am doing|target|
571|will-go|sērtān|I will go|target|
571|personal-ending|n|usual personal termination|target-affix|Not assigned a unique person by this statement.
571|was-ante|antē|it was|target|
571|was-andur|aṇḍur|he was; she was|target|Source calls r-ending forms properly plural despite supplied singular translations.
571|was-anter|antēr|it was|target|Source calls r-ending forms properly plural despite supplied singular translation.
571|were|aṇḍēr|they were|target|
571|said-questioned|Iūn|he said|qualified-source-reading|Author explicitly says probably wrong; preserve printed reading and uncertainty.
571|negative-marker|ē|negative verb marker added to root|target-affix|
571|went-not|sēr-ē-n|he went not|target|
571|gave-not|siy-ē-tēn|he gave not|target|
571|broke-not|mōḍāp-ē-tān|I broke not|target|
571|gavest-not|siy-ē-tūn|thou gavest not|target|
571|give|sī|give|target|Explicit imperative.
571|put|īḍ|put|target|Explicit imperative.
571|let-see|Āykēkād|let us see|target|Source states translated as, without a morphological analysis.
571|put-not|tābgārē|he should not put|target|Source calls negative imperative.
571|quarrel-not|jhagḍīlnēr|do not quarrel|target|Source calls negative imperative.
571|infinitive-suffixes|n / l|infinitive suffixes|target-affix|
571|feed|mēpēn|in order to feed|target|
571|saying-for|eṇḍēn-sāṭī|saying-for|target|
571|do|kakēl / kakkēn|to do|target|Shared printed gloss.
571|conjunctive-suffixes|tun / un|conjunctive participle endings|target-affix|
571|collected|jamāktun|having collected|target|
571|gone|seddun|having gone|target|
571|eating-conjunctive|tinnun|eating|target|Explicitly listed under conjunctive participles.
571|called|karūktu|having called|target|Other participle.
571|eating-other|tinnun / tindunu|eating|target|Other participles; retain this separate source occurrence.
571|eating-tinchin|tiñchin|eating|target|Other participle.
571|playing|aḍsīn|playing|target|Other participle.
571|taking|tōnān|taking|target|Other participle.
571|agency-noun|Kaktēn|noun of agency|qualified-source-analysis|Source says apparently; do not treat surrounding perfect translation as a gloss for this component.
571|done-phrase|kakṭēn anlēn|I have done|target-expression|Keyed transcription has ṭ in this phrase versus t in surrounding examples; final glyph review required.
571|sin-phrase|pāp kaktēn anlēn|I am one who did sin|target-expression|Preserve attributed agency-noun interpretation.
'''

def main():
    rows = []
    for line in READINGS.splitlines():
        if not line:
            continue
        page, group, form, gloss, disposition, caveat = line.split('|')
        rows.append(dict(source_unit_key=f'grierson1906naiki:p{page}:prose:{group}',
                         printed_page=int(page), physical_djvu_page=int(page)+20,
                         raw_form_group=form, raw_gloss_or_analysis=gloss,
                         disposition=disposition, source_caveat=caveat,
                         review_status='root first reading; independent original/output review pending',
                         canonical_changed=False))
    assert len({r['source_unit_key'] for r in rows}) == len(rows)
    (HERE/'prose-first-reading.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    print(f'{len(rows)} prose evidence groups; canonical unchanged')

if __name__ == '__main__':
    main()
