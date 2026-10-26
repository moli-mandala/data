"""Scholarly source qualifications for Chapter 4, separated from review history."""

PROSE_NOTES = {
    'p81:ch4:prose:l43:a1': 'The prose gives -ani; subsequent printed derivations have different endings.',
    'p85:ch4:prose:l61:a1': 'The author attributes the historical development discussion to Masica.',
    'p88:ch4:prose:l35:a2': 'The source identifies verbal number agreement.',
    'p88:ch4:prose:l67:a1': 'Reported for Nawalparasi by Dawadi (2040 VS); the author says this form was not found in his own corpus.',
    'p88:ch4:prose:l67:a2': 'Reported for Nawalparasi by Dawadi (2040 VS); the author says this form was not found in his own corpus.',
    'p90:ch4:prose:l88:a1': 'The source attributes this plural possessive use to Tanahun and explicitly excludes it from Chitwan.',
    'p90:ch4:prose:l88:a2': 'The source attributes this plural possessive use to Tanahun, not Chitwan.',
    'p92:ch4:prose:l29:a1': 'The author discusses non-human use and subsequently permits human use as well.',
    'p92:ch4:prose:l29:a2': 'The author discusses non-human use and subsequently permits human use as well.',
    'p92:ch4:prose:l29:a3': 'The author discusses non-human use and subsequently permits human use as well.',
    'p92:ch4:prose:l31:a1': 'The original places the underdot beneath a in -otạ; its intended interpretation is uncertain.',
    'p93:ch4:prose:l62:a1': 'In human use this classifier agrees in gender, according to the author.',
    'p93:ch4:prose:l63:a1': 'The source excludes non-human use and marks example 33f unacceptable.',
    'p104:ch4:prose:l68:a1': 'The author tentatively describes this use as an oblique phrase meaning by means of.',
    'p105:ch4:prose:l43:a1': 'The author identifies this postposition as borrowed from Nepali.',
    'p106:ch4:prose:l50:a1': 'The author explicitly gives this form without nasalization.',
    'p106:ch4:prose:l51:a1': 'The author describes an epenthetic vowel in this form.',
    'p109:ch4:prose:l72:a1': 'The printed benefactive form has a, unlike the allative ləɡi.',
    'p110:ch4:prose:l81:a1': 'The author describes this use as possession.',
    'p113:ch4:prose:l38:a1': 'High-grade honorific; the author says its etymology is difficult to trace.',
    'p113:ch4:prose:l41:a1': 'The source reports address to kin, equals and children.',
    'p113:ch4:prose:l46:a1': 'The source reports address to daughters or sisters-in-law.',
    'p113:ch4:prose:l46:a2': 'The source reports high-honorific address to strangers or in-laws.',
    'p113:ch4:prose:l54:a1': 'The author discusses a speculative etymology proposed by van Driem, not an established borrowing.',
    'p114:ch4:prose:l49:a2': 'Proximal honorific according to the following source paragraph.',
    'p114:ch4:prose:l49:a3': 'Proximal honorific according to the following source paragraph.',
    'p114:ch4:prose:l49:a4': 'Distal honorific.',
    'p114:ch4:prose:l50:a1': 'Distal honorific.',
    'p119:ch4:prose:l51:a1': 'The author describes this as an alternative to the repeated reflexive construction.',
    'p120:ch4:prose:l75:a1': 'The author reports no reciprocal constructions in his narrative corpus and supplies an elicited example.',
    'p120:ch4:prose:l75:a2': 'The author reports no reciprocal constructions in his narrative corpus and supplies an elicited example.',
}


def scholarly_notes(row):
    key = row['source_unit_key'].removeprefix('dhakal2011:')
    notes = [PROSE_NOTES[key]] if key in PROSE_NOTES else []
    if row.get('record_type') == 'prose' and row.get('note') == 'Reported Darai1996 Tanahun forms.':
        notes.append('The author cites Darai (1996) for this Tanahun form.')
    if 'dukʰədukʰə retained' in row.get('note', ''):
        notes.append('The table prints dukʰədukʰə with a happiness gloss, whereas the following example prints dukʰəsukʰə.')
    if 'non-honorific' in row.get('tags', []):
        notes.append('The source classifies this form as non-honorific.')
    return ' '.join(notes)
