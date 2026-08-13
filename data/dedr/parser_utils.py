import re


TAG = re.compile(r"<.*?>")

GRAMMATICAL_MARKERS = {
    "n": "noun",
    "v": "verb",
    "vb": "verb",
    "adj": "adj",
    "adv": "adv",
    "fem": "f",
    "masc": "m",
    "f": "f",
    "m": "m",
    "neut": "n",
    "hon": "honorific",
    "sg": "sg",
    "pl": "pl",
    "du": "du",
    "tr": "tr",
    "intr": "intr",
    "caus": "caus",
    "pass": "pass",
    "refl": "refl",
    "intens": "intensive",
    "obl": "obl",
    "nom": "nom",
    "acc": "acc",
    "dat": "dat",
    "gen": "gen",
    "loc": "loc",
    "abl": "abl",
    "instr": "instr",
    "inf": "inf",
    "impv": "impv",
    "pres": "pres",
    "fut": "fut",
    "pp": "pp",
}

ITALIC = re.compile(r"<i>(?P<content>.*?)</i>", re.DOTALL)


def _tag_text(value):
    return TAG.sub("", value).strip()


def _grammatical_marker(value):
    """Return a schema tag for one exact, lowercase grammatical abbreviation."""
    text = _tag_text(value).strip(" ()")
    if not text.endswith("."):
        return ""
    # Case matters: DEDR ``Tr.`` is a Gondi source/dialect, while ``tr.`` is
    # the transitivity label.
    return GRAMMATICAL_MARKERS.get(text[:-1], "")


def _grammatical_markers(value):
    text = _tag_text(value).strip(" ()")
    if text == "pl. action":
        return ["pl"]
    tags = []
    fields = re.findall(r"[a-z]+\.", text)
    if not fields:
        fields = re.split(r"\s*,\s*", text)
    for field in fields:
        tag = _grammatical_marker(field)
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def grammatical_tags(*html_values):
    """Extract ordered schema tags from explicitly italic grammatical labels."""
    tags = []
    for value in html_values:
        for match in ITALIC.finditer(value or ""):
            for tag in _grammatical_markers(match.group("content")):
                if tag not in tags:
                    tags.append(tag)
    return tags


def grammatical_tag(value):
    """Extract a tag from a marker-only bold lemma such as ``n.``."""
    return _grammatical_marker(value)


def marker_only_tags(value):
    """Return tags only when the entire value is grammatical-label markup."""
    text = _tag_text(value).strip(" ()")
    if text == "pl. action":
        return ["pl"]
    fields = re.findall(r"[a-z]+\.", text)
    remainder = re.sub(r"[a-z]+\.", "", text)
    if not fields or remainder.strip(" ,"):
        return []
    tags = []
    for field in fields:
        tag = _grammatical_marker(field)
        if not tag:
            return []
        if tag not in tags:
            tags.append(tag)
    return tags


def strip_grammatical_markers(value):
    """Lift italic grammatical labels out of display text after tagging them."""
    def replace(match):
        return "" if _grammatical_markers(match.group("content")) else match.group(0)

    value = ITALIC.sub(replace, value or "")
    value = re.sub(r"<i>\s*</i>", "", value)
    value = re.sub(r"\(\s*\)", "", value)
    value = re.sub(r"\s+([,;:.])", r"\1", value)
    return re.sub(r" {2,}", " ", value).strip()


def detach_leading_grammatical_note(form_html):
    """Detach ``(tr.);``-style labels that describe the preceding lemma."""
    pattern = re.compile(
        r"^\s*(?:\(\s*<i>(?P<label_out>.*?)</i>\s*\)|"
        r"<i>\s*\((?P<label_in>.*?)\)\s*</i>)\s*;\s*",
        re.DOTALL,
    )
    match = pattern.match(form_html)
    if not match:
        return [], form_html
    tags = _grammatical_markers(match.group("label_out") or match.group("label_in"))
    return (tags, form_html[match.end():]) if tags else ([], form_html)


def detach_trailing_forward_grammatical_note(gloss_html):
    """Detach a post-semicolon label that introduces the following lemma."""
    pattern = re.compile(
        r"^(?P<body>.*?;)\s*<i>(?P<label>.*?)</i>"
        r"(?P<source>\s*\([^)]*\))?\s*$",
        re.DOTALL,
    )
    match = pattern.match(gloss_html or "")
    if not match:
        return [], gloss_html
    tags = _grammatical_markers(match.group("label"))
    if not tags:
        return [], gloss_html
    return tags, (match.group("body") + (match.group("source") or "")).strip()


def split_tagged_forms(form_html, initial_tags=None):
    """Split a bold lemma run and associate local grammatical labels."""
    parts = re.split(r"([;,])(?![^()]*\))", form_html)
    active = list(initial_tags or [])
    trailing = re.search(r"<i>.*?</i>\s*$", form_html, re.DOTALL)
    if trailing:
        active = list(dict.fromkeys(active + grammatical_tags(trailing.group(0))))
    result = []
    sticky = {"noun", "verb", "adj", "adv"}
    delimiter = ""
    for part in parts:
        if part in {",", ";"}:
            delimiter = part
            if part == ";":
                active = [tag for tag in active if tag in sticky]
            continue
        if not part.strip():
            continue
        markers = grammatical_tags(part)
        prefix = re.match(r"^\s*(?:\(\s*)?<i>.*?</i>(?:\s*\))?\s*", part, re.DOTALL)
        if prefix and markers:
            active = markers[:]
        tags = list(dict.fromkeys(active + markers))
        form, _ = clean_form_html(part, "")
        form = re.sub(r"^\([^)]*\)\s*", "", form).strip()
        if form:
            result.append((form, tags))
        delimiter = ""
    return result


def split_language_spans(section, abbreviations):
    """Split one entry section at language labels, including malformed HTML.

    The DEDR source frequently leaves a surrounding ``<b>`` open across a
    language boundary.  BeautifulSoup then nests the following language in the
    preceding lemma, so DOM ancestry/source positions are not reliable.  The
    language labels themselves are consistently italicised; scanning their raw
    markup preserves the intended boundary.
    """
    labels = sorted(abbreviations, key=len, reverse=True)
    label_pattern = "|".join(re.escape(label) for label in labels)
    label_at_end = re.compile(rf"(?P<prefix>.*?)(?P<label>{label_pattern})\.?$")
    markers = []

    for match in re.finditer(r"<i>(?P<content>.*?)</i>", section, re.DOTALL):
        text = _tag_text(match.group("content"))
        found = label_at_end.fullmatch(text)
        source_suffix = ""
        if not found:
            source_marker = re.fullmatch(
                rf"(?P<label>{label_pattern})\.?\s*(?P<suffix>\([^)]*\))", text
            )
            if not source_marker:
                continue
            prefix = ""
            label = source_marker.group('label')
            source_suffix = source_marker.group('suffix')
        else:
            prefix = found.group("prefix")
            label = found.group("label")
        # A malformed bold run can put the preceding grammatical note in the
        # same italic element, e.g. ``<i>(tr.). Kui</i>``.  Do not interpret
        # ordinary italic prose ending in a short language-like token.
        if prefix and (len(prefix) > 40 or not re.search(r"[.)]\s*$", prefix)):
            continue
        markers.append((match.start(), match.end(), label, prefix.strip(), source_suffix))

    # A few entries mark an uncertain language as plain text (``? Go. (Mu.)``)
    # rather than italics.  The parenthetical followed by a bold lemma makes
    # this narrow fallback distinguishable from cross-references such as
    # ``with 11 Ta. akar``.
    uncertain = re.compile(
        rf"\?\s*(?P<label>{label_pattern})\.?\s*(?=\([^)]*\)\s*<b>)"
    )
    for match in uncertain.finditer(section):
        if any(start <= match.start() < end for start, end, *_ in markers):
            continue
        markers.append((match.start(), match.end(), match.group("label"), "", ""))

    markers.sort()
    spans = []
    section_label = section[: markers[0][0]] if markers else section
    for index, (start, end, label, _, source_suffix) in enumerate(markers):
        next_start = markers[index + 1][0] if index + 1 < len(markers) else len(section)
        content = (source_suffix + ' ' if source_suffix else '') + section[end:next_start]
        if index + 1 < len(markers):
            next_prefix = markers[index + 1][3]
            if next_prefix:
                prefix = next_prefix.removesuffix('.').strip()
                content += f"<i>{prefix}</i>"
        spans.append([label, content])
    return _tag_text(section_label), spans


def split_forms(value):
    """Split a bold lemma run without splitting punctuation in parentheses."""
    pieces = re.split(r"[;,](?![^()]*\))", value)
    return [piece.strip() for piece in pieces if piece.strip()]


def split_alternatives(value):
    """Split slash alternatives, but preserve slashes inside parentheses."""
    return [piece.strip() for piece in re.split(r"/(?![^()]*\))", value)]


def propagate_shared_glosses(rows, fallback=""):
    """Fill definitionless forms from the nearest defined form in a section.

    Compact DEDR entries frequently state a meaning once and then give bare
    equivalents for several further languages.  Ties prefer the preceding
    definition, matching the dictionary's normal forward-sharing convention.
    """
    # Parenthetical material in the gloss position is commonly an inflection
    # or source displaced by malformed markup, not a definition.
    for row in rows:
        detail = _tag_text(row[3]).strip()
        if re.fullmatch(r"\([^)]*\)", detail) or (fallback and detail.startswith("(")):
            row[2] = f"{row[2]} {row[3]}".strip()
            row[3] = ""

    defined = [
        index for index, row in enumerate(rows)
        if row[3].strip() and not _tag_text(row[3]).lstrip().startswith("(")
    ]
    if not defined:
        if fallback:
            for row in rows:
                if not row[3].strip():
                    row[3] = fallback
        return rows
    for index, row in enumerate(rows):
        if not row[3].strip():
            nearest = min(defined, key=lambda pos: (abs(pos - index), pos > index))
            row[3] = rows[nearest][3]
    return rows


def mark_unbolded_compact_forms(span):
    """Mark a bare comma-list of forms in compact cross-language entries."""
    if '<b>' in span or '</b>' in span:
        return span
    match = re.fullmatch(r"\s*(?P<body>.+?)\.\s*", span, re.DOTALL)
    if not match:
        return span
    parts = [part.strip() for part in re.split(r",(?![^()]*\))", match.group('body'))]
    if not parts or any(not part for part in parts):
        return span
    # Bare compact forms contain no prose-like multiword segment.  Spaces used
    # around a middle dot and inside a parenthetical variant are harmless.
    for part in parts:
        plain = re.sub(r"\([^)]*\)", "", _tag_text(part))
        plain = re.sub(r"\s*[··]\s*", "·", plain).strip()
        if re.search(r"\s", plain):
            return span
    return f"<b>{match.group('body')}</b>."


def clean_form_html(form_html, gloss):
    """Return cleaned lemma markup and gloss, repairing common source markup."""
    # Botanical names are occasionally italicised *inside* the bold lemma,
    # with no separately marked gloss (``virigi <i>Cordia sebestena</i>``).
    italic_suffix = re.fullmatch(r"(.*?)\s*<i>([^<]+)</i>\s*", form_html, re.DOTALL)
    if italic_suffix and len(italic_suffix.group(2).split()) >= 2:
        form_html = italic_suffix.group(1)
        gloss = " ".join(filter(None, (italic_suffix.group(2).strip(), gloss)))

    form = _tag_text(strip_grammatical_markers(form_html))
    marker_names = "|".join(sorted(map(re.escape, GRAMMATICAL_MARKERS), key=len, reverse=True))
    form = re.sub(rf"^(?:{marker_names})\.\s*", "", form)
    # A language boundary can cut an unclosed bold run immediately after its
    # sentence punctuation: ``<b><i>fem.</i> polati. <i>Tu.</i>``.
    if grammatical_tags(form_html) and re.fullmatch(r"[^\s.]+\.", form):
        form = form[:-1]
    return form.strip(), gloss


def mark_unbolded_gender_forms(span):
    """Bold forms that follow an italic ``fem.``/``masc.`` source label."""
    pattern = re.compile(
        r"<i>(?P<label>fem|masc)\.</i>\s*(?P<form>[^<.;]+)(?P<end>[.;])"
    )

    def replace(match):
        before = span[:match.start()]
        # A language span can start inside a bold element and therefore have
        # unbalanced tag counts.  The nearest boundary remains reliable.
        if before.rfind('<b>') > before.rfind('</b>'):
            return match.group(0)
        form = match.group('form').strip()
        if not form or form.startswith(')'):
            return match.group(0)
        return f"<b><i>{match.group('label')}.</i> {form}</b>{match.group('end')}"

    return pattern.sub(replace, span)


def is_prose_misparse(raw_form):
    """Recognise prose made bold only because the source HTML is malformed."""
    raw_form = raw_form.strip()
    if re.match(r"^\s*<i>.*?</i>", raw_form, re.DOTALL) and grammatical_tags(raw_form):
        marked_forms = split_tagged_forms(raw_form)
        if marked_forms and all(re.fullmatch(r"\S+", form) for form, _ in marked_forms):
            return False
    if raw_form.endswith("."):
        return True
    text = _tag_text(raw_form)
    if re.fullmatch(r"[A-Z]\.\s+[a-z][a-z-]+", text):
        return True
    return text.lower() in {"kill", "proof", "split (tr.)"}
