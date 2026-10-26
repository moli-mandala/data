"""Focused tests for the ordered sound-change applier (sound_changes/sca.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sound_changes"))

import sca  # noqa: E402


def run(rules_text: str, form: str) -> str:
    classes: dict = {}
    rules = []
    for n, line in enumerate(rules_text.strip().splitlines(), 1):
        line = sca.strip_comment(line).strip()
        if line.startswith("class "):
            name, _, members = line[6:].partition("=")
            classes[name.strip()] = sca.expand_members(members.split(), classes)
        elif line:
            rules.append(sca.parse_rule(line, n, classes))
    return "".join(sca.apply_rules(rules, sca.tokenize(form)))


def test_segments_are_profile_units():
    assert sca.tokenize("áṣāḍʰā") == ("á", "ṣ", "ā", "ḍʰ", "ā")
    assert sca.tokenize("*kr̩tá-") == ("k", "r̩", "t", "á")


def test_gold_tokens_split_hiatus_and_diaeresis():
    assert sca.gold_tokens("tarai") == ("t", "a", "r", "a", "i")
    assert sca.gold_tokens("kaüa") == ("k", "a", "u", "a")
    assert sca.gold_tokens("agaru(a)") == ("a", "g", "a", "r", "u")


def test_environment_and_boundary():
    assert run("a > ∅ / _ #", "bʰakta") == "bʰakt"
    assert run("{k t} > ∅ / V _ V", "kataka") == "kaaa"


def test_simultaneous_application_reads_rule_input():
    # both t's are intervocalic in the input, even though deleting the first changes neighbours
    assert run("t > ∅ / V _ V", "atata") == "aaa"


def test_parallel_sets_and_at_mapping():
    assert run("{ṭ ṭʰ} > {ḍ ḍʰ} / V _ V", "paṭʰa") == "paḍʰa"
    assert run("{t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1", "madʰya") == "majjʰa"


def test_backreference_modifiers():
    assert run("V > $1-acc", "áṣṭā́") == "aṣṭā"
    assert run("class Stop = k t tʰ\ns Stop > $2-asp $2+asp", "hasta") == "hattʰa"
    assert run("C C > $2-asp $2", "dʰarma") == "dʰamma"
    assert run("{ā} > $1-long / _ C C", "mārga") == "marga"


def test_insertion_negation_optional_and_exception():
    assert run("∅ > i / # C _ l", "glāni") == "gilāni"
    assert run("n > ṇ / !# _", "nanu") == "naṇu"
    assert run("t > d / V (C) _ #", "ant") == "and"
    assert run("n > ṇ // _ t", "nanta") == "ṇanta"


def test_comments_do_not_swallow_boundaries():
    assert run("a > ∅ / _ #   ; drop final a", "kata") == "kat"


def test_stacked_modifiers_use_profile_mark_order():
    assert run("V ṁ > $1+long+nas / _ C", "piṁḍa") == "pī̃ḍa"
    assert sca.tokenize(run("V ṁ > $1+long+nas / _ C", "piṁḍa"))[1] == "ī̃"


def test_same_segment_target_matches_geminates_only():
    rules = "class G = k kʰ t tʰ n h\nV G $2 > $1+long $3"
    assert run(rules, "hattʰa") == "hātʰa"
    assert run(rules, "lihna") == "lihna"


def test_class_names_expand_inside_sets():
    assert run("class Nasal = n m\nclass Stop = g t\nStop {r Nasal} > $1 $1", "agni") == "aggi"


def test_gold_strips_vowel_stress_but_not_consonant_acute():
    assert sca.gold_tokens("naś") == ("n", "a", "ś")
    assert sca.gold_tokens("guniyá²") == ("g", "u", "n", "i", "y", "a")
    # vowel qualities are never merged (ʌ stays ʌ), spelling variants are unified, tone marks dropped
    assert sca.gold_tokens("pʌrkumʌn") == ("p", "ʌ", "r", "k", "u", "m", "ʌ", "n")
    assert sca.gold_tokens("kʰεr") == ("kʰ", "ɛ", "r") and sca.gold_tokens("gâṛ") == ("g", "a", "ṛ")
    assert sca.gold_tokens("kə̃dʰo") == ("k", "ə̃", "dʰ", "o")


def test_source_blocks_are_separate_from_the_chain(tmp_path):
    p = tmp_path / "OIA-x.sc"
    p.write_text("a > e\n%source webster\ne > i / _ #   ; @notation\n%end\n", encoding="utf-8")
    rules, tails = sca.load_rules_full(str(p))
    assert [r.text for r in rules] == ["a > e"]
    assert [r.text for r in tails["webster"]] == ["e > i / _ #"]
    assert sca.load_meta(str(p))["source_tails"].keys() == {"webster"}


def test_star_environments_first_and_final_syllable_and_harmony():
    assert run("a > o / # C* _ C* a", "karman") == "korman"
    assert run("a > o / # C* _ C* a", "kaman") == "koman"
    assert run("a > o / # C* _ C* a", "kāman") == "kāman"
    assert run("a > ∅ / _ C* #", "kamala") == "kamal"
    assert run("a > ∅ / _ C* #", "kamalan") == "kamaln"


def test_accent_classes_pretonic_posttonic_and_long():
    assert run("a > ∅ / _ !#* Acc", "kapotá") == "kpotá"
    assert run("{a ā} > ə / Acc !#* _", "rákṣati") == "rákṣəti"
    assert run("Long > $1-long", "kāma") == "kama"
    assert run("Short > $1+long / # C* _", "kama") == "kāma"


def test_if_conditions_use_word_flags():
    rules = "if:verb a t i > n ā / _ #\nif:!verb a > ā / _ #"
    classes: dict = {}
    parsed = [sca.parse_rule(line, n, classes) for n, line in enumerate(rules.splitlines(), 1)]
    verb = sca.Word(sca.tokenize("likʰati"), {"verb"})
    noun = sca.Word(sca.tokenize("kamala"), set())
    assert "".join(sca.apply_rules(parsed, verb)) == "likʰnā"
    assert "".join(sca.apply_rules(parsed, noun)) == "kamalā"
    assert sca.apply_rules(parsed, verb).flags == {"verb"}


def test_optional_target_tokens():
    assert run("(a) (y) a t i > i k / _ #", "tārayati") == "tārik"
    assert run("(a) (y) a t i > i k / _ #", "patati") == "patik"


def test_same_segment_in_exception_and_set_difference():
    assert run("class Stop = k t p\nStop > ∅ / V _ // _ $1", "akkaata") == "akkaaa"
    assert run("class Stop = k t p\nclass NoT = Stop - t\nNoT > ∅ / V _ V", "akata") == "aata"
    assert run("{k t p - t} > ∅ / V _ V", "akata") == "aata"


def test_accented_diphthongs_and_braces_tokenize_as_units():
    assert sca.tokenize("saínya")[1] == "aí"
    assert sca.tokenize("*ga{l}lati") == sca.tokenize("*gallati")
    assert sca.canon(("s", "aí", "n")) == ("s", "a", "í", "n")


def _chain(rules_text: str, form: str):
    rules = [sca.parse_rule(sca.strip_comment(l).strip(), n, {})
             for n, l in enumerate(rules_text.strip().splitlines(), 1) if sca.strip_comment(l).strip()]
    return sca.apply_rules(rules, sca.tokenize(form))


def test_accent_gold_marks_location_only_when_the_source_writes_it():
    assert sca.accent_tokens("aʦ̣ʰī̂") == ("a", "ʦ̣ʰ", "ī́")        # circumflex, caron, grave → one acute
    assert sca.accent_tokens("hagā̀r") == ("h", "a", "g", "ā́", "r")
    assert sca.accent_tokens("agār") is None                      # unmarked: not accent-scored
    assert sca.accent_tokens("śaś") is None                       # the acute of ś is not an accent
    # one accent penalty (0.5), whether the accent is misplaced or missing
    assert sca.accent_edit_distance(("a", "g", "ā́", "r"), ("á", "g", "ā", "r")) == 0.5
    assert sca.accent_edit_distance(("a", "g", "ā", "r"), ("a", "g", "ā́", "r")) == 0.5
    assert sca.accent_edit_distance(("a", "g", "ā́"), ("a", "g", "ā́", "r")) == 1.0
    assert sca._acc_vowel("aî") == "aí" and sca._acc_vowel("ái") == "ái"   # diphthong keeps its accent site


def test_latent_accent_is_invisible_to_rules_and_carried_by_them():
    out = _chain("V > $1-acc\n{ai} > {ē}\nm > ∅ / _ #", "taílam")
    assert "".join(out) == "tēla"                                  # rules match plain vowels
    assert sca.realize(out, "vedic") == ("t", "ḗ", "l", "a")        # the accent moves onto ē
    out = _chain("V > $1-acc\na > ∅ / _ #", "tailá")
    assert sca.realize(out, "vedic") == ("t", "ai", "l")            # deleted with its vowel
    out = _chain("V > $1-acc\na > ∅ / _ #", "áśva")
    assert sca.realize(out, "chain") == ("a", "ś", "v")            # chain model ignores the Vedic accent
    assert sca.realize(out, "vedic") == ("á", "ś", "v")


def test_stress_placement_wins_and_never_changes_segments():
    out = _chain("V > $1-acc\nV > $1+stress / _ C* V C* #\n{a i u} > ∅ / V C* _ #", "aŋgúli")
    assert "".join(out) == "aŋgul"
    assert sca.realize(out, "vedic") == ("a", "ŋ", "g", "ú", "l")   # rule accent beats the Vedic one
    assert sca.deacc_pred(sca.realize(out, "chain")) == tuple(out)


def test_stress_class_reads_the_latent_tier_for_accent_shifts():
    rules = """V > $1-acc
V > $1+stress / _ C* V C* #
V > $1+stress / {bʰ dʰ gʰ jʰ ḍʰ h} Stress C* _
Stress > $1-stress / {bʰ dʰ gʰ jʰ ḍʰ h} _ C* V"""
    assert sca.realize(_chain(rules, "gʰaná"), "chain") == ("gʰ", "a", "n", "á")   # shifted right
    assert sca.realize(_chain(rules, "kā́ka"), "chain") == ("k", "ā́", "k", "a")    # plain onset: stays


def test_extension_menu_replaces_the_final_vowel_and_skips_verbs():
    import evaluate as ev
    w = sca.Word(sca.tokenize("ghṓṭa"), frozenset())
    assert "".join(ev.extend(w, "aka")) == "ghṓṭaka"
    assert "".join(ev.extend(sca.Word(sca.tokenize("kanyā"), frozenset()), "ikā")) == "kanyikā"
    assert "verb" not in ev.extend(sca.Word(sca.tokenize("karōti"), frozenset({"verb"})), "aka").flags   # nominal derivative
    assert "ext" in ev.extend(w, "akam").flags                        # citation rules can say if:!ext
    assert {"akka", "ikka", "iyaka", "akam"} <= set(ev.EXT_MENU)


def _central(tmp_path, monkeypatch, text):
    """A throwaway changes.sc (the live chains no longer use one; the engine still supports `use`)."""
    c = tmp_path / "changes.sc"
    c.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sca, "CENTRAL", str(c))


def test_use_lines_load_central_changes_with_central_classes(tmp_path, monkeypatch):
    _central(tmp_path, monkeypatch, "class G = k kʰ t tʰ\nchange degem-plain.g = G $1 > $2   ; @degem-plain — demo\n")
    defs = sca.central()
    name = next(n for n, (r, _) in defs.items() if r.text == "G $1 > $2")
    p = tmp_path / "OIA-X.sc"
    p.write_text("class G = k\nuse " + name + "   ; local note\n", encoding="utf-8")
    (rule,) = sca.load_rules(str(p))
    assert rule.name == name and rule.text == defs[name][0].text   # the central text and classes, not the local G
    assert "".join(sca.apply_rules([rule], sca.tokenize("atta"))) == "ata"


def test_verb_form_menu_builds_on_the_root():
    import evaluate as ev
    w = sca.Word(sca.tokenize("trōṭáyati"), frozenset({"verb"}))
    assert "".join(ev.verb_form(w, "anaka")) == "trōṭanaka"
    assert "".join(ev.verb_form(sca.Word(sca.tokenize("jānā́ti"), frozenset({"verb"})), "ita")) == "jānita"
    assert "".join(ev.verb_form(sca.Word(sca.tokenize("ghōṭa"), frozenset()), "anaka")) == "ghōṭanaka"   # every headword gets the verb menu
    V = lambda f: sca.Word(sca.tokenize(f), frozenset({"verb"}))
    assert "".join(ev.verb_form(V("jita"), "anaka")) == "jitanaka"          # participle headword: final vowel dropped
    assert "".join(ev.verb_form(V("*bukka"), "anaka")).endswith("bukkanaka")  # starred root
    assert "".join(ev.verb_form(V("pacyatēaya"), "anaka")) == "pacyanaka"   # -tē + aya causative
    assert "".join(ev.verb_form(V("viśáti"), "anaka")) == "viśanaka"        # ś keeps its acute (not s)


def test_derivation_columns_keep_the_first_of_a_degeminated_pair():
    import derive
    cols = derive.Columns(("t", "u", "m", "b", "a"))
    cols.step(("t", "u", "m", "m", "a"), "r1")          # b > m
    cols.step(("t", "u", "m", "a"), "r2")               # mm > m: the copy (old b) is lost
    ety = [(c["ety"], c["cur"], c["by"]) for c in cols.cols]
    assert ety[2] == ("m", "m", []) and ety[3] == ("b", None, ["r1", "r2"])


def test_derivation_variant_step_replaces_the_final_vowel_after_the_stem():
    import derive
    cols = derive.Columns(("g", "ʰ", "o", "ṭ", "a"))
    cols.prefix_step(("g", "ʰ", "o", "ṭ", "a", "k", "a"), "+aka")   # common prefix ghoṭa kept whole
    assert [c["ety"] for c in cols.cols] == ["g", "ʰ", "o", "ṭ", "a", None, None]
    cols = derive.Columns(("p", "ī", "t", "a", "l", "a"))
    cols.prefix_step(("p", "ī", "t", "a", "l", "i", "y", "a", "k", "a"), "+iyaka")
    # the headword's final a is replaced; the suffix columns follow it, not precede it
    assert [(c["ety"], c["cur"]) for c in cols.cols][5:7] == [("a", None), (None, "i")]


def test_derivation_steps_round_trip_through_the_edit_encoding():
    import derive
    start = ["a", "b", "c"]
    steps = [["r1", ["a", "x", "c"]], ["r2", ["a", "x", "c", "d"]], ["r3", ["x", "c", "d"]]]
    text = derive.encode_steps(start, steps)
    assert text == "r1 1 1 x ; r2 3 0 d ; r3 0 1"
    assert derive.decode_steps(start, text) == [(r, f) for r, f in steps]


def test_derive_chain_traces_a_real_form():
    import derive
    items = [("f1", "Bi", "e1", "mḗtʰiṣṭʰa", "", "mehṭā", "CDIAL")]
    chain, out, table = derive.derive_chain(("Bi", "Bi", items))
    (d,) = out
    # the trace ends on the prediction, and the steps are real chain rules in chain order
    assert d["steps"][-1][1] == d["pred"]
    orders = {r["Rule_Ref"]: r["Order"] for r in table}
    rules = [s[0] for s in d["steps"] if not s[0].startswith(("+", "v:", "d:"))]  # menu steps (+aka) aren't chain rules
    assert [orders[r] for r in rules] == sorted(orders[r] for r in rules)
    # every etymon segment has one column, in order
    assert [c[0] for c in d["cols"] if c[0] is not None] == d["etymon"]


def test_menu_offers_both_menus_to_every_headword():
    """Unified score (2026-09-30): every headword gets the bare form, every noun extension and every
    verb formation; verb formations carry `verb`, noun extensions drop it."""
    import evaluate as ev
    noun = sca.Word(sca.tokenize("ārta"), frozenset())
    verb = sca.Word(sca.tokenize("karōti"), frozenset({"verb"}))
    for w in (noun, verb):
        names = [n for n, _ in ev.menu(w)]
        assert names[0] == "bare" and "aka" in names and "v:anaka" in names
    m = dict(ev.menu(noun))
    assert "".join(m["v:anakam"]) == "ārtanakam" and "verb" in m["v:anakam"].flags
    assert "".join(dict(ev.menu(verb))["v:anaka"]) == "karanaka"
    assert "verb" not in dict(ev.menu(verb))["aka"].flags


def test_verb_menu_directive_adds_lect_specific_formations(tmp_path):
    p = tmp_path / "OIA-X.sc"
    p.write_text("%verb-menu +alla +aka\nclass G = k\n", encoding="utf-8")
    assert sca.load_meta(str(p))["verb_menu"] == ("alla", "aka")
    q = tmp_path / "OIA-Y.sc"
    q.write_text("%ext-menu +iyakam\n", encoding="utf-8")
    assert sca.load_meta(str(q))["ext_menu"] == ("iyakam",)


def test_survey_clause_glosses_mark_verbs():
    import evaluate as ev
    toks = sca.tokenize("māráyati")
    assert "verb" in ev.word_flags("māráyati", toks, ["he killed"])
    assert "verb" in ev.word_flags("píbati", sca.tokenize("píbati"), ["drink!, he drank"])
    assert "verb" not in ev.word_flags("māráyati", toks, ["killing, murder"])


def test_tone_score_charges_tone_type_separately_from_position():
    g = sca.tone_tokens("kǎra")
    P = lambda *t: sca.tone_pred(t)
    assert sca.tone_edit_distance(P("k", "ǎ", "r", "a"), g) == 0.0
    assert sca.tone_edit_distance(P("k", "â", "r", "a"), g) == sca.TONE_COST       # right vowel, wrong type
    assert sca.tone_edit_distance(P("k", "a", "r", "á"), g) == sca.ACCENT_COST     # wrong vowel
    assert sca.tone_tokens("kara") is None                                        # unmarked: not scored


def test_vedic_feature_reads_only_the_carried_vedic_accent():
    # after accent loss the Vedic accent survives only on the latent tier; `Vedic` matches that vowel
    acc_loss = sca.parse_rule("V > $1-acc", 1, {})
    lengthen = sca.parse_rule("Vedic > $1+long", 2, {})
    out = sca.apply_rules([acc_loss, lengthen], sca.tokenize("kárana"))
    assert "".join(sca.deacc_pred(sca.realize(out, "chain"))) == "kārana"


def test_split_off_marks_on_ipa_vowels():
    # the profile splits combining marks off ɨ ə ʊ; accents must still be dropped (plain) / read (accent)
    assert sca.gold_tokens("kɨ́r") == ("k", "ɨ", "r")
    assert sca.gold_tokens("kə̃dʰo") == ("k", "ə̃", "dʰ", "o")
    assert sca.accent_tokens("kɨ́r") == ("k", "ɨ́", "r")


def test_accented_greek_epsilon_is_unified():
    assert sca.gold_tokens("bέʦε") == ("b", "ɛ", "ʦ", "ɛ")
    assert sca.accent_tokens("bέʦε") == ("b", "ɛ́", "ʦ", "ɛ")


def test_realize_keeps_rule_written_tone():
    # a rule-placed accent must not overwrite a tone a later rule wrote (falling/low used to become high)
    assert sca.realize((sca.stressed("â", "rule"), "k")) == ("â", "k")
    assert sca.realize((sca.stressed("à", "rule"),)) == ("à",)
    assert sca.realize((sca.stressed("a", "rule"),)) == ("á",)


def test_gold_drop_applies_to_accent_and_tone():
    # Strand's ǐ is a glide, not a rising tone: `%gold-drop caron` removes it from every score
    caron = frozenset({"̌"})
    assert sca.accent_tokens("gǐu") == ("g", "í", "u")
    assert sca.accent_tokens("gǐu", caron) is None          # no accent left: not accent-scored
    assert sca.tone_tokens("gǐú", caron) == ("g", "i", "ú")


def test_canon_splits_marked_diphthongs():
    # nasal/long/accented a+i/u tokens split like plain ai/au (gold ãũṭab vs predicted ã ũ ṭ a b)
    assert sca.canon(("ãũ", "ṭ", "a", "b")) == ("ã", "ũ", "ṭ", "a", "b")
    assert sca.canon(("ai", "áu", "ãĩ", "āi", "ei")) == ("a", "i", "á", "u", "ã", "ĩ", "ā", "i", "ei")


def test_cdial_long_centralised_o_is_not_a_tone():
    # ɔ̌ is the house spelling of Turner's ɔ̈̄: no accent, no tone
    assert sca.accent_tokens("yɔ̌k") is None
    assert sca.gold_tokens("yɔ̌k") == ("y", "ɔ", "k")


def test_gold_drop_per_source(tmp_path):
    p = tmp_path / "OIA-x.sc"
    p.write_text("%gold-drop caron @strand\n%gold-drop grave\na > a\n", encoding="utf-8")
    meta = sca.load_meta(str(p))
    assert meta["gold_drop"] == frozenset({"̀"})
    assert meta["gold_drop_src"] == {"strand": frozenset({"̌"})}


def test_verb_flag_from_bare_or_punctuated_gloss():
    import evaluate as ev
    flag = lambda e, g: "verb" in ev.word_flags(e, sca.tokenize(e), g)
    assert flag("ā́nayati", ["bring"])                      # Kullui: bare English verb
    assert flag("muñcáti", ["releases, leaves behind."])   # 3rd person with a comma after it
    assert not flag("*jʰaṭati", ["broom"])


def test_verb_flag_skips_sense_numbers():
    import evaluate as ev
    assert "verb" in ev.word_flags("kʰā́dati", sca.tokenize("kʰā́dati"), ["1) eat; 2) bite"])


def test_verb_flag_for_bare_roots_and_tag_parsing():
    import evaluate as ev
    flag = lambda e, g: "verb" in ev.word_flags(e, sca.tokenize(e), g)
    assert flag("*cāh", ["desires"]) and not flag("vā́c", ["speech"]) and not flag("vā́c", [])
    # an @name inside the note is not a tag
    assert sca.comment_tags(" @hiatus @x — the @morph rule was removed") == ("@hiatus", "@x")


def test_bare_pronoun_gloss_is_not_a_verb():
    import evaluate as ev
    assert "verb" not in ev.word_flags("asmad", sca.tokenize("asmad"), ["we"])
    assert "verb" in ev.word_flags("*mārati", sca.tokenize("*mārati"), ["he killed"])


def test_include_expands_a_block_in_place(tmp_path, monkeypatch):
    blocks = tmp_path / "blocks"; blocks.mkdir()
    (blocks / "b.sc").write_text("class X = p t\nX > b          ; @demo — block rule\n", encoding="utf-8")
    monkeypatch.setattr(sca, "BLOCKS", str(blocks)); sca.block_lines.cache_clear()
    lect = tmp_path / "OIA-x.sc"
    lect.write_text("a > e\ninclude b      ; the block\ni > u\n", encoding="utf-8")
    rules = sca.load_rules(str(lect))
    assert [r.text for r in rules] == ["a > e", "X > b", "i > u"]
    assert rules[1].origin == "b:2" and rules[1].line == 2 and "@demo" in rules[1].tags
    assert sca.raw_line(rules[1], lect.read_text().split("\n")).startswith("X > b")
    sca.block_lines.cache_clear()


def test_optional_rules_are_scored_with_and_without(tmp_path, monkeypatch):
    _central(tmp_path, monkeypatch, "change assim-r.rt = r {t d} > {t d}@2 {t d}   ; @assim-r — demo\n")
    r = sca.parse_rule("optional: r {t d} > {ṭ ḍ}@2 {ṭ ḍ}", 1, {})
    assert r.optional
    forms = [f for f, _ in sca.apply_rules_multi([r], sca.tokenize("vartatē"))]
    assert ["".join(f) for f in forms] == ["vaṭṭatē", "vartatē"]                        # forks only where it applies
    assert len(sca.apply_rules_multi([r], sca.tokenize("kapāla"))) == 1
    lect = tmp_path / "OIA-x.sc"
    lect.write_text("optional: use assim-r.rt\n", encoding="utf-8")
    (rule,) = sca.load_rules(str(lect))
    assert rule.optional and rule.name == "assim-r.rt"


def test_several_outcome_rule_forks_into_each_outcome():
    """`X > A | B`: the multi-derivation forks into every outcome, a single derivation takes the first,
    and rules_taken names the outcome a fork used."""
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".sc", delete=False, encoding="utf-8") as f:
        f.write("{s ś ṣ} m > mʰ | s s\n")
    try:
        rules = sca.load_rules(f.name)
    finally:
        os.unlink(f.name)
    forks = {"".join(w): sk for w, sk in sca.apply_rules_multi(rules, sca.tokenize("asmad"))}
    assert set(forks) == {"amʰad", "assad"}
    assert "".join(sca.apply_rules(rules, sca.tokenize("asmad"))) == "amʰad"
    assert sca.rules_taken(rules, forks["assad"])[0].text.endswith("s s")


def test_family_file_selects_each_lects_lines(tmp_path):
    """A family file: `[A B]` lines belong to those lects, `[-C]` to all but C, `%set` names a subset,
    unprefixed lines to every lect; a lect is still addressed as rules/OIA-<stem>.sc."""
    fam = tmp_path / "fam.sc"
    fam.write_text("%lects A B C\n%set ab = A B\n"
                   "a > e          ; everyone\n"
                   "[B] i > u      ; @demo — B only\n"
                   "[-C] o > u\n"
                   "[ab -A] k > g\n"
                   "[C] %ext-menu +ka\n", encoding="utf-8")
    texts = lambda s: [r.text for r in sca.load_rules(str(tmp_path / f"OIA-{s}.sc"))]
    assert texts("A") == ["a > e", "o > u"]
    assert texts("B") == ["a > e", "i > u", "o > u", "k > g"]
    assert texts("C") == ["a > e"]
    assert sca.load_meta(str(tmp_path / "OIA-C.sc"))["ext_menu"] == ("ka",)
    assert sca.load_meta(str(tmp_path / "OIA-A.sc"))["ext_menu"] == ()
    (rule,) = [r for r in sca.load_rules(str(tmp_path / "OIA-B.sc")) if r.text == "i > u"]
    assert rule.line == 4 and "@demo" in rule.tags
    assert sca.raw_line(rule, sca.source_lines(str(tmp_path / "OIA-B.sc"))).startswith("i > u")
    assert [os.path.basename(p) for p in sca.lect_paths(str(tmp_path))] == ["OIA-A.sc", "OIA-B.sc", "OIA-C.sc"]
    assert sca.has_chain(str(tmp_path / "OIA-B.sc")) and not sca.has_chain(str(tmp_path / "OIA-D.sc"))


def test_family_selector_rejects_unknown_lects(tmp_path):
    (tmp_path / "fam.sc").write_text("%lects A B\n[A Bx] a > e\n", encoding="utf-8")
    try:
        sca.load_rules(str(tmp_path / "OIA-A.sc"))
    except ValueError as e:
        assert "Bx" in str(e)
    else:
        raise AssertionError("a typo in a selector must not silently drop the line")


def test_one_lect_file_wins_over_family(tmp_path):
    """A real OIA-<stem>.sc (a candidate or snapshot) is read as before, even beside a family file."""
    (tmp_path / "fam.sc").write_text("%lects A\na > e\n", encoding="utf-8")
    (tmp_path / "OIA-A.sc").write_text("a > o\n", encoding="utf-8")
    assert [r.text for r in sca.load_rules(str(tmp_path / "OIA-A.sc"))] == ["a > o"]
    try:
        sca.require_one_lect_file(str(tmp_path / "OIA-A.sc"))
        (tmp_path / "OIA-A.sc").unlink()
        sca.require_one_lect_file(str(tmp_path / "OIA-A.sc"))
    except ValueError as e:
        assert "family" in str(e)
    else:
        raise AssertionError("tools that rewrite a chain file must refuse a family lect")


def test_family_merge_reproduces_each_lect():
    """families.merge_group: each lect's lines, in order, are exactly its own (shared lines once)."""
    import families
    seqs = {"A": ["x > y", "a > e", "o > u"], "B": ["a > e", "i > u", "o > u"], "C": ["o > u", "a > e"]}
    rows = families.merge_group(seqs)
    for s, want in seqs.items():
        assert [r for lects, r in rows if s in lects] == want
    assert sum(1 for lects, _ in rows if len(lects) >= 2) >= 2


def test_family_insert_and_drop_lects(tmp_path):
    """families.insert_lines writes one selected line for lects sharing a position;
    drop_lects takes lects off matching lines (deleting a line no lect keeps)."""
    import families
    fam = tmp_path / "fam.sc"
    fam.write_text("%lects A B C\na > e\n[B C] i > u\no > u\n", encoding="utf-8")
    P = lambda s: str(tmp_path / f"OIA-{s}.sc")
    texts = lambda s: [r.text for r in sca.load_rules(P(s))]
    # before each lect's own `o > u` (index 2 for A: "%lects…", "a > e", "o > u")
    families.insert_lines([(P("A"), 2, "k > g"), (P("C"), 3, "k > g")])
    assert texts("A") == ["a > e", "k > g", "o > u"] and texts("C") == ["a > e", "i > u", "k > g", "o > u"]
    assert texts("B") == ["a > e", "i > u", "o > u"]
    assert fam.read_text().count("k > g") == 1                      # one line, selected for A and C
    families.drop_lects(str(fam), r"^k > g", ["A"])
    assert texts("A") == ["a > e", "o > u"] and "k > g" in texts("C")
    families.drop_lects(str(fam), r"^o > u", ["B"])
    assert texts("B") == ["a > e", "i > u"] and "[-B] o > u" in fam.read_text()
