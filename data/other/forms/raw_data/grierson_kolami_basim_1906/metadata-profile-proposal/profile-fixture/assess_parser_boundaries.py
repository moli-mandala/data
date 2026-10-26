"""Run the unmodified parser and discovered YAML in a disposable local tree."""
import csv
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unicodedata as ud
import yaml
from prepare_fixture import HERE, PACKAGE, ROOT, sha, display

DATA = ROOT / "data"
FIELDS = ("lang", "param", "old_form", "gloss", "native", "ipa", "notes", "source", "cognateset", "etymology", "entry_key", "variant_of_key", "borrowed_from_key", "derivation_parent_keys", "tags")
BOUNDARIES = "1234⁴5⁵67⁷,;."


def main():
    atoms = []
    lines = {}
    pins = {}
    for page in (564, 565, 568, 569):
        path = PACKAGE / "specimens" / f"p{page}-first-reading.jsonl"
        pins[str(path.relative_to(ROOT))] = sha(path)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                lines[row["source_unit_key"]] = row
                atoms.extend(row["target_atoms"])
    assert len(atoms) == 561
    by_key = {a["source_unit_key"]: a for a in atoms}
    head_key = "grierson1906kolamibasim:p568:specimen76-line:11:word:8"
    tail_key = "grierson1906kolamibasim:p568:specimen76-line:12:word:1"
    head, tail = by_key[head_key], by_key[tail_key]
    assert (head["raw_source_form"], tail["raw_source_form"]) == ("Bāp-", "phōkān")
    joined = dict(source_unit_key=head_key + ":joined-policy-control", raw_source_form=head["raw_source_form"] + tail["raw_source_form"], raw_interlinear_gloss="father's-near")
    # Physical-line quotation control is not offered as a source sentence boundary.
    quote_line = lines["grierson1906kolamibasim:p568:specimen76-line:10"]
    quotation = dict(source_unit_key=quote_line["source_unit_key"] + ":physical-line-policy-control", raw_source_form=quote_line["raw_target_line"], raw_interlinear_gloss=quote_line["raw_aligned_english_line"])
    cases = atoms + [joined, quotation]
    rows = [["Naikri" if ":p568:" in a["source_unit_key"] or ":p569:" in a["source_unit_key"] else "Kolami", "", a["raw_source_form"], a["raw_interlinear_gloss"], "", "", "Source transcription retained; preliminary parser policy fixture.", "grierson1906kolamibasim[specimen policy fixture]", "", "", a["source_unit_key"], "", "", "", ""] for a in cases]
    profile = HERE / "grierson-kolami-basim-1906-preliminary.txt"
    codepins = {name:sha(DATA / name) for name in ("make_cldf.py", "source_meta.py", "form_grammar.py")}
    observations = []
    checks = {}
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="boundary-parser-", dir=HERE) as temporary:
        scratch = Path(temporary)
        target = scratch / "data/other/forms"
        target.mkdir(parents=True)
        (scratch / "conversion").mkdir()
        for name in ("make_cldf.py", "source_meta.py"):
            shutil.copyfile(DATA / name, scratch / name)
        shutil.copyfile(profile, scratch / "conversion/preliminary-boundary-profile.txt")
        for route in ("profile", "literal"):
            settings = dict(file=f"{route}-boundaries.csv", defaults=dict(transcription=dict(profile="preliminary-boundary-profile", input="form", preserve_hyphens=True) if route == "profile" else dict(convert=False), identity=dict(dedupe_by_entry_key=True)))
            if route == "profile":
                settings["sources"] = dict(grierson1906kolamibasim=dict(forms=dict(split_alternates=False), gloss=dict(grammar_tags=False), reference=dict(editor="OpenAI Codex", ocr=True, etymology_provenance="source")))
            (target / f"{route}-boundaries.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
        os.chdir(scratch)
        sys.path[:0] = [str(scratch), str(DATA)]
        import make_cldf
        import source_meta
        meta = source_meta.load()
        assert meta.transcription("grierson1906kolamibasim", "profile-boundaries.csv", "Kolami") == ("preliminary-boundary-profile", True)
        assert meta.transcription("grierson1906kolamibasim", "literal-boundaries.csv", "Kolami") == (None, False)
        for route in ("profile", "literal"):
            for normalization in ("NFC", "NFD"):
                inputs = [[ud.normalize(normalization, value) if i == 2 else value for i, value in enumerate(row)] for row in rows]
                path = target / f"{route}-boundaries.csv"
                with path.open("w", newline="") as stream:
                    csv.writer(stream).writerows(inputs)
                errors = io.StringIO()
                parsed, stats = make_cldf.parse_file(str(path), errors, name=f"{route}-boundaries")
                assert not errors.getvalue(), errors.getvalue()
                assert len(parsed) == len(rows) == 563
                assert stats["converted"] == (563 if route == "profile" else 0)
                for actual, expected in zip(parsed, inputs):
                    assert [getattr(actual, field) for field in FIELDS] == expected, actual.entry_key
                    expected_display = display(expected[2].strip(BOUNDARIES)) if route == "profile" else expected[2]
                    assert ud.normalize("NFC", actual.form) == ud.normalize("NFC", expected_display), (actual.entry_key, actual.form, expected_display)
                    if route == "profile" and normalization == "NFC" and expected[2].strip(BOUNDARIES) != expected[2]:
                        observations.append(dict(key=actual.entry_key, original=expected[2], boundary_stripped_form=expected[2].strip(BOUNDARIES), final_display=actual.form))
                checks[f"{route}-{normalization}"] = dict(rows=563, original_and_all15_fields_exact=True, errors=0, converted=stats["converted"], ordered_keys=True)
        os.chdir(original_cwd)
    report = dict(status="actual parser boundary behavior verified; final emission policy pending root/owner review", input_hashes=pins, profile_sha256=sha(profile), code_hashes=codepins, checks=checks, stripped_boundary_cases=len(observations), stripped_boundary_records=observations, join_control=dict(first_key=head_key, tail_key=tail_key, physical_originals=[head["raw_source_form"], tail["raw_source_form"]], candidate_joined_form=joined["raw_source_form"], expected_display="bāp-phōkān", status="parser control only; owner original-reviewed joining still required", tail_not_separate_head_in_final_candidate=True), quotation_scope="Physical line is a test control, not a source whole sentence or acceptance of synthesized translation.", schema_options=dict(profile_with_preserve_hyphens="Original exact; display strips comma/semicolon/fullstop/digits at boundaries but preserves quotes and hyphens. Internal punctuation remains literal.", convert_false="All literal boundary punctuation and capitalization survive; does not perform approved lowercase/ṅ display normalization.", input_phonemic="Not appropriate: Phonemic must remain blank without source phonemic evidence; do not populate a normalized display surrogate."), recommendation="Use existing generic profile route with preserve_hyphens:true for lexical atoms, preserve every exact raw fragment and punctuation in Original/audit, and explicitly document boundary display changes. For full utterances needing exact punctuation and casing, a separate convert:false file is schema-compatible; do not infer sentence boundaries from physical lines. No existing YAML option simultaneously preserves all terminal punctuation and applies the profile. Any stricter display policy requires root adjudication rather than a source-specific parser branch.", canonical_changes=False, monkeypatch=False, temporary_sandbox_removed=True)
    (HERE / "actual-parser-boundary-review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(checks=checks, stripped_boundary_cases=len(observations)), indent=2))


if __name__ == "__main__":
    main()
