"""Compile independent source-notation readings; never alter installed inputs."""
from pathlib import Path
import hashlib
import json
import runpy

HERE = Path(__file__).resolve().parent
LEDGER = HERE.parent / "manual_review_data.py"
READINGS = HERE / "notation-literal-pages-20260927.json"
ADJUDICATIONS = HERE / "notation-narrow-adjudications-20260927.json"
ENCODING = HERE / "notation-iota-encoding-review-20260927.json"


def main():
    old = {}
    for page in runpy.run_path(str(LEDGER))["ROWS"]:
        for offset, form in enumerate(page["Forms"]):
            key = (page["Site"], page["First_Item"] + offset)
            assert key not in old, key
            old[key] = (page, form)
    readings = json.loads(READINGS.read_text())
    adjudications = json.loads(ADJUDICATIONS.read_text())["records"]
    adjudicated = {r["key"]: r for r in adjudications}
    assert len(adjudicated) == len(adjudications)
    encoding_rows = json.loads(ENCODING.read_text())["records"]
    encoding = {r["key"]: r for r in encoding_rows}
    assert len(encoding) == len(encoding_rows)
    result = []
    seen = set()
    for page in sorted(readings["pages"], key=lambda p: (p["pdf_page"], p["site"])):
        for offset, literal in enumerate(page["forms"]):
            item = page["first_item"] + offset
            key = (page["site"], item)
            assert key not in seen, key
            seen.add(key)
            prior, form = old[key]
            assert prior["PDF_Page"] == page["pdf_page"]
            uncertainty = page.get("uncertainties", {}).get(str(item), "")
            source_key = f"wordlist:{key[0]}:{item}"
            adjudication = adjudicated.get(source_key)
            first_literal = literal
            encoding_review = encoding.get(source_key)
            if encoding_review:
                assert literal == encoding_review["provisional"]
                assert page["pdf_page"] == encoding_review["pdf_page"]
                literal = encoding_review["literal_glyph_candidate"]
            if adjudication:
                assert adjudication["pdf_page"] == page["pdf_page"]
                literal = adjudication["literal"]
                uncertainty = "" if adjudication["status"] == "confirmed" else adjudication["reason"]
            sign_review = None
            if any(sign in literal for sign in ("?", "ʔ")):
                sign_review = {
                    "status": "interpretation_uncertain",
                    "literal_signs": [sign for sign in ("?", "ʔ") if sign in literal],
                    "prior_review_note": uncertainty,
                    "decision": "Preserve the transcribed question-shaped sign. Its role as phonetic notation versus collector doubt is unresolved; neither interpretation is asserted. The chosen Unicode shape is not an approved phonemic analysis.",
                    "authority": "Root independent review and instruction, 2026-09-27; supersedes categorical interpretation in first-reading notes without changing their historical evidence.",
                }
                uncertainty = sign_review["decision"]
            result.append({
                "source_unit_key": source_key,
                "entry_key_prefix": f"silkoya1985:{key[0].lower()}:i{item:03d}",
                "pdf_page": page["pdf_page"], "printed_page": page["printed_page"],
                "site": key[0], "item": item, "column": prior["Column"],
                "previous_ledger_form": form, "literal_source_proposal": literal,
                "first_literal_reading": first_literal,
                "narrow_adjudication": adjudication,
                "encoding_review": encoding_review,
                "question_shaped_sign_review": sign_review,
                "normalized_ipa_proposal": None,
                "changed": form != literal,
                "review_uncertainty": uncertainty,
                "status": "uncertain_question_shaped_sign_interpretation" if sign_review else (adjudication["status"] if adjudication else ("review_note_or_pending_question" if uncertainty else "first_independent_literal_reading")),
                "installation_approved": False,
            })
    out = HERE / "notation-keyed-proposal-20260927.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result))
    sign_census = [{"source_unit_key": r["source_unit_key"], "pdf_page": r["pdf_page"], "literal_source_proposal": r["literal_source_proposal"], **r["question_shaped_sign_review"]} for r in result if r["question_shaped_sign_review"]]
    (HERE / "notation-question-shaped-sign-census-20260927.json").write_text(json.dumps({"scope": "All target wordlist readings; literal forms unchanged by this interpretation census.", "records": sign_census}, ensure_ascii=False, indent=2) + "\n")
    expected = {k for k in old if k[0] not in {"TEL", "ORI"}}
    assert set(adjudicated) <= {r["source_unit_key"] for r in result}
    assert set(encoding) <= {r["source_unit_key"] for r in result}
    summary = {
        "status": "in_progress_not_installable",
        "reviewed_target_ledger_slots": len(seen),
        "expected_target_ledger_slots": len(expected),
        "absent_slot_in_ledger": "MAL:200 is not physically printed on PDF111; original stops199, nextpage starts201.",
        "reviewed_physically_printed_target_cells": len(seen - {("MAL", 200)}),
        "remaining_target_cells": len(expected - seen),
        "changed_literal_proposals": sum(r["changed"] for r in result),
        "explicit_review_notes_or_pending_questions": sum(bool(r["review_uncertainty"]) for r in result),
        "question_shaped_sign_interpretation_uncertain_cells": len(sign_census),
        "ledger_sha256": hashlib.sha256(LEDGER.read_bytes()).hexdigest(),
        "readings_sha256": hashlib.sha256(READINGS.read_bytes()).hexdigest(),
        "adjudications_sha256": hashlib.sha256(ADJUDICATIONS.read_bytes()).hexdigest(),
        "encoding_review_sha256": hashlib.sha256(ENCODING.read_bytes()).hexdigest(),
        "proposal_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "ipa_mapping": "Not approved; literal single-dot/colon/glide distinctions preserved separately.",
        "canonical_or_parent_proposal_modified": False,
    }
    (HERE / "notation-progress-20260927.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
