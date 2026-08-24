#!/usr/bin/env python3
"""Recover curated etymology links from the final legacy NeoJambu SQLite database.

The legacy web database stored manual ancestry directly in ``lemmas.origin_lemma_id``.  Those
relations were not part of the raw lexical CSVs and were therefore omitted when Jambu moved to
``data/etymology-assignments.csv``.  This importer resolves legacy IDs through the durable alias
table, falls back to conservative exact source/language/form matching, and installs only links
whose child currently has no accepted etymology.  Modern accepted links are never overwritten.

By default the command is a dry run.  Pass ``--install`` to update the assignment table and write
the complete compressed audit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT.parents[1] / "neojambu" / "data.db"
FORMS = ROOT / "cldf/forms.csv"
EDGES = ROOT / "cldf/edges.csv"
ALIASES = ROOT / "cldf/form-id-aliases.csv"
ASSIGNMENTS = ROOT / "data/etymology-assignments.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260820-neojambu-etymology-restoration-audit.csv.gz"
SUMMARY = ROOT / "data/other/forms/raw_data/20260820-neojambu-etymology-restoration-summary.json"

ASSIGNMENT_FIELDS = ["Form_ID", "Etymon_ID", "Kind", "Rank", "Status", "Source", "Notes"]
AUDIT_FIELDS = [
    "Status", "Legacy_Form_ID", "Legacy_Etymon_ID", "Resolved_Form_ID",
    "Resolved_Etymon_ID", "Form_Resolution", "Etymon_Resolution", "References",
    "Language_ID", "Original", "Gloss", "Restored_Kind", "Current_Parent_ID", "Reason",
]
DRAVIDIAN_CLADES = {
    "Old Dravidian", "S. Dravidian I", "S. Dravidian II", "C. Dravidian",
    "N. Dravidian", "Brahui",
}
INDO_ARYAN_CLADES = {
    "OIA", "MIA", "Early NIA", "Nuristani", "Pashai", "Chitrali", "Shinaic",
    "Kohistani", "Kunar", "Kashmiric", "Sindhic", "Lahndic", "Punjabic",
    "W. Pahari", "C. Pahari", "E. Pahari", "Eastern", "Bihari", "E. Hindi",
    "W. Hindi", "Rajasthanic", "Gujaratic", "Bhil", "Khandeshi", "Marathi-Konkani",
    "Halbic", "Insular", "Migratory",
}
SURVEY_REFERENCES = {
    "chattisgarhi", "bagri", "dhundari", "hadothi", "marwari", "mewari", "mewati",
}


def normalized(value: str | None) -> str:
    return " ".join(unicodedata.normalize("NFC", value or "").split())


def compact_form(value: str | None) -> str:
    """Normalize editorial separators that changed during survey re-ingestion."""
    return re.sub(r"[\s\-‐‑‒–—]+", "", normalized(value))


def folded_form(value: str | None) -> str:
    """Comparison key that ignores combining phonetic detail but retains base symbols."""
    return "".join(
        char for char in unicodedata.normalize("NFD", compact_form(value))
        if unicodedata.category(char) != "Mn"
    )


def citation_keys(value: str | None) -> set[str]:
    return {
        part.strip().split("[", 1)[0]
        for part in (value or "").split(";")
        if part.strip().split("[", 1)[0]
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(buffer.getvalue().encode("utf-8"))


class Resolver:
    def __init__(
        self,
        forms: dict[str, dict[str, str]],
        aliases: dict[str, str],
        old_lemmas: dict[str, sqlite3.Row],
        old_references: dict[str, set[str]],
    ) -> None:
        self.forms = forms
        self.aliases = aliases
        self.old_lemmas = old_lemmas
        self.old_references = old_references
        self.cache: dict[str, tuple[str, str]] = {}
        self.strict: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
        self.loose: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        self.compact: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
        self.by_reference_lect: dict[tuple[str, str], set[str]] = defaultdict(set)
        for form_id, row in forms.items():
            original = normalized(row.get("Original") or row.get("Form"))
            gloss = normalized(row.get("Gloss"))
            dialect_lects = {
                match.group(2)
                for match in re.finditer(r"(?:^|\s)dialect:([^:\s]+):([^:\s]+):", row.get("Tags", ""))
            }
            # A survey row's Language_ID is now often the shared base language while the
            # legacy row used the individual survey lect.  Index the latter exclusively when
            # present; indexing the base as well makes identical forms from every lect collide.
            lects = dialect_lects or {row.get("Language_ID", "")}
            for reference in citation_keys(row.get("Source")):
                for lect in lects:
                    self.strict[(reference, lect, original, gloss)].add(form_id)
                    self.loose[(reference, lect, original)].add(form_id)
                    self.compact[(reference, lect, compact_form(original), gloss)].add(form_id)
                    self.by_reference_lect[(reference, lect)].add(form_id)

    def resolve_children(self, legacy_id: str) -> tuple[list[str], str]:
        """Resolve a legacy child, allowing a formerly merged survey row to fan out by sense."""
        form_id, method = self.resolve(legacy_id)
        if form_id:
            return [form_id], method
        if legacy_id not in self.old_lemmas or not re.fullmatch(r"\d+-\d+", legacy_id):
            return [], method
        legacy_references = self.old_references.get(legacy_id, set())
        if not legacy_references.intersection(SURVEY_REFERENCES):
            return [], method

        row = self.old_lemmas[legacy_id]
        language = row["language_id"] or ""
        old_forms = {
            compact_form(part)
            for part in re.split(r"\s*;\s*", row["original"] or row["word"] or "")
            if part
        }
        old_glosses = {
            normalized(part)
            for part in re.split(r"\s*;\s*", row["gloss"] or "")
            if part
        }
        candidates: set[str] = set()
        for reference in legacy_references:
            for candidate in self.by_reference_lect.get((reference, language), set()):
                current = self.forms[candidate]
                current_forms = {
                    compact_form(part)
                    for part in re.split(
                        r"\s*;\s*", current.get("Original") or current.get("Form") or ""
                    )
                    if part
                }
                if not current_forms.intersection(old_forms):
                    continue
                current_gloss = normalized(current.get("Gloss"))
                if current_gloss in old_glosses:
                    candidates.add(candidate)
        if candidates:
            return sorted(candidates), "source-language-composite-form-gloss-expansion"

        old_folded = folded_form(row["original"] or row["word"])
        for reference in legacy_references:
            for candidate in self.by_reference_lect.get((reference, language), set()):
                current = self.forms[candidate]
                if normalized(current.get("Gloss")) != normalized(row["gloss"]):
                    continue
                if folded_form(current.get("Original") or current.get("Form")) == old_folded:
                    candidates.add(candidate)
        if candidates:
            return sorted(candidates), "source-language-diacritic-folded-form-gloss"

        gloss_only: set[str] = set()
        for reference in legacy_references:
            gloss_only.update(
                candidate
                for candidate in self.by_reference_lect.get((reference, language), set())
                if normalized(self.forms[candidate].get("Gloss")) == normalized(row["gloss"])
            )
        if len(gloss_only) == 1:
            return sorted(gloss_only), "unique-source-language-gloss"

        # The Chhattisgarhi legacy UI stored a small set of manually lemmatised verb roots
        # (kʰɐ- 'eat', dʒəʋ- 'eat', tʃɐb- 'bite'), while the retained survey rows contain their
        # inflected elicitation forms.  Project each root's curated parent only onto same-lect,
        # same-prompt forms beginning with that explicitly marked bound root.
        original = normalized(row["original"] or row["word"])
        if "chattisgarhi" in legacy_references and original.endswith("-"):
            root = compact_form(original)
            for reference in legacy_references:
                for candidate in self.by_reference_lect.get((reference, language), set()):
                    current = self.forms[candidate]
                    if normalized(current.get("Gloss")) != normalized(row["gloss"]):
                        continue
                    if compact_form(current.get("Original") or current.get("Form")).startswith(root):
                        candidates.add(candidate)
            if candidates:
                return sorted(candidates), "source-language-bound-root-prefix-expansion"
        return [], method

    def resolve(self, legacy_id: str) -> tuple[str, str]:
        if legacy_id in self.cache:
            return self.cache[legacy_id]
        positional_id = bool(re.fullmatch(r"\d+-\d+", legacy_id))
        if legacy_id in self.old_lemmas and positional_id:
            # These IDs were row positions, not durable identities.  Later source insertions
            # reused them for unrelated forms, so neither a same-looking current ID nor the
            # modern alias table is evidence.  Resolve them from lexical/source identity only.
            result = self.resolve_lexically(legacy_id)
        elif legacy_id in self.forms:
            result = (legacy_id, "current-id")
        elif self.aliases.get(legacy_id) in self.forms:
            result = (self.aliases[legacy_id], "legacy-alias")
        elif legacy_id in self.old_lemmas:
            result = self.resolve_lexically(legacy_id)
        else:
            result = ("", "unmatched")
        self.cache[legacy_id] = result
        return result

    def resolve_lexically(self, legacy_id: str) -> tuple[str, str]:
        row = self.old_lemmas[legacy_id]
        original = normalized(row["original"] or row["word"])
        gloss = normalized(row["gloss"])
        language = row["language_id"] or ""
        strict_matches: set[str] = set()
        loose_matches: set[str] = set()
        compact_matches: set[str] = set()
        for reference in self.old_references.get(legacy_id, set()):
            strict_matches.update(self.strict.get((reference, language, original, gloss), set()))
            loose_matches.update(self.loose.get((reference, language, original), set()))
            compact_matches.update(
                self.compact.get((reference, language, compact_form(original), gloss), set())
            )
        if len(strict_matches) == 1:
            result = (next(iter(strict_matches)), "unique-source-language-form-gloss")
        elif len(loose_matches) == 1:
            result = (next(iter(loose_matches)), "unique-source-language-form")
        elif len(compact_matches) == 1:
            result = (next(iter(compact_matches)), "unique-source-language-compact-form-gloss")
        elif strict_matches or loose_matches or compact_matches:
            result = ("", "ambiguous")
        else:
            result = ("", "unmatched")
        return result


def load_legacy(path: Path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    lemmas = {
        row["id"]: row
        for row in connection.execute(
            "SELECT id, word, gloss, original, language_id, origin_lemma_id FROM lemmas"
        )
    }
    references: dict[str, set[str]] = defaultdict(set)
    for row in connection.execute("SELECT lemma_id, reference_id FROM lemma_reference"):
        references[row["lemma_id"]].add(row["reference_id"])
    connection.close()
    return lemmas, references


def restored_kind(
    form_id: str, etymon_id: str, forms: dict[str, dict[str, str]], clades: dict[str, str]
) -> str:
    child = forms[form_id]
    parent = forms[etymon_id]
    child_clade = clades.get(child["Language_ID"], "")
    parent_clade = clades.get(parent["Language_ID"], "")
    if child_clade in DRAVIDIAN_CLADES and parent["Language_ID"] != "PDr":
        return "borrowed"
    if parent_clade in INDO_ARYAN_CLADES and child_clade not in INDO_ARYAN_CLADES:
        return "borrowed"
    return "reflex"


def plan_restoration(db_path: Path, restored_existing: set[tuple[str, str]] | None = None):
    restored_existing = restored_existing or set()
    forms = {row["ID"]: row for row in read_csv(FORMS)}
    clades = {row["ID"]: row["Clade"] for row in read_csv(ROOT / "cldf/languages.csv")}
    aliases = {row["Legacy_ID"]: row["Form_ID"] for row in read_csv(ALIASES)}
    old_lemmas, old_references = load_legacy(db_path)
    resolver = Resolver(forms, aliases, old_lemmas, old_references)

    restored_children = {child for child, _ in restored_existing}
    current_rank1 = {
        row["Child_ID"]: row
        for row in read_csv(EDGES)
        if row["Rank"] == "1" and row["Kind"] in {"reflex", "borrowed", "variant"}
        and row["Child_ID"] not in restored_children
    }
    linked = [row for row in old_lemmas.values() if row["origin_lemma_id"]]
    prepared = []
    valid_by_child: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in linked:
        form_ids, form_method = resolver.resolve_children(row["id"])
        etymon_id, etymon_method = resolver.resolve(row["origin_lemma_id"])
        # A legacy lemma sometimes represented several elicitation senses that the current
        # source correctly keeps as separate entries.  Audit and restore each resolved child.
        for form_id in form_ids or [""]:
            item: dict[str, object] = {
                "legacy": row,
                "form_id": form_id,
                "etymon_id": etymon_id,
                "form_method": form_method,
                "etymon_method": etymon_method,
            }
            prepared.append(item)
            if form_id and etymon_id and forms[etymon_id].get("Status") != "unlinked":
                valid_by_child[form_id].append(item)

    restore_edges: set[tuple[str, str]] = set()
    audit_rows = []
    status_counts: Counter[str] = Counter()
    reference_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in prepared:
        row = item["legacy"]
        assert isinstance(row, sqlite3.Row)
        form_id = str(item["form_id"])
        etymon_id = str(item["etymon_id"])
        current = current_rank1.get(form_id)
        reason = ""
        if not form_id:
            status = "unresolved-child"
            reason = str(item["form_method"])
        elif not etymon_id:
            status = "unresolved-etymon"
            reason = str(item["etymon_method"])
        elif forms[etymon_id].get("Status") == "unlinked":
            status = "unresolved-etymon"
            reason = "resolved target is itself unlinked"
        else:
            parents = {str(entry["etymon_id"]) for entry in valid_by_child[form_id]}
            if len(parents) > 1:
                status = "legacy-merge-conflict"
                reason = "legacy forms collapsed to one current form with different parents"
            elif current and current["Parent_ID"] == etymon_id:
                status = "already-present"
            elif current:
                status = "current-link-preserved"
                reason = f"current accepted {current['Kind']} edge takes precedence"
            else:
                status = "restored"
                restore_edges.add((form_id, etymon_id))

        refs = sorted(old_references.get(row["id"], set()))
        status_counts[status] += 1
        for reference in refs:
            reference_counts[reference][status] += 1
        audit_rows.append(
            {
                "Status": status,
                "Legacy_Form_ID": row["id"],
                "Legacy_Etymon_ID": row["origin_lemma_id"],
                "Resolved_Form_ID": form_id,
                "Resolved_Etymon_ID": etymon_id,
                "Form_Resolution": item["form_method"],
                "Etymon_Resolution": item["etymon_method"],
                "References": ";".join(refs),
                "Language_ID": row["language_id"] or "",
                "Original": row["original"] or row["word"] or "",
                "Gloss": row["gloss"] or "",
                "Restored_Kind": restored_kind(form_id, etymon_id, forms, clades)
                if form_id in forms and etymon_id in forms else "",
                "Current_Parent_ID": current["Parent_ID"] if current else "",
                "Reason": reason,
            }
        )

    audit_rows.sort(key=lambda row: (row["Status"], row["Legacy_Form_ID"]))
    return forms, clades, restore_edges, audit_rows, status_counts, reference_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if not args.db.exists():
        raise FileNotFoundError(f"legacy NeoJambu database not found: {args.db}")

    existing = read_csv(ASSIGNMENTS)
    restored_existing = {
        (row["Form_ID"], row["Etymon_ID"])
        for row in existing
        if row.get("Notes") == "Restored from legacy NeoJambu origin_lemma_id"
        and row.get("Status", "accepted").lower() in {"accepted", "yes", "active"}
    }
    forms, clades, restore_edges, audit_rows, status_counts, reference_counts = plan_restoration(
        args.db, restored_existing
    )
    existing = [
        row for row in existing
        if row.get("Notes") != "Restored from legacy NeoJambu origin_lemma_id"
    ]
    existing_keys = {
        (row["Form_ID"], row["Etymon_ID"], row.get("Kind", "reflex"), row.get("Rank", "1"))
        for row in existing
        if row.get("Status", "accepted").lower() in {"accepted", "yes", "active"}
    }
    additions = [
        {
            "Form_ID": form_id,
            "Etymon_ID": etymon_id,
            "Kind": restored_kind(form_id, etymon_id, forms, clades),
            "Rank": "1",
            "Status": "accepted",
            "Source": "",
            "Notes": "Restored from legacy NeoJambu origin_lemma_id",
        }
        for form_id, etymon_id in sorted(restore_edges)
        if not any(key[0] == form_id and key[1] == etymon_id and key[3] == "1" for key in existing_keys)
    ]

    digest = hashlib.sha256(args.db.read_bytes()).hexdigest()
    summary = {
        "legacy_database": str(args.db),
        "legacy_database_sha256": digest,
        "legacy_link_rows": len({row["Legacy_Form_ID"] for row in audit_rows}),
        "audit_rows": len(audit_rows),
        "unique_restored_edges": len(restore_edges),
        "restored_assignment_rows": len(additions),
        "status_counts": dict(sorted(status_counts.items())),
        "reference_counts": {
            reference: dict(sorted(counts.items()))
            for reference, counts in sorted(reference_counts.items())
            if counts.get("restored") or counts.get("unresolved-child") or counts.get("unresolved-etymon")
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.install:
        return

    assignments = existing + additions
    assignments.sort(
        key=lambda row: (
            row["Form_ID"], int(row.get("Rank") or 1), row["Etymon_ID"], row.get("Kind", "")
        )
    )
    write_csv(ASSIGNMENTS, ASSIGNMENT_FIELDS, assignments)
    write_gzip_csv(AUDIT, AUDIT_FIELDS, audit_rows)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"installed {len(additions):,} assignments; wrote {len(audit_rows):,} audit rows")


if __name__ == "__main__":
    main()
