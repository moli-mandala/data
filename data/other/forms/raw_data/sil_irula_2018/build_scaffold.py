#!/usr/bin/env python3
"""Parse the reproducible OCR pass into one record per printed site response."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITES = (
    "KUN", "KOL", "CHE", "KIL", "MET", "CHO", "MAV", "ANA", "BOO",
    "THA", "NEL", "CBT", "MAD", "KAN", "BAD", "ALU", "BET", "JEN",
)
TARGET_SITES = SITES[:11]
SITE_INDEX = {site: index for index, site in enumerate(SITES)}
HEADER = re.compile(r"^([0-9lLIi]{1,3})\.\s*(.*)$")
MARKER = re.compile(r"^@@\s+pdf(\d+)-p(\d+)-c(\d+)$")


def glosses() -> dict[int, str]:
    result = {}
    for line in (HERE / "glosses.tsv").read_text(encoding="utf-8").splitlines():
        number, gloss = line.split("\t", 1)
        result[int(number)] = gloss
    if set(result) != set(range(1, 188)):
        raise AssertionError("glosses.tsv must account for items 1-187")
    return result


def number(value: str) -> int:
    return int(value.translate(str.maketrans("lLIiOo", "111100")))


def normalize_site(token: str) -> str | None:
    cleaned = re.sub(r"[^A-Z]", "", token.upper())
    if cleaned in SITE_INDEX:
        return cleaned
    # Only accept a unique one-edit repair.  It handles OCR such as €HE while
    # refusing to guess arbitrary page-edge noise.
    def distance(a: str, b: str) -> int:
        previous = list(range(len(b) + 1))
        for i, left in enumerate(a, 1):
            current = [i]
            for j, right in enumerate(b, 1):
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
            previous = current
        return previous[-1]

    candidates = []
    for site in SITES:
        if distance(cleaned, site) <= 1:
            candidates.append(site)
    return candidates[0] if len(candidates) == 1 else None


def group_and_form(rest: str) -> tuple[str, str] | None:
    parts = rest.split(maxsplit=1)
    if len(parts) != 2:
        return None
    group = parts[0].translate(str.maketrans("lLIiOo", "111100"))
    group = re.sub(r"[^0-9,]", "", group).strip(",")
    if not group or not all(piece.isdigit() for piece in group.split(",")):
        return None
    return group, parts[1].strip()


def parse(path: Path) -> tuple[list[dict], list[dict]]:
    expected_glosses = glosses()
    records: list[dict] = []
    unexplained: list[dict] = []
    pdf_page = printed_page = column = item = None
    last_site = pending = None
    response_index: Counter[tuple[int, str]] = Counter()

    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text:
            continue
        marker = MARKER.match(text)
        if marker:
            pdf_page, printed_page, column = map(int, marker.groups())
            last_site = pending = None
            continue
        header = HEADER.match(text)
        if header:
            candidate = number(header.group(1))
            if 1 <= candidate <= 187 and (item is None or candidate >= item):
                item = candidate
                last_site = pending = None
                continue

        parts = text.split(maxsplit=1)
        site = normalize_site(parts[0]) if parts else None
        if site:
            if item is None:
                unexplained.append({"line": line_number, "raw": raw, "reason": "site before item"})
                continue
            rest = parts[1].strip() if len(parts) > 1 else ""
            last_site = site
        else:
            rest = text

        missing = rest.lower().replace(" ", "").startswith("missingdat")
        parsed = group_and_form(rest)
        if site and missing:
            group, form = "", "missing data"
        elif parsed and (site or last_site):
            site = site or last_site
            group, form = parsed
        elif site and re.fullmatch(r"[0-9lLIiOo,?§]+", rest):
            pending = (site, re.sub(r"[^0-9,]", "", rest.translate(str.maketrans("lLIiOo", "111100"))))
            continue
        elif pending and not normalize_site(parts[0] if parts else ""):
            site, group = pending
            form = text
            pending = None
        else:
            # Headers wrap after punctuation; page numbers and clipped adjacent
            # characters also occur. Keep every such line visible for review.
            if text not in {"Appendix B: Irula wordlis", "Appendix B: Irula wordlists"} and not text.isdigit():
                unexplained.append(
                    {
                        "line": line_number, "raw": raw, "reason": "non-record OCR line",
                        "pdf_page": pdf_page, "printed_page": printed_page, "column": column,
                        "item": item,
                    }
                )
            continue

        response_index[item, site] += 1
        records.append(
            {
                "pdf_page": pdf_page,
                "printed_page": printed_page,
                "column": column,
                "item": item,
                "gloss": expected_glosses[item],
                "site": site,
                "target": site in TARGET_SITES,
                "response": response_index[item, site],
                "group": group,
                "ocr": form,
                "raw_ocr": raw,
                "line": line_number,
                "site_explicit": normalize_site(parts[0]) is not None,
            }
        )

    # A dropped site code leaves a bare group+form line.  Reassign such a line
    # only when the item's site inventory proves which single site was skipped;
    # otherwise it remains an additional response for the preceding site.
    by_item = defaultdict(list)
    for record in records:
        by_item[record["item"]].append(record)
    for item_records in by_item.values():
        counts = Counter(record["site"] for record in item_records)
        missing_sites = [site for site in SITES if counts[site] == 0]
        duplicated = [site for site in SITES if counts[site] > 1]
        if len(missing_sites) == len(duplicated) == 1:
            old_site, new_site = duplicated[0], missing_sites[0]
            candidates = [r for r in item_records if r["site"] == old_site and not r["site_explicit"]]
            if len(candidates) == 1 and SITE_INDEX[new_site] == SITE_INDEX[old_site] + 1:
                candidates[0]["site"] = new_site
                candidates[0]["target"] = new_site in TARGET_SITES
                candidates[0]["response"] = 1
    return records, unexplained


def validate(records: list[dict]) -> list[str]:
    errors = []
    primary = defaultdict(list)
    for record in records:
        primary[record["item"]].append(record["site"])
    for item in range(1, 188):
        counts = Counter(primary[item])
        missing = [site for site in SITES if not counts[site]]
        if missing:
            errors.append(f"item {item}: missing site rows {','.join(missing)}")
        # Repeated sites are permitted only as additional printed responses.
        seen_order = []
        for site in primary[item]:
            if site not in seen_order:
                seen_order.append(site)
        if tuple(seen_order) != SITES:
            errors.append(f"item {item}: site order is {' '.join(seen_order)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unexplained", type=Path, required=True)
    args = parser.parse_args()
    records, unexplained = parse(args.raw)
    errors = validate(records)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.unexplained.write_text(json.dumps(unexplained, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"records={len(records)} target={sum(r['target'] for r in records)} unexplained={len(unexplained)}")
    print(f"missing={sum(r['ocr'].lower() == 'missing data' for r in records)} validation_errors={len(errors)}")
    for error in errors:
        print(error)


if __name__ == "__main__":
    main()
