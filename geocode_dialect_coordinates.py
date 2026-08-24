#!/usr/bin/env python3
"""Geocode dialect localities and apply a fully reviewed decision table.

Named localities are looked up once with Nominatim and cached. Application is
allowed only after every missing row has an explicit, provenance-bearing entry
in ``data/dialect-coordinate-decisions.csv``; there is no automatic parent-point
fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).parent
DIALECTS = ROOT / "cldf/dialects.csv"
LANGUAGES = ROOT / "cldf/languages.csv"
CACHE = ROOT / "data/dialect-coordinate-geocoding.json"
AUDIT = ROOT / "data/dialect-coordinate-geocoding-audit.csv"
DECISIONS = ROOT / "data/dialect-coordinate-decisions.csv"
USER_AGENT = "Jambu-dialect-metadata/1.0 (one-time scholarly dataset repair)"

COUNTRY_CODES = {
    "Afghanistan": "af",
    "Bangladesh": "bd",
    "India": "in",
    "Malaysia": "my",
    "Nepal": "np",
    "Pakistan": "pk",
    "Tajikistan": "tj",
}

NON_PLACE = re.compile(
    r"^(?:\?|MN|RMF|RAKW|AKM IFM|IFM Laspur|MNN IF|MNN IWA|MNN; WSiC\)?|"
    r"MNN; WSiC; RKB|MS MA|RAKR \(1988\)|RAKR\. IWA|RAKR; WSiC|RKB IF|"
    r"RKB; MA|SWKA MS|SWKA\) RKB|ZK \(in (?:story|tale)\)|ZK (?:story|tale)|"
    r"RK story|short story .+|Source: .+|< Prs\.)$",
    re.IGNORECASE,
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def country_code(location: str) -> str | None:
    for country, code in COUNTRY_CODES.items():
        if country in location:
            return code
    return None


def clean_query(row: dict[str, str]) -> str | None:
    location = row["Location"].strip()
    if NON_PLACE.fullmatch(location):
        return None
    # Source commentary is useful in the registry but harmful to a geocoder.
    location = location.split(";", 1)[0]
    location = re.sub(r"\s+\((?:historically|source |Western |Central |Dewas |Done/).*$", "", location)
    location = re.sub(r"\s+women$", "", location, flags=re.IGNORECASE)
    location = location.replace(" NP,", ",")
    location = re.sub(
        r"\b(?:rural municipality|municipality|district|division|VDC|taluk|Circle)\b",
        "",
        location,
        flags=re.IGNORECASE,
    )
    location = re.sub(r"\s+,", ",", location)
    location = re.sub(r"\s{2,}", " ", location).strip(" ,")
    if row["Language_ID"] == "Kho":
        location = re.sub(r"^Proper Chitral$", "Chitral", location)
        location = re.sub(r"^Upper Chitral$", "Upper Chitral District", location)
        location = re.sub(r"^Lower Chitral$", "Lower Chitral District", location)
        if not location.endswith("Pakistan"):
            location += ", Chitral, Khyber Pakhtunkhwa, Pakistan"
    elif row["ID"] == "brahui_rakhshan":
        location = "Rakhshan, Balochistan, Pakistan"
    elif row["ID"] == "ThuiYasin":
        location = "Thui, Yasin Valley, Gilgit-Baltistan, Pakistan"
    elif row["ID"] == "Yasin":
        location = "Darkot, Yasin Valley, Gilgit-Baltistan, Pakistan"
    elif row["ID"] == "Ishkoman":
        location = "Imit, Ishkoman Valley, Gilgit-Baltistan, Pakistan"
    elif row["ID"] == "lsi_gypsyeuropean":
        return None
    elif row["ID"] == "lsi_easternbengali":
        return None
    return location


def load_cache() -> dict[str, list[dict[str, object]]]:
    if not CACHE.exists():
        return {}
    return json.loads(CACHE.read_text(encoding="utf-8"))


def save_cache(cache: dict[str, list[dict[str, object]]]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch(query: str, code: str | None) -> list[dict[str, object]]:
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": "3",
        "addressdetails": "1",
        "layer": "address,natural",
    }
    if code:
        params["countrycodes"] = code
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def collect() -> None:
    _, rows = read_csv(DIALECTS)
    cache = load_cache()
    queries: dict[str, str | None] = {}
    for row in rows:
        if row["Latitude"].strip() and row["Longitude"].strip():
            continue
        query = clean_query(row)
        if query:
            queries[query] = country_code(query)

    pending = [(query, code) for query, code in queries.items() if not cache.get(query)]
    print(f"{len(queries)} unique place queries; {len(pending)} uncached")
    for index, (query, code) in enumerate(pending, 1):
        try:
            cache[query] = fetch(query, code)
        except Exception as error:  # preserve progress before surfacing transient failures
            save_cache(cache)
            raise RuntimeError(f"geocoding failed for {query!r}: {error}") from error
        save_cache(cache)
        print(f"[{index}/{len(pending)}] {query}: {len(cache[query])} result(s)", flush=True)
        if index != len(pending):
            time.sleep(1.05)


def build_audit() -> tuple[int, int]:
    _, rows = read_csv(DIALECTS)
    cache = load_cache()
    audit_rows = []
    matched = fallback = 0
    for row in rows:
        if row["Latitude"].strip() and row["Longitude"].strip():
            continue
        query = clean_query(row)
        results = cache.get(query, []) if query else []
        result = results[0] if results else {}
        status = "geocoded" if result else "regional-fallback"
        matched += bool(result)
        fallback += not bool(result)
        audit_rows.append({
            "ID": row["ID"],
            "Query": query or "",
            "Status": status,
            "Latitude": str(result.get("lat", "")),
            "Longitude": str(result.get("lon", "")),
            "OSM_Type": str(result.get("osm_type", "")),
            "OSM_ID": str(result.get("osm_id", "")),
            "Match": str(result.get("display_name", "")),
        })
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit_rows)
    return matched, fallback


def apply() -> tuple[int, int]:
    fields, rows = read_csv(DIALECTS)
    decisions = {row["ID"]: row for row in read_csv(DECISIONS)[1]}
    applied = approximate = 0
    for row in rows:
        if row["Latitude"].strip() and row["Longitude"].strip():
            continue
        try:
            decision = decisions[row["ID"]]
        except KeyError as error:
            raise ValueError(f"No reviewed coordinate decision for {row['ID']}") from error
        row["Latitude"] = decision["Latitude"]
        row["Longitude"] = decision["Longitude"]
        applied += 1
        if decision["Method"] in {"manual-map", "manual-region", "manual-centroid", "survey-map", "source-region"} or "approximate" in decision["Note"].lower():
            row["Quality"] = "C"
            approximate += 1

    fd, temporary = tempfile.mkstemp(prefix=DIALECTS.name + ".", dir=DIALECTS.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, DIALECTS)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return applied, approximate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "audit", "apply"))
    command = parser.parse_args().command
    if command == "collect":
        collect()
    elif command == "audit":
        matched, fallback = build_audit()
        print(f"audit: {matched} geocoded, {fallback} regional fallbacks")
    else:
        applied, approximate = apply()
        print(f"applied: {applied} reviewed decisions ({approximate} explicitly approximate)")


if __name__ == "__main__":
    main()
