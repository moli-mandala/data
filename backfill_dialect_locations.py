#!/usr/bin/env python3
"""Backfill human-readable locations for legacy dialect registry rows.

The legacy rows already identify either a named locality/region or a source
lect.  This migration makes that information explicit for the frontend while
keeping uncertain coordinates blank.  Quality C marks broad, inherited, or
otherwise approximate location metadata.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).parent
DIALECTS = ROOT / "cldf/dialects.csv"


EXACT = {
    "arm": "Armenia (comparative-language label in the Romani source)",
    "boh": "Bohemia, Czechia (comparative-language label in the Romani source)",
    "bul": "Bulgaria (comparative-language label in the Romani source)",
    "eng": "England, United Kingdom (comparative-language label in the Romani source)",
    "germ": "Germany (comparative-language label in the Romani source)",
    "gr": "Greece (comparative-language label in the Romani source)",
    "hung": "Hungary (comparative-language label in the Romani source)",
    "it": "Italy (comparative-language label in the Romani source)",
    "norw": "Norway (comparative-language label in the Romani source)",
    "pol": "Poland (comparative-language label in the Romani source)",
    "rum": "Romania (comparative-language label in the Romani source)",
    "rus": "Russia (comparative-language label in the Romani source)",
    "sp": "Spain (comparative-language label in the Romani source)",
    "wel": "Wales, United Kingdom (comparative-language label in the Romani source)",
    "SEeur": "Balkans (regional comparative-language label in the Romani source)",
    "serb": "Serbia (comparative-language label in the Romani source)",
    "kar": "Karachi, Pakistan (source label; stored coordinate is inconsistent)",
    "pal": "Palestine (comparative-language label in the Domari source)",
    "pers": "Iran (Persian comparative-language label in the Domari source)",
    "lowerperak": "Lower Perak, Malaysia",
    "rod": "Sri Lanka (Rodiya regional variety)",
    "RP": "Rangpur, Bangladesh",
    "SH": "Shalkumar, Bangladesh",
    "TH": "Thakurgaon, Bangladesh",
    "MH": "Mahayespur, Nepal",
    "RL": "Rangeli, Morang district, Nepal",
    "Bj": "Bajui, Gorno-Badakhshan, Tajikistan",
    "Brt": "Bartang Valley, Gorno-Badakhshan, Tajikistan",
    "nig": "Nigali Sagar/Nigliva, Lumbini Province, Nepal (Ashokan inscription site)",
    "man": "Mansehra, Khyber Pakhtunkhwa, Pakistan (Ashokan inscription site)",
    "shah": "Shahbazgarhi, Khyber Pakhtunkhwa, Pakistan (Ashokan inscription site)",
    "dialect:Altit": "Altit, Hunza, Gilgit-Baltistan, Pakistan",
    "dialect:Ashkun%3A%20Sanu": "Sanu/Wama, Nuristan, Afghanistan",
    "dialect:Bhateri": "Batera area, Kohistan, Khyber Pakhtunkhwa, Pakistan",
    "dialect:Eastern%20Burushaski": "Hunza and Nagar, Gilgit-Baltistan, Pakistan",
    "dialect:Ganish": "Ganish, Hunza, Gilgit-Baltistan, Pakistan",
    "dialect:Hillside": "Hunza, Gilgit-Baltistan, Pakistan (source topographic label)",
    "dialect:Hopar": "Hopar, Nagar, Gilgit-Baltistan, Pakistan",
    "dialect:Hunza": "Hunza, Gilgit-Baltistan, Pakistan",
    "dialect:Kamviri": "Barg-i Matal area, Nuristan, Afghanistan",
    "dialect:Katavari%3A%20Ktivi": "Kantiwa/Ktivi, Nuristan, Afghanistan",
    "dialect:Khowar": "Chitral, Khyber Pakhtunkhwa, Pakistan",
    "dialect:NH": "Nagar and Hunza, Gilgit-Baltistan, Pakistan (source abbreviation NH)",
    "dialect:Nager": "Nagar, Gilgit-Baltistan, Pakistan",
    "dialect:Nuristani%20Kalasha%3A%20Amesdes": "Amesdes, Nuristan, Afghanistan",
    "dialect:Nuristani%20Kalasha%3A%20Nisheigram": "Nisheigram, Nuristan, Afghanistan",
    "dialect:Nuristani%20Kalasha%3A%20Vagal": "Waigal Valley, Nuristan, Afghanistan",
    "dialect:Palula": "Ashret and Biori valleys, Chitral, Khyber Pakhtunkhwa, Pakistan",
    "dialect:Pashai%3A%20Gorayk%20%28Degano%29": "Gorayk, Nuristan, Afghanistan",
    "dialect:Prasun": "Prasun Valley, Nuristan, Afghanistan",
    "dialect:Prasun%3A%20Sec": "Sech/Pronj, Prasun Valley, Nuristan, Afghanistan",
    "dialect:Prasun%3A%20Supu": "Supu/Ishtewi, Prasun Valley, Nuristan, Afghanistan",
    "dialect:Prasun%3A%20Ucu": "Ucu/Dewa, Prasun Valley, Nuristan, Afghanistan",
    "dialect:Prasun%3A%20Usut": "Usut/Pashki, Prasun Valley, Nuristan, Afghanistan",
    "dialect:Prasun%3A%20Zumu": "Zumu, Prasun Valley, Nuristan, Afghanistan",
    "dialect:Riverfront": "Hunza, Gilgit-Baltistan, Pakistan (source topographic label)",
    "dialect:Tregami%3A%20Gambir": "Gambir Valley, Nuristan, Afghanistan",
    "nured-Kt-w": "Western Katë-speaking area, Nuristan, Afghanistan",
    "nured-Kt-ne": "Northeastern Katë-speaking area, Nuristan, Afghanistan",
    "nured-Kt-se": "Southeastern Katë-speaking area, Nuristan, Afghanistan",
    "nured-Kt-kt": "Kantiwa/Ktivi, Nuristan, Afghanistan",
    "nured-Kt-kl": "Kulem, Nuristan, Afghanistan",
    "nured-Kt-rm": "Ramgel, Nuristan, Afghanistan",
    "nured-Kt-mm": "Mandagal Sufla, Nuristan, Afghanistan",
    "nured-Kt-br": "Barg-i Matal, Nuristan, Afghanistan",
    "nured-Kt-mr": "Bumburet Sheikhandeh, Chitral, Khyber Pakhtunkhwa, Pakistan",
    "nured-Kt-kun": "Rumbur Sheikhandeh, Chitral, Khyber Pakhtunkhwa, Pakistan",
    "nured-Wg-z": "Arans, Waigal Valley, Nuristan, Afghanistan",
    "nured-Wg-n": "Nisheigram, Nuristan, Afghanistan",
    "nured-Wg-wg": "Waigal Valley, Nuristan, Afghanistan",
    "nured-Wg-kg": "Kegal, Nuristan, Afghanistan",
    "nured-Gmb-gm": "Gambir Valley, Nuristan, Afghanistan",
    "nured-Gmb-dv": "Devoz, Nuristan, Afghanistan",
    "nured-Ash-s": "Wama, Nuristan, Afghanistan",
    "nured-Ash-tt": "Titin, Majegal, Nuristan, Afghanistan",
    "nured-Pr-p": "Pashki, Prasun Valley, Nuristan, Afghanistan",
    "nured-Pr-k": "Katar, Prasun Valley, Nuristan, Afghanistan",
    "nured-Pr-d": "Dewa, Prasun Valley, Nuristan, Afghanistan",
    "nured-Pr-pr": "Pronj, Prasun Valley, Nuristan, Afghanistan",
    "nured-Pr-i": "Ishtewi, Prasun Valley, Nuristan, Afghanistan",
    "nured-Pr-z": "Zumu, Prasun Valley, Nuristan, Afghanistan",
}


AFGHANISTAN = {"Pas", "Wg", "Kam", "Kt", "Pr", "Gmb", "Ash"}
PAKISTAN = {
    "Sh", "Kho", "Mai", "Bur", "L", "S", "srk", "jhang", "awan", "poth",
    "bhatr", "Psht", "Phal", "Kal",
}
MALDIVES = {"Md"}
INDIA = {
    "Pk", "MIA", "Ap", "As", "Gondi", "bang", "Kannada", "pampa", "K", "G", "Ku",
    "Kolami", "Konda", "Koraga", "KS", "Kurux", "Kuwi", "Aw", "Bi", "Malayalam",
    "Tamil", "Telugu", "NDu", "SSr", "P", "dog", "kaithal", "bagri_fatehabad",
    "dhundari_badagaon", "dhundari_bamore", "had", "hadothi_kelwada", "marwari_bagra",
    "Marw", "mewari_bannoda", "mewari_basad", "mewari_dholpura", "Rj", "mewati_akera",
    "mewati_jhambaus", "markodi", "bagheli_lakshman", "Brj", "bundeli_atarra", "M",
    "H", "Buksa", "Rana", "Kathoriya", "Sunha", "Dang",
    "Gadaba", "khash", "bhad",
}


def inferred_location(row: dict[str, str]) -> str:
    if row["ID"] in EXACT:
        return EXACT[row["ID"]]
    language = row["Language_ID"]
    name = row["Name"]
    if language in AFGHANISTAN:
        return f"{name}, Afghanistan"
    if language in PAKISTAN:
        return f"{name}, Pakistan"
    if language in MALDIVES:
        return f"{name}, Maldives"
    if language in INDIA:
        if language in {"Pk", "MIA", "Ap"}:
            return f"{name}, India (approximate historical reference point)"
        if language == "As":
            return f"{name}, India (historical inscription site or regional label)"
        return f"{name}, India"
    if language == "BH":
        return f"{name}, eastern India"
    raise ValueError(f"No location rule for {row['ID']} ({language}: {name})")


def backfill(path: Path = DIALECTS) -> tuple[int, int]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    locations = qualities = 0
    for row in rows:
        if not row["Location"].strip():
            row["Location"] = inferred_location(row)
            locations += 1
        if not row["Quality"].strip():
            row["Quality"] = "C"
            qualities += 1

    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return locations, qualities


if __name__ == "__main__":
    locations, qualities = backfill()
    print(f"filled {locations} locations and {qualities} quality ratings")
