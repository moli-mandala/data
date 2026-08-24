#!/usr/bin/env python3
"""Build the reviewed coordinate decision table for dialect rows lacking points.

The table deliberately records provenance. Survey GPS readings and reviewed
gazetteer records take precedence; broad or composite source labels are marked
as approximate display points rather than being presented as village fixes.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).parent
DIALECTS = ROOT / "cldf/dialects.csv"
AUDIT = ROOT / "data/dialect-coordinate-geocoding-audit.csv"
SUGGESTIONS = Path("/tmp/jambu-geonames-suggestions.csv")
DECISIONS = ROOT / "data/dialect-coordinate-decisions.csv"


def dms(degrees: float, minutes: float = 0, seconds: float = 0) -> float:
    return degrees + minutes / 60 + seconds / 3600


def point(lat, lon, method, source, note=""):
    return (str(round(float(lat), 7)), str(round(float(lon), 7)), method, source, note)


# Coordinates printed in the cited survey appendices.
SURVEY = {
    "maikoti_maikot": point(dms(28, 40, 35.7), dms(82, 52, 38.8), "survey-gps", "Maikoti Kham survey appendix"),
    "maikoti_arjal": point(dms(28, 40, 45.0), dms(82, 48, 57.9), "survey-gps", "Maikoti Kham survey appendix"),
    "maikoti_hukam": point(dms(28, 40, 27.2), dms(82, 51, 29.7), "survey-gps", "Maikoti Kham survey appendix"),
    "maikoti_ranma": point(dms(28, 42, 2.3), dms(82, 51, 28.3), "survey-gps", "Maikoti Kham survey appendix"),
    "southern_yamphu_devitar": point(27 + 27.737 / 60, 87 + 16.571 / 60, "survey-gps", "Yamphu survey appendix"),
    "southern_yamphu_rajarani": point(26 + 53.533 / 60, 87 + 24.815 / 60, "survey-gps", "Yamphu survey appendix"),
    "yamphu_hedangna": point(27 + 35.080 / 60, 87 + 18.997 / 60, "survey-gps", "Yamphu survey appendix"),
    "yamphu_khoktak": point(27 + 32.521 / 60, 87 + 17.636 / 60, "survey-gps", "Yamphu survey appendix"),
    "yamphu_seduwa": point(27 + 34.646 / 60, 87 + 15.717 / 60, "survey-gps", "Yamphu survey appendix"),
    "eastern_gorkha_tamang_kashigaun": point(dms(28, 11, 30.40), dms(84, 53, 54.14), "survey-gps", "Western Tamang survey appendix"),
    "western_tamang_jharlang": point(dms(28, 7, 25.6), dms(85, 2, 12.1), "survey-gps", "Western Tamang survey appendix"),
    "gurung_yangjakot": point(dms(28, 15, .13), dms(84, 6, 7.26), "survey-gps", "Gurung survey appendix"),
    "gurung_birethanti": point(dms(28, 18, 34.81), dms(84, 46, 30.81), "survey-gps", "Gurung survey appendix", "GPS is for Mohoria/Dangsing survey site near Birethanti"),
    "gurung_bhurdumpola": point(dms(28, 8, 15.49), dms(83, 49, 8.28), "survey-gps", "Gurung survey appendix"),
    "gurung_ajirkot": point(dms(28, 9, 5.60), dms(84, 38, 24.24), "survey-gps", "Gurung survey appendix"),
    "gurung_maling": point(dms(28, 14, 9.16), dms(84, 17, 20.70), "survey-gps", "Gurung survey appendix"),
    "gurung_pyarjung": point(28.15512, 84.53017, "gazetteer", "GeoNames 7966927", "wordlist was collected in Kathmandu; point is the named home village"),
    "eastern_magar_sarlahi": point(dms(27, 3, 11.1), dms(85, 37, 58.9), "survey-gps", "Eastern Magar survey appendix"),
    "eastern_magar_nawalparasi": point(dms(27, 47, 54.2), dms(84, 9, 10.7), "survey-gps", "Eastern Magar survey appendix"),
    "eastern_magar_dhankuta": point(dms(26, 53, .88), dms(87, 29, 50.71), "survey-gps", "Eastern Magar survey appendix"),
    "eastern_magar_panchthar": point(dms(27, 10, 48.9), dms(87, 55, 48.4), "survey-gps", "Eastern Magar survey appendix"),
    "majhi_kunauri": point(27 + 22.457 / 60, 86 + 2.066 / 60, "survey-gps", "Majhi/Bote survey appendix", "Appendix prints E06; interpreted as E086 from the site and companion entries"),
    "majhi_majhigau": point(27 + 15.841 / 60, 86 + 39.897 / 60, "survey-gps", "Majhi/Bote survey appendix"),
    "bote_kawasoti": point(27 + 34.207 / 60, 84 + 6.208 / 60, "survey-gps", "Majhi/Bote survey appendix"),
    "majhi_gaikura": point(27 + 24.214 / 60, 86 + 3.966 / 60, "survey-gps", "Majhi/Bote survey appendix"),
    "majhi_pachuwar": point(27 + 32.791 / 60, 85 + 45.853 / 60, "survey-gps", "Majhi/Bote survey appendix"),
    "danuwar_chandanpur": point(27 + 2.526 / 60, 86 + 2.787 / 60, "survey-gps", "Dewas Rai/Danuwar survey appendix"),
    "done_danuwar_jaretar": point(27 + 36.593 / 60, 85 + 39.646 / 60, "survey-gps", "Dewas Rai/Danuwar survey appendix"),
    "dewas_rai_majhgaun": point(27 + 15.039 / 60, 85 + 28.614 / 60, "survey-gps", "Dewas Rai/Danuwar survey appendix"),
    "dewas_rai_mahendra_jhyadi": point(27 + 16.533 / 60, 85 + 33.533 / 60, "survey-gps", "Dewas Rai/Danuwar survey appendix"),
    "dewas_rai_singoul": point(27 + 12.441 / 60, 85 + 15.084 / 60, "survey-gps", "Dewas Rai/Danuwar survey appendix"),
    "kochariya_singoul": point(27 + 12.156 / 60, 85 + 14.135 / 60, "survey-gps", "Dewas Rai/Danuwar survey appendix"),
    "humla_til": point(30 + 13.292 / 60, 81 + 27.154 / 60, "survey-gps", "Humla Tibetan survey appendix"),
    "humla_muchu": point(30 + 3.989 / 60, 81 + 32.498 / 60, "survey-gps", "Humla Tibetan survey appendix"),
    "humla_bargaun": point(29 + 57.54 / 60, 81 + 51.16 / 60, "survey-gps", "Humla Tibetan survey appendix"),
    "humla_dojam": point(29 + 57.22 / 60, 81 + 55.47 / 60, "survey-gps", "Humla Tibetan survey appendix"),
    "humla_yakpa": point(30 + 2.20 / 60, 81 + 49.29 / 60, "survey-gps", "Humla Tibetan survey appendix"),
    "humla_kermi": point(30 + 3.07 / 60, 81 + 41.57 / 60, "survey-gps", "Humla Tibetan survey appendix"),
}

for ident, lat, lon in [
    ("western_magar_mathagadhi", 27.7810048, 83.6820386), ("central_magar_mityal", 27.8705296, 83.8656951),
    ("western_magar_jhokedi", 27.856735, 83.702993), ("central_magar_siluwa", 27.833416, 83.836814),
    ("western_magar_lasargha", 27.965935, 83.5023057), ("central_magar_dhardh", 28.061498, 83.89293),
    ("central_magar_inaskot", 27.943441, 83.833593), ("central_magar_rhising", 27.9227391, 84.1765527),
    ("central_magar_michhurlung", 27.957774, 84.09343), ("central_magar_arkhala", 27.798514, 84.152704),
    ("central_magar_raikot", 27.793812, 84.124636), ("central_magar_bhadari", 27.729583, 84.012809),
]:
    SURVEY[ident] = point(lat, lon, "survey-gps", "2024 Magar survey site metadata")


# Individually reviewed locality decisions not already represented by survey GPS.
MANUAL = {
    "tsum_chekampar": point(28.47291, 85.0839, "gazetteer-area", "GeoNames 10318012", "administrative area point"),
    "southern_ghale_kyaura": point(28.102, 84.705, "manual-map", "Saurpani locality context", "approximate village display point"),
    "northern_ghale_uiya": point(28.28459, 84.88885, "gazetteer", "GeoNames 7997392"),
    "kutang_ghale_chyak": point(28.53, 84.84, "manual-map", "Kutang survey map", "approximate village display point"),
    "kutang_ghale_rana": point(28.55, 84.79, "manual-map", "Kutang survey map", "corrects unrelated southern Nepal match; approximate"),
    "western_tamang_lamagara": point(28.02, 85.04, "manual-map", "Midkot survey locality context", "approximate; rejects same-name western Nepal result"),
    "kurux_lochani": point(26.64008, 87.31626, "gazetteer", "GeoNames 7970190"),
    "kurux_siddhapur": point(27.16, 84.88, "manual-map", "Parsa district locality context", "approximate; rejects same-name eastern Nepal result"),
    "kurux_tokla": point(26.72229, 88.15386, "gazetteer", "GeoNames 7953940"),
    "loke_lo_manthang": point(29.18391, 83.96267, "gazetteer", "GeoNames 1283080"),
    "loke_ghiling": point(28.9438, 83.8184, "manual-map", "Upper Mustang settlement map"),
    "loke_chhosher": point(29.145, 83.958, "manual-map", "Upper Mustang settlement map", "approximate village center"),
    "loke_jharkot": point(28.81937, 83.84811, "gazetteer", "GeoNames 6941673"),
    "loke_kagbeni": point(28.83717, 83.78363, "gazetteer", "GeoNames 1283278"),
    "kochila_morang_east": point(26.56281, 87.33297, "gazetteer", "GeoNames 7971221"),
    "kochila_siraha_central": point(26.70755, 86.4431, "gazetteer", "GeoNames 7996357"),
    "bote_madi": point(27 + 30.223 / 60, 84.18, "survey-gps", "Majhi/Bote survey appendix", "longitude is truncated in extracted appendix; set from Madi/Pandapnagar map context"),
    "kudiya_g1": point(12.42149, 75.7387543, "gazetteer", "OpenStreetMap node for Madikeri", "source code G1 only identifies the taluk"),
    "kudiya_k1": point(12.315, 75.09, "manual-map", "Hosdurg/Kanhangad taluk context", "source code K1 only identifies the taluk"),
    "dotyali_baitadi": point(29.53715, 80.70543, "gazetteer", "GeoNames 7981926"),
    "humla_yalbang": point(30.05763, 81.62784, "gazetteer", "GeoNames 7951131"),
    "western_tamang_sahugaun": point(27.97637, 85.15427, "gazetteer", "GeoNames 7974918"),
    "lohorung_angala": point(27.44285, 87.16231, "gazetteer", "GeoNames 7990551"),
    "lohorung_dhupu": point(27.4257, 87.27035, "gazetteer", "GeoNames 7990382", "village point rather than administrative centroid"),
    "yamphu_num": point(27.53749, 87.30097, "gazetteer-area", "GeoNames 7802325"),
    "pahari_salintar": point(27.59928, 85.31391, "gazetteer", "GeoNames 7965469"),
    "pahari_maasdada": point(27.55, 85.31, "manual-map", "Godawari survey map", "approximate Maasdada/Pachma display point"),
    "pahari_jamune": point(27.67371, 85.78769, "gazetteer", "GeoNames 7995517"),
    "naaba_pibu": point(27.78, 87.36, "manual-map", "Naaba survey map; Pibu/Ridak in Bhotkhola", "approximate village display point"),
    "puroik_phereng": point(27.64, 93.58, "manual-map", "Puroik survey map", "approximate"),
    "puroik_gari": point(27.55, 93.58, "manual-map", "Mengio survey locality context", "approximate"),
    "puroik_chug": point(27.418, 92.238, "manual-map", "Parchu/Chug survey locality context", "approximate"),
    "bugun_wangho": point(27.22, 92.49, "manual-map", "Bugun survey map", "approximate"),
    "bugun_namphri": point(27.23, 92.53, "manual-map", "Bugun survey map", "approximate"),
    "tagin_baki": point(28.20, 94.02, "manual-map", "Siyum survey locality context", "approximate"),
    "tagin_maskia": point(28.13, 94.08, "manual-map", "Upper Subansiri survey map", "approximate"),
    "tagin_takseng": point(28.37481, 93.24366, "gazetteer", "GeoNames 12523474"),
    "ghatage-kasargod1970": point(12.52616, 75.12132, "gazetteer-area", "GeoNames 12684168", "district centroid"),
    "ThuiYasin": point(36.49003, 73.3461, "gazetteer", "GeoNames 11041152"),
    "Yasin": point(36.64332, 73.44532, "gazetteer", "GeoNames 1180499", "Darkot village, not the river"),
    "Ishkoman": point(36.50401, 73.90274, "gazetteer", "GeoNames 1176695"),
    # The two Bangladesh reports give district/subdistrict and travel-route
    # metadata, but not GPS. These are therefore honest administrative-area
    # display points, not fabricated village centroids.
    "hajong_nugapara": point(25.52, 90.22, "survey-map", "Hajong report figure 3", "approximate West Garo Hills map point"),
    "hajong_chilapara": point(25.62, 90.18, "survey-map", "Hajong report figure 3", "approximate West Garo Hills map point"),
    "hajong_nirghini": point(25.70, 90.25, "survey-map", "Hajong report figure 3", "approximate West Garo Hills map point"),
    "hajong_dalugau": point(25.55, 90.40, "survey-map", "Hajong report figure 3", "approximate West Garo Hills map point"),
    "hajong_balachanda": point(25.45, 90.30, "survey-map", "Hajong report figure 3", "approximate West Garo Hills map point"),
    "hajong_dhamor": point(26.12791, 90.60974, "gazetteer-area", "GeoNames 1271152", "Goalpara district display point"),
    "hajong_gopalbari": point(25.07889, 90.9, "gazetteer-area", "GeoNames 11286987", "Kalmakanda subdistrict display point from report metadata"),
    "hajong_gopalpur": point(25.09, 90.70, "gazetteer-area", "GeoNames 7646874", "Durgapur subdistrict display point; replaces unrelated OSM Gopalpur"),
    "hajong_bhalukapara": point(25.11, 90.52, "gazetteer-area", "GeoNames 7646387", "Dhobaura subdistrict display point"),
    "hajong_nokshi": point(25.18559, 90.06678, "gazetteer-area", "GeoNames 7912475", "Jhinaigati subdistrict display point"),
    "santali_rajarampur": point(25.63, 88.53, "survey-map", "Santali Cluster report figure 3 and Birol/Azimpur metadata", "approximate"),
    "santali_rautnagar": point(25.8866, 88.25867, "gazetteer-area", "GeoNames 9258754", "Ranisankail subdistrict display point"),
    "santali_paharpur": point(25.41997, 89.09019, "gazetteer-area", "GeoNames 11282264", "Nawabganj, Dinajpur subdistrict display point; replaces unrelated OSM road"),
    "santali_patichora": point(25.0464, 88.713, "gazetteer-area", "GeoNames 11286168", "Patnitala area display point from report route metadata"),
    "santali_jabri": point(24.84154, 88.55464, "gazetteer-area", "GeoNames 11282751", "Niamatpur subdistrict display point"),
    "santali_bodobelghoria": point(24.41112, 88.98673, "gazetteer-area", "GeoNames 7483813", "Natore area display point"),
    "santali_rashidpur": point(24.34251, 91.52037, "gazetteer-area", "GeoNames 11283078", "Bahubal subdistrict display point"),
    "mundari_nijpara": point(26.0, 88.58333, "gazetteer-area", "GeoNames 9278416", "Birganj subdistrict display point"),
    "mundari_begunbari": point(25.14, 88.85, "gazetteer-area", "GeoNames 7645713", "Dhamoirhat subdistrict display point; replaces road-feature result"),
    "mundari_karimpur": point(24.54396, 91.87018, "gazetteer-area", "GeoNames 11285113", "Rajnagar subdistrict display point"),
    "mahali_abirpara": point(25.24495, 89.24711, "gazetteer-area", "GeoNames 11284706", "Ghoraghat subdistrict display point"),
    "mahali_matindor": point(25.0464, 88.713, "gazetteer-area", "GeoNames 11286168", "Patnitala subdistrict display point"),
    "mahali_pachondor": point(24.62, 88.53, "gazetteer-area", "GeoNames 7645716", "Tanore subdistrict display point"),
    "koda_kundang": point(24.62, 88.53, "gazetteer-area", "GeoNames 7645716", "Tanore subdistrict display point"),
    "koda_krishnupur": point(24.36873, 88.82882, "gazetteer-area", "GeoNames 11286570", "Puthia subdistrict display point"),
    "kol_babudaing": point(24.47928, 88.34737, "gazetteer-area", "GeoNames 11284651", "Godagari subdistrict display point"),
    "rabha_rongdani": point(25.60, 90.62, "survey-map", "Rabha survey map", "approximate Naguapara display point"),
    "rabha_maituri": point(25.56, 90.70, "survey-map", "Rabha survey map", "approximate Boro Paham display point"),
}


KHOWAR_PLACES = {
    "chitral town": (35.850889, 71.79019), "proper chitral": (35.850889, 71.79019), "chitral museum": (35.850889, 71.79019),
    "drosh": (35.56163, 71.79756), "booni": (36.25392, 72.22284), "mastuj": (36.28356, 72.51942),
    "chapali": (36.3357, 72.60138), "balim": (36.07012, 72.43959), "bang": (36.52283, 72.76388),
    "laspur": (36.04784, 72.46796), "sor laspur": (36.04784, 72.46796), "pasum": (36.30426, 72.55493),
    "parwak": (36.27759, 72.3901), "parkusap": (36.2888, 72.5292), "rayin": (36.39233, 72.37763),
    "shagram": (36.34458, 72.13311), "shogram": (36.34458, 72.13311), "sor rech": (36.6326, 72.5621),
    "uzhnu": (36.54541, 72.46876), "warijun": (36.30044, 72.21299), "zondrangram": (36.39444, 72.22829),
    "terich": (36.39444, 72.22829), "torkhow": (36.45309, 72.42228), "mulkhow": (36.30044, 72.21299),
    "yarkhun": (36.52283, 72.76388), "sonoghor": (36.30, 72.18), "lutkoh": (36.01231, 71.65609), "lotkoh": (36.01231, 71.65609),
    "chumurkun": (35.79784, 71.78801), "jughoor": (35.82708, 71.78633), "thingshen": (35.85158, 71.77971),
    "mogh": (36.01231, 71.65609), "madaglasht": (35.77558, 72.03137), "zargarandeh": (35.8506, 71.7925),
    "uthul": (36.30442, 72.18117), "mahrting": (36.49059, 72.72128), "singoor": (35.89778, 71.79791),
    "mroi": (35.93, 71.82), "reshun": (36.15365, 72.09928), "meragram": (36.26364, 72.37142),
    "khot": (36.50216, 72.53267), "khairabad": (36.78961, 73.0418), "karimabad": (35.99193, 71.81522),
    "shyaqotek": (35.85, 71.79), "lower chitral": (35.75, 71.78), "upper chitral": (36.33, 72.29),
}


NURISTANI = {
    "dialect:Ashkun%3A%20Sanu": (35.1202001, 70.7324182), "dialect:Bhateri": (34.9582, 72.92674),
    "dialect:Kamviri": (35.40986, 71.33679), "dialect:Katavari%3A%20Ktivi": (35.3277082, 70.7227742),
    "dialect:Nuristani%20Kalasha%3A%20Amesdes": (35.04, 70.98), "dialect:Nuristani%20Kalasha%3A%20Nisheigram": (35.0833322, 70.8245782),
    "dialect:Nuristani%20Kalasha%3A%20Vagal": (35.04, 70.98), "dialect:Pashai%3A%20Gorayk%20%28Degano%29": (34.6458, 70.9008),
    "dialect:Tregami%3A%20Gambir": (34.75, 70.96), "nured-Kt-kt": (35.32264, 70.73716),
    "nured-Kt-mm": (35.44991, 71.31615), "nured-Kt-mr": (35.698358, 71.686882), "nured-Kt-kun": (35.775859, 71.692544),
    "nured-Kt-kl": (35.38, 70.78), "nured-Kt-rm": (35.45, 70.80), "nured-Kt-br": (35.67283, 71.34339),
    "nured-Wg-z": (35.12675, 70.95981), "nured-Wg-n": (35.0833322, 70.8245782), "nured-Wg-wg": (35.04, 70.98), "nured-Wg-kg": (35.12, 70.95),
    "nured-Gmb-gm": (34.75, 70.96), "nured-Gmb-dv": (34.77, 70.94), "nured-Ash-s": (35.18228, 70.795553), "nured-Ash-tt": (35.10, 70.78),
    "nured-Pr-p": (35.33007, 70.89342), "nured-Pr-d": (35.39845, 70.9308), "nured-Pr-i": (35.45631, 70.938),
}


REGIONAL = {
    "Rabha": (25.72, 90.55), "Puroik": (27.55, 93.55), "Bugun": (27.21, 92.55), "Tagin": (28.10, 94.00),
    "Naaba": (27.78, 87.37), "Hajong": (25.20, 90.35), "sa": (24.75, 88.85), "mu": (24.55, 88.65),
    "Mahali": (24.65, 88.70), "Koda": (24.70, 88.82), "KolBangladesh": (24.53, 88.337),
    "Kt": (35.41, 70.98), "Wg": (35.04, 70.98), "Gmb": (34.75, 70.96), "Pr": (35.40, 70.92),
}


def main() -> None:
    rows = list(csv.DictReader(DIALECTS.open(encoding="utf-8")))
    audit = {r["ID"]: r for r in csv.DictReader(AUDIT.open(encoding="utf-8"))}
    decisions = []
    for row in rows:
        if row["Latitude"].strip() and row["Longitude"].strip():
            continue
        ident, lang, loc = row["ID"], row["Language_ID"], row["Location"]
        choice = SURVEY.get(ident) or MANUAL.get(ident)

        if choice is None and lang == "Kho":
            low = loc.lower()
            hits = [v for k, v in KHOWAR_PLACES.items() if k in low]
            if not hits:
                hits = [(35.850889, 71.79019)]
                note = "opaque or non-local source label; Chitral display point"
            else:
                note = "centroid" if len(hits) > 1 else "reviewed source-home locality"
            lat = sum(v[0] for v in hits) / len(hits); lon = sum(v[1] for v in hits) / len(hits)
            choice = point(lat, lon, "manual-source-home", "Bashir source-home registry and reviewed Chitral map", note)

        if choice is None and ident in NURISTANI:
            lat, lon = NURISTANI[ident]
            choice = point(lat, lon, "manual-cross-reference", "existing Jambu locality point / reviewed gazetteer")

        if choice is None and ident.startswith("dialect:Prasun"):
            choice = point(35.40, 70.92, "manual-map", "Prasun Valley locality map", "approximate display point")
        if choice is None and ident.startswith("nured-Pr-"):
            choice = point(35.40, 70.92, "manual-map", "Prasun Valley locality map", "approximate named-village display point")
        if choice is None and ident.startswith("nured-"):
            lat, lon = REGIONAL.get(lang, (35.2, 70.95))
            choice = point(lat, lon, "manual-region", "NURISTAN dialect region", "approximate dialect-area display point")

        if choice is None and ident in {"dialect:Eastern%20Burushaski", "dialect:NH", "domaaki_nager"}:
            choice = point(36.30425, 74.26652, "manual-region", "Nagar/Hunza community map", "regional or composite display point")
        if choice is None and ident == "domaaki_hunza":
            choice = point(36.31156, 74.64450, "manual-region", "Hunza/Mominabad community map", "community-area display point")
        if choice is None and ident == "dialect:Palula":
            choice = point((35.43206 + 35.4646) / 2, (71.74515 + 71.810718) / 2, "manual-centroid", "Ashret and Biori gazetteer points", "two-valley centroid")

        if choice is None and ident == "lsi_gypsyeuropean":
            choice = point(46.82, 14.85, "source-region", "LSI broad European variety", "broad regional display point")
        if choice is None and ident == "lsi_easternbengali":
            choice = point(24.0, 90.0, "source-region", "LSI broad Eastern Bengali variety", "broad regional display point")

        # Retain an exact, reviewed first-pass OpenStreetMap hit unless a more
        # authoritative decision above replaced it.
        if choice is None and audit[ident]["Status"] == "geocoded":
            a = audit[ident]
            choice = point(a["Latitude"], a["Longitude"], "openstreetmap", f"OSM {a['OSM_Type']} {a['OSM_ID']}", a["Match"])

        if choice is None and lang in REGIONAL:
            lat, lon = REGIONAL[lang]
            choice = point(lat, lon, "manual-region", "source survey map and named administrative context", "approximate named-site display point")
        if choice is None:
            # Remaining named sites are sparse survey-map labels. This branch
            # is deliberately explicit in the audit and is never called a GPS fix.
            fallback = {
                "Nubri": (28.62, 84.73), "Tsum": (28.52, 85.08), "SouthernGhale": (28.20, 84.75),
                "NorthernGhale": (28.37, 84.82), "KutangGhale": (28.55, 84.75), "EasternGorkhaTamang": (28.18, 84.90),
                "WesternTamang": (28.02, 85.06), "Kurux": (26.7, 87.2), "Loy": (28.98, 83.90), "Ths": (28.75, 83.69),
                "KochilaTharu": (26.65, 86.75), "Bote": (27.55, 84.30), "Kudiya": (12.3, 75.5), "Dotyali": (29.27, 80.94),
                "Gurung": (28.33, 84.33), "Humla": (30.15, 81.57), "Lohorung": (27.42, 87.22), "PahariNewar": (27.566, 85.318),
            }.get(lang)
            if fallback is None:
                raise ValueError(f"no reviewed decision for {ident}: {loc}")
            choice = point(*fallback, "manual-region", "source survey map and administrative context", "approximate display point")

        lat, lon, method, source, note = choice
        decisions.append({"ID": ident, "Latitude": lat, "Longitude": lon, "Method": method, "Source": source, "Note": note})

    with DECISIONS.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decisions[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(decisions)
    print(f"wrote {len(decisions)} reviewed decisions to {DECISIONS}")


if __name__ == "__main__":
    main()
