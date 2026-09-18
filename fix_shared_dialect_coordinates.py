#!/usr/bin/env python3
"""Replace shared regional display points on named-locality dialects with reviewed site points.

Several ingestions registered every named survey site beneath one base language at that
language's (or one district's) single coordinate. Each row below was reviewed by hand against
the site table printed in the source (SIL report site tables, Morgenstierne/Strand locality
lists, the SDML site list) and resolved through OpenStreetMap/Nominatim, the GeoNames country
dumps, or an explicit map reading. Quality ``B`` = a gazetteer point for the named village;
``C`` = the nearest administrative unit the source names, or an approximate map reading.

    uv run python fix_shared_dialect_coordinates.py          # apply to cldf/dialects.csv + decisions table
    uv run python fix_shared_dialect_coordinates.py --check  # print the changes without writing
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
DIALECTS = ROOT / "cldf/dialects.csv"
DECISIONS_TABLE = ROOT / "data/dialect-coordinate-decisions.csv"


def p(lat, lon, quality, method, source, note=""):
    return (str(lat), str(lon), quality, method, source, note)


def osm(lat, lon, osm_ref, note):
    return p(lat, lon, "B", "openstreetmap", f"OSM {osm_ref}", note)


def gn(lat, lon, geonames_id, note):
    return p(lat, lon, "B", "gazetteer", f"GeoNames {geonames_id}", note)


def gn_approx(lat, lon, geonames_id, note):
    return p(lat, lon, "C", "gazetteer", f"GeoNames {geonames_id}", note)


def area(lat, lon, source, unit):
    return p(lat, lon, "C", "gazetteer-area", source, f"village not in gazetteers; {unit} display point (approximate)")


def region(lat, lon, source, note):
    return p(lat, lon, "C", "manual-region", source, note)


# Administrative display points reused by several sites (Nominatim results for the unit).
LAMA = area(21.7553161, 92.21, "OSM Lama Upazila", "Lama upazila, Bandarban")
THANCHI = area(21.8226132, 92.43, "OSM Thanchi Upazila", "Thanchi upazila, Bandarban")
RUMA = area(22.0320769, 92.4419057, "OSM Ruma Upazila", "Ruma upazila, Bandarban")
ROWANGCHHARI = area(22.1729615, 92.39, "OSM Rowangchhari Upazila", "Rowangchhari upazila, Bandarban")
MATIRANGA = area(23.0444506, 91.86, "OSM Matiranga", "Matiranga, Khagrachhari")
KHAGRACHHARI = area(23.1077999, 91.98, "OSM Khagrachhari", "Khagrachhari town")
NALITABARI = area(25.1231518, 90.17, "OSM Nalitabari Upazila", "Nalitabari upazila, Sherpur")
HALUAGHAT = area(25.1222567, 90.33, "OSM Haluaghat Upazila", "Haluaghat upazila, Mymensingh")
DHOBAURA = area(25.0954038, 90.52, "OSM Dhobaura Upazila", "Dhobaura upazila, Mymensingh")
KALMAKANDA = area(25.0719466, 90.85, "OSM Kalmakanda", "Kalmakanda, Netrokona")
DURGAPUR = area(25.1238305, 90.6781, "OSM Durgapur", "Durgapur town, Netrokona")
WEST_GARO_HILLS = area(25.6065343, 90.20, "OSM West Garo Hills", "West Garo Hills district, Meghalaya")
AGALI = area(11.1057095, 76.65, "OSM Agali", "Agali (Attappady block HQ), Palakkad")
NEMMARA = area(10.5939500, 76.60, "OSM Nemmara", "Nemmara, Palakkad")
NALLEPILLY = area(10.7300881, 76.78, "OSM Nallepilly", "Nallepilly, Chittur taluk, Palakkad")
METTUPALAYAM = area(11.3059363, 76.93, "OSM Mettupalayam", "Mettupalayam, Coimbatore")
KOTAGIRI = area(11.4230431, 76.86, "OSM Kotagiri", "Kotagiri, Nilgiris")
MASINAGUDI = area(11.5682904, 76.63, "OSM Masinagudi", "Masinagudi, Nilgiris")
ADIMALI = area(10.0144274, 76.95, "OSM Adimali", "Adimali block, Idukki")
DEVIKULAM = area(10.0656229, 77.10, "OSM Devikulam", "Devikulam block, Idukki")
KATTAPPANA = area(9.7561717, 77.11, "OSM Kattappana", "Kattappana block, Idukki")
NEDUMKANDAM = area(9.8404224, 77.15, "OSM Nedumkandam", "Nedumkandam block, Idukki")
PEERMADE = area(9.5720149, 76.99, "OSM Peermade", "Peermade block, Idukki")
KOTHAMANGALAM = area(10.1332804, 76.73, "OSM Kothamangalam", "Kothamangalam, Ernakulam")
PACHIPENTA = area(18.4792170, 83.11, "OSM Pachipenta", "Pachipenta mandal HQ, Vizianagaram")
BHADERWAH = area(32.9783699, 75.7185638, "OSM Bhaderwah", "Bhaderwah, Doda")
NIJRAB = area(35.0462800, 69.6459760, "OSM Nijrab District", "Nijrab district, Kapisa")
TAGAB = area(34.7945590, 69.6793040, "OSM Tagab District", "Tagab district, Kapisa")
KALKOT = area(35.4175245, 72.18, "OSM Kalkot", "Kalkot, Dir Kohistan")
ATARRA = area(25.2589804, 80.61, "OSM Atarra", "Atarra, Banda")

# Existing Jambu points reused for the same village registered under another source.
PASHKI = p(35.33007, 70.89342, "B", "manual-cross-reference", "existing Jambu point nured-Pr-p", "Pashki village")
DEWA = p(35.39845, 70.9308, "B", "manual-cross-reference", "existing Jambu point nured-Pr-d", "Dewa village")
ISHTIWI = p(35.45631, 70.938, "B", "manual-cross-reference", "existing Jambu point nured-Pr-i", "Ishtiwi village")
PRONZ = p(35.43, 70.935, "C", "manual-map", "Parun valley village order (Strand)", "Pronz lies between Dewa and Ishtiwi; approximate")
ZUMU = p(35.49, 70.94, "C", "manual-map", "Parun valley village order (Strand)", "Zumu is the uppermost Parun village, above Ishtiwi; approximate")
KATAR = p(35.365, 70.905, "C", "manual-map", "Parun valley village order (Strand)", "Kushtaki/Katar lies between Pashki and Dewa; approximate")
AMESHDESH = osm(35.1635119, 70.9237647, "node Ameshdesh", "Ameshdesh, Waygal district, Nuristan")
WAYGAL = gn(35.19069, 70.99478, "1121652", "Waygal village, Nuristan")
GAJNI = osm(25.2423898, 90.0455190, "node Gajni", "Gajni, Jhenaigati, Sherpur")
KUNJAPANAI = gn(11.35982, 76.93374, "11461332", "Kunjapanai, Kotagiri taluk")

DECISIONS: dict[str, tuple[str, str, str, str, str, str]] = {
    # --- Kim et al. 2011, Kok Borok of Bangladesh (silesr2011_038), table 3 -------------------
    "tripura2011-a-boro-pharangsia": LAMA,
    "tripura2011-b-choto-madhuk": THANCHI,
    "tripura2011-c-dolchari": MATIRANGA,
    "tripura2011-d-lombapara": osm(23.2469323, 91.9254482, "node Lombapara", "Lombapara, Panchhari/Khagrachhari"),
    "tripura2011-e-pakkhipara": osm(23.0137422, 91.9682532, "node Pakkhipara", "Pakkhipara, Khagrachhari Sadar"),
    "tripura2011-f-tongpaipara": osm(23.2208398, 91.8220067, "node Tongpaipara", "Tongpaipara, Matiranga"),
    "tripura2011-g-mildhanpara": osm(23.3358658, 91.8485713, "node Mildhanpara", "Mildhanpara, Panchhari"),
    "tripura2011-h-noimail-gutchagram": osm(23.1996378, 92.0354211, "node Noimail Gutchagram", "Noimail Gutchagram, Dighinala"),
    "tripura2011-i-beltolipara": KHAGRACHHARI,
    "tripura2011-j-robertpara": osm(21.8332494, 92.4374533, "node Robertpara", "Robertpara, Thanchi"),
    "tripura2011-k-bethanipara": RUMA,
    "tripura2011-l-katchaptali": ROWANGCHHARI,
    "tripura2011-m-laiphu-karbaripara": osm(23.2786340, 91.8133481, "node Laiphu Karbaripara", "Laiphu Karbaripara, Matiranga"),
    "tripura2011-n-krishna-dayalpara": osm(23.2826382, 91.7623680, "node Krishna Dayalpara", "Krishna Dayalpara, Matiranga"),
    "tripura2011-o-jarichandrapara": osm(23.0135981, 91.8818934, "node Jarichandrapara", "Jarichandrapara, Matiranga"),
    "tripura2011-p-doluchara": osm(24.3037670, 91.7803850, "node Doluchara", "Doluchara, Kamalganj/Sreemangal, Moulvibazar"),
    "tripura2011-q-satchari-tripura-basti": osm(24.1138280, 91.4240949, "node Satchari Tripura Basti", "Satchari, Chunarughat, Habiganj"),
    "tripura2011-r-khumulung": osm(23.8001795, 91.4409328, "node Khumulwng", "Khumulwng, West Tripura, India"),
    "tripura2011-s-barbakpur": osm(23.7519411, 89.6427186, "node Barbakpur", "Barbakpur, Rajbari Sadar"),
    "tripura2011-t-oldlankar": osm(23.2570916, 92.3413662, "node Old Lankar", "Old Lankar, Baghaichhari, Rangamati"),
    "tripura2011-u-gajni": GAJNI,
    "tripura2011-v-nagar-sontosh": DHOBAURA,
    "tripura2011-w-nalchapra": KALMAKANDA,
    # --- Kim, Kim & Sangma 2012, Garo of Bangladesh (silesr2012_007), appendix P ----------------
    "garobd2012-a-gajni": GAJNI,
    "garobd2012-b-nokshi": p(25.19, 90.07, "C", "manual-map", "Koch report figure 4; Nokshi lies on the border in Jhenaigati", "approximate"),
    "garobd2012-c-kholchanda": NALITABARI,
    "garobd2012-d-namchapara": HALUAGHAT,
    "garobd2012-e-songra": HALUAGHAT,
    "garobd2012-f-nagar-sontosh": DHOBAURA,
    "garobd2012-g-sapmari": DHOBAURA,
    "garobd2012-h-digholbag": DHOBAURA,
    "garobd2012-i-chunia": gn(24.63547, 90.12171, "1205677", "Chunia, Madhupur, Tangail"),
    "garobd2012-j-panchgaon": KALMAKANDA,
    "garobd2012-k-sonyasipara": KALMAKANDA,
    "garobd2012-l-bharatpur": DURGAPUR,
    "garobd2012-m-nalchapra": KALMAKANDA,
    "garobd2012-n-utrail": DURGAPUR,
    "garobd2012-o-birisiri": gn(25.08816, 90.67778, "11286988", "Birisiri union, Durgapur, Netrokona"),
    # --- Kim et al. 2011, Koch dialects of Meghalaya and Assam (silesr2011_033) -----------------
    "koch2011-garo-tura": osm(25.5125616, 90.2171526, "node Tura", "Tura, West Garo Hills"),
    "koch2011-harigaya-koch-ampati": osm(25.4704958, 89.9367062, "node Ampati", "Ampati, South West Garo Hills"),
    "koch2011-harigaya-koch-harigaon": WEST_GARO_HILLS,
    "koch2011-margan-koch-marganpara": WEST_GARO_HILLS,
    "koch2011-tintekiya-koch-haldibari": WEST_GARO_HILLS,
    "koch2011-tintekiya-koch-noksi": p(25.20, 90.08, "C", "manual-map", "Nokshi straddles the Meghalaya–Sherpur border", "Indian side of Nokshi; approximate"),
    "koch2011-wanang-koch-khalpara": WEST_GARO_HILLS,
    "koch2011-wanang-koch-khilbui": WEST_GARO_HILLS,
    "koch2011-koch-rabha-debitola": area(26.1322750, 90.12, "OSM Dhubri", "Dhubri district, Assam"),
    "koch2011-tintekiya-koch-santipuri": area(26.0491877, 90.60, "OSM Goalpara", "Goalpara district, Assam"),
    # --- SIL 2015 Palakkad tribal survey (silesr2015_028), table 4 ------------------------------
    "palakkad-irula-elachivazhi": AGALI,
    "palakkad-irula-goolikadavu": AGALI,
    "palakkad-irula-kolappady": gn_approx(11.13449, 76.58221, "11460692", "Kolapadigai, Attappady; spelling variant of Kolappady"),
    "palakkad-irula-mandhimala": AGALI,
    "palakkad-irula-nakkuppathy": gn(11.08629, 76.62764, "11460747", "Nakkapadi, 4 km from Agali; spelling variant of Nakkupathy"),
    "palakkad-irula-varagampady": gn(11.06513, 76.70686, "11460815", "Varagambadi, Attappady"),
    "palakkad-irula-nadupathy": gn(10.85586, 76.82626, "11500603", "Natuppatti, Walayar"),
    "palakkad-irula-vilamarathoor": METTUPALAYAM,
    "palakkad-irula-kunjapana": KUNJAPANAI,
    "palakkad-muduga-chittoor": gn(11.06294, 76.65167, "11460742", "Chittur, Attappady"),
    "palakkad-muduga-dundoor": gn(11.06364, 76.61425, "11460738", "Thunduru, Attappady"),
    "palakkad-kurumba-gottiyoorkandi": gn(11.12388, 76.56155, "11460697", "Gottiyakandi, Attappady"),
    "palakkad-kurumba-thudukki": AGALI,
    "palakkad-kadar-cherunelli-colony": NEMMARA,
    "palakkad-kadar-parambikulam": gn(10.39045, 76.79239, "1260383", "Parambikulam"),
    "palakkad-malasar-oriental-estate": NEMMARA,
    "palakkad-malasar-ellakkadu": NALLEPILLY,
    "palakkad-eravallan-sarkarpathy": gn(10.48142, 76.86982, "11484629", "Sarkarpathy, near Parambikulam"),
    "palakkad-eravallan-chappakkad": gn(10.57605, 76.78924, "11484452", "Chappakkad"),
    # --- SIL 2015 Idukki tribal survey (silesr2015_029), table 5 -------------------------------
    "idukki-muthuvan-itticity": ADIMALI,
    "idukki-muthuvan-chempakathozhu": DEVIKULAM,
    "idukki-muthuvan-kavakudi": DEVIKULAM,
    "idukki-muthuvan-kozhiyala": DEVIKULAM,
    "idukki-muthuvan-valsapetti": DEVIKULAM,
    "idukki-muthuvan-kunchipara": KOTHAMANGALAM,
    "idukki-muthuvan-thalayirappan": ADIMALI,
    "idukki-muthuvan-kurathikudi": ADIMALI,
    "idukki-mannan-vattamedu": osm(9.8552162, 76.9883899, "node Vattamedu Bhagam", "Vattamedu, Mariyapuram, Idukki"),
    "idukki-mannan-veliyampara": p(10.1166868, 76.9200180, "C", "openstreetmap", "OSM node Veliyampara Bhagam Pond", "Veliyampara locality, Mankulam; approximate"),
    "idukki-mannan-kumily": osm(9.6067952, 77.1671018, "node Kumily", "Kumily, Peermade"),
    "idukki-mannan-kovilmala": KATTAPPANA,
    "idukki-mannan-kodakallu": ADIMALI,
    "idukki-mannan-chinnaparakudi": ADIMALI,
    "idukki-mannan-thinkalkadu": NEDUMKANDAM,
    "idukki-urali-poovandikudi": KATTAPPANA,
    "idukki-urali-vanchivayal": PEERMADE,
    # --- SIL 2018 Nilgiri Irula survey (silesr2018_010) ------------------------------------------
    "nilgiri-irula-kunjapanai": KUNJAPANAI,
    "nilgiri-irula-kolikarai": gn(11.36899, 76.91901, "11461327", "Kolikkara, Kotagiri taluk; spelling variant of Kolikarai"),
    "nilgiri-irula-chemmanarai": gn_approx(11.38733, 76.93077, "11461348", "Sembanare, Kotagiri taluk; spelling variant of Chemmanarai"),
    "nilgiri-irula-kilkupkad": KOTAGIRI,
    "nilgiri-irula-mettukal": gn(11.41661, 76.97158, "11461490", "Mettukal, Kotagiri taluk"),
    "nilgiri-irula-chokkanalli": gn(11.52621, 76.71067, "11460120", "Sokkanhalli near Masinagudi; spelling variant of Chokkanalli"),
    "nilgiri-irula-mavanalla": osm(11.5478834, 76.6749160, "node Mavanalla", "Mavanalla, Masinagudi"),
    "nilgiri-irula-anaikatty": MASINAGUDI,
    "nilgiri-irula-bookapuram": gn(11.5401, 76.63761, "11460064", "Bokkepuram, Masinagudi"),
    "nilgiri-irula-thaliyur": METTUPALAYAM,
    "nilgiri-irula-nellithurai": osm(11.2855250, 76.8858111, "node Nellithurai", "Nellithurai, Mettupalayam"),
    # --- SIL 2019 Mudhili Gadaba survey (silesr2019_005) -----------------------------------------
    "sil-gadaba-2019-bobbilivalasa": gn(18.5069, 83.11408, "10782147", "Bobbilivalasa, Pachipenta mandal"),
    "sil-gadaba-2019-gogaduvalasa": PACHIPENTA,
    "sil-gadaba-2019-suregadivalasa": gn_approx(18.46445, 83.19036, "10782532", "Siragadivalasa, Pachipenta mandal; spelling variant of Suregadivalasa"),
    "sil-gadaba-2019-chinachipuruvalasa": gn(18.54307, 83.15568, "10782171", "Chinna Chipuruvalasa"),
    "sil-gadaba-2019-panukuvalasa": gn(18.43661, 83.19322, "10782538", "Panukuvalasa, 9 km from Salur"),
    "sil-gadaba-2019-reyavanivalasa": gn_approx(18.60248, 83.1999, "10782200", "Rayivanivalasa, Salur mandal; spelling variant of Reyavanivalasa"),
    "sil-gadaba-2019-kothavalasa": gn(18.62327, 83.16057, "10782212", "Kottavalasa, Salur mandal (nearest of several)"),
    # --- SIL 2011 Kuki-Chin of Bangladesh --------------------------------------------------------
    "sil-kuki-chin-2011-prongphung-para": RUMA,
    # --- Santali Cluster 2010 (Bangladesh) -------------------------------------------------------
    "santali_patichora": gn(25.06643, 88.76347, "7481552", "Patichara, Patnitala, Naogaon"),
    "mahali_matindor": gn(25.03302, 88.64375, "7488924", "Matindar, Patnitala, Naogaon"),
    "mahali_pachondor": gn(24.59277, 88.48884, "1191644", "Pachandhar, Tanore, Rajshahi"),
    "koda_kundang": gn_approx(24.57887, 88.52234, "7482065", "Kundain, Tanore, Rajshahi; spelling variant of Kundang"),
    # --- Lehr / Morgenstierne Pashai localities --------------------------------------------------
    "gul": gn(35.1336, 69.3015, "1140833", "Golbahar, Kapisa"),
    "pch": NIJRAB,
    "nij": osm(35.0462800, 69.6459760, "relation Nijrab District", "Nijrab district, Kapisa"),
    "nir": NIJRAB,
    "shut": gn(35.29402, 69.30271, "7054342", "Shutul district, Panjshir"),
    "uzb": gn(34.74083, 69.88391, "1472753", "Uzbin valley, Tagab"),
    "sham": region(34.90, 69.74, "Kapisa province", "Shamakat not located; Kapisa display point"),
    "laur": gn_approx(34.82727, 69.8286, "1134680", "Lowrowan, Tagab area; identification with Laurowan approximate"),
    "kch": gn_approx(34.83702, 70.31238, "1138916", "Kachur, Laghman; identification with Kachur-i Sala not certain"),
    "knd": gn(34.72718, 70.60425, "1473997", "Kandak, Laghman"),
    "kurd": gn_approx(34.99954, 70.64269, "1135091", "Kuz Kordar, Alingar valley; identification with Kurdar approximate"),
    "weg": gn_approx(34.72862, 70.55518, "1121825", "Waygal (Laghman), Alingar valley; identification with Wegal approximate"),
    "dar": gn(34.62382, 70.60348, "1473914", "Darah-ye Nur, Nangarhar"),
    "lagh": osm(34.6517980, 70.0950830, "relation Mihtarlam District", "Mihtarlam, Laghman"),
    "ish": TAGAB,
    "isk": TAGAB,
    "she": gn(34.57169, 70.58859, "1124973", "Shewa, Nangarhar"),
    "HKAT-psh_ai": gn(34.8965, 69.72049, "1149107", "Alasay district centre, Kapisa (Alasai and Alingar previously shared one point)"),
    # --- Nuristani villages -----------------------------------------------------------------------
    "ames": AMESHDESH,
    "dialect:Nuristani%20Kalasha%3A%20Amesdes": AMESHDESH,
    "vagal": WAYGAL,
    "dialect:Nuristani%20Kalasha%3A%20Vagal": WAYGAL,
    "nured-Wg-wg": WAYGAL,
    "usut": PASHKI, "dialect:Prasun%3A%20Usut": PASHKI,
    "ucu": DEWA, "dialect:Prasun%3A%20Ucu": DEWA,
    "supu": ISHTIWI, "dialect:Prasun%3A%20Supu": ISHTIWI,
    "sec": PRONZ, "dialect:Prasun%3A%20Sec": PRONZ, "nured-Pr-pr": PRONZ,
    "zumu": ZUMU, "dialect:Prasun%3A%20Zumu": ZUMU, "nured-Pr-z": ZUMU,
    "nured-Pr-k": KATAR,
    "nured-Kt-w": region(35.38, 70.78, "Strand's Kâta dialect map", "Western Katë: Ktivi/Kulam/Ramgel valleys centroid"),
    "nured-Kt-ne": region(35.67, 71.34, "Strand's Kâta dialect map", "Northeastern Katë: upper Bashgal around Barg-i Matal"),
    "nured-Kt-se": region(35.45, 71.32, "Strand's Kâta dialect map", "Southeastern Katë: middle Bashgal around Mumgrom/Mandagal"),
    # --- Bhadarwah (Doda district) villages of the Bhadrawahi/Khashali source ------------------
    "bhid": BHADERWAH, "rudh": BHADERWAH, "hrudh": BHADERWAH, "midrudh": BHADERWAH, "lrudh": BHADERWAH,
    "marm": gn(33.01797, 75.51985, "10675494", "Marmat, Doda"),
    "seu": gn_approx(33.08686, 75.59173, "10675808", "Seuth, Doda; spelling variant of Seuti"),
    # --- Pakistan --------------------------------------------------------------------------------
    "Kand": osm(35.4704725, 73.1558797, "relation Kandia Tehsil", "Kandia valley, Upper Kohistan"),
    "ChilasKhinar": osm(35.4325159, 74.1513326, "node Khinar", "Khinar, Chilas"),
    "kol": gn(35.04738, 72.96556, "1173451", "Kolai, Kolai-Palas Kohistan"),
    "SSNP-hindko-JAM": area(34.1436136, 73.21, "OSM Abbottabad", "Abbottabad (Jammun village not located)"),
    "SSNP-hindko-MAN": osm(34.3286, 73.1993, "node Mansehra", "Mansehra city"),
    "SSNP-kohistani-DSH": KALKOT,
    "SSNP-kohistani-RAJ": KALKOT,
    "Kho-Bashir-place-khost": osm(36.4413989, 72.3260428, "node Roi Khot", "Khot valley, Torkhow, Chitral"),
    "Kho-Bashir-place-warijun": osm(36.2971736, 72.2134465, "node Warijun", "Warijun, Mulkhow, Chitral"),
    "Kho-Bashir-place-bang": osm(36.5252375, 72.7648026, "node Bang Bala", "Bang, Yarkhun valley, Chitral"),
    "Kho-Bashir-place-sor-laspur": gn(36.04784, 72.46796, "1164545", "Sor Laspur, Laspur valley, Chitral"),
    "Kho-Bashir-place-mogh": osm(36.0125926, 71.6543542, "node Mogh", "Mogh, Lutkoh valley, Chitral"),
    "Hoper": p(36.1600452, 74.7465891, "B", "manual-cross-reference", "existing Jambu point dialect:Hopar", "Hopar valley, Nagar"),
    # --- India, misc -----------------------------------------------------------------------------
    "mewari_godra": area(25.2892459, 73.82, "OSM Rajsamand", "Rajsamand district (Godra village not located); previously shared Gothda's point"),
    "Dehati-Kirkkichiyapur": ATARRA,
    "Hindi-Gabchariyapur": ATARRA,
    "Tharu-BNM": area(29.2949950, 79.41, "OSM Nainital", "Nainital district (Madinapur not located)"),
    "Tharu-BNT": gn(29.28241, 79.05254, "10804431", "Thari, Ramnagar tehsil"),
    "beine_bsa": gn(19.05734, 82.00928, "10773988", "Sargipal, Jagdalpur tahsil"),
    "patiala": osm(30.3301995, 76.4007656, "relation Patiala", "Patiala city"),
    "pow": region(30.70, 76.60, "Powadh region (Rupnagar–Fatehgarh Sahib)", "regional label; centroid of the Powadh tract"),
    "jt": region(28.90, 76.40, "Jat tract (Rohtak–Bhiwani–Hisar)", "regional label; centroid of the Jatu-speaking tract"),
    "bang": region(29.30, 76.30, "Bangar tract (Jind–Kaithal)", "regional label; centroid of the Bangar tract"),
    "koh": region(35.17, 73.32, "Kolai-Palas Kohistan", "regional label kept at the Kolai-Palas Shina area point"),
    # --- SDML sites whose printed coordinates duplicate a neighbouring village -----------------
    "sdml-wardha-ashti-khadka": gn(21.08263, 78.1513, "10212950", "Khadka, Ashti taluka, Wardha; source printed Yeoor's coordinate"),
    "sdml-aurangabad-paithan-pachod-budruk": osm(19.5736425, 75.6273197, "node Pachod", "Pachod, Paithan; source printed Telwadi's coordinate"),
    "sdml-thane-ambernath-wangni": gn(19.09244, 73.29709, "11186259", "Vangani, Ambernath; source printed Usatane's coordinate"),
    "sdml-amravati-daryapur-bhambora": osm(20.9293660, 77.4264690, "node Bhambora", "Bhambora, Daryapur"),
    "sdml-amravati-daryapur-jitapur": osm(20.9357091, 77.4256480, "node Jitapur", "Jitapur, Daryapur"),
    "sdml-amravati-dharni-kusumkot-khurd": osm(21.5371524, 76.8651391, "node Kusumkot Bk", "Kusumkot (Khurd/Budruk), Dharni; source printed Kawdaziri's coordinate"),
    # --- Historical and comparative-language labels ---------------------------------------------
    "mg": region(25.10, 85.40, "Magadha historical region", "approximate historical reference point (Magadha, south Bihar)"),
    "pais": region(33.50, 73.00, "traditional north-western attribution of Paiśācī", "approximate historical reference point; attribution disputed"),
    "bul": p(42.73, 25.49, "C", "manual-centroid", "country centroid", "comparative-language label; Bulgaria"),
    "gr": p(39.07, 21.82, "C", "manual-centroid", "country centroid", "comparative-language label; Greece"),
    "it": p(42.50, 12.57, "C", "manual-centroid", "country centroid", "comparative-language label; Italy"),
    "SEeur": region(42.67, 21.17, "Balkan peninsula", "regional comparative-language label; Balkans"),
    "boh": p(49.80, 15.00, "C", "manual-centroid", "regional centroid", "comparative-language label; Bohemia"),
    "hung": p(47.16, 19.50, "C", "manual-centroid", "country centroid", "comparative-language label; Hungary"),
    "pol": p(52.07, 19.48, "C", "manual-centroid", "country centroid", "comparative-language label; Poland"),
    "rus": p(55.75, 37.62, "C", "manual-centroid", "country reference point", "comparative-language label; Russia (Moscow)"),
    "rum": p(45.94, 24.97, "C", "manual-centroid", "country centroid", "comparative-language label; Romania"),
    # --- Bhil-area SIL surveys (ESR 2018-011 Bareli, ESR 2012 Nimadi, ESR 2009 Malvi, Dhule 2013,
    #     ESR 2015-012 Noira). These rows had no coordinates at all, so the browser drew every site
    #     on its parent language's point. Tahsil/district context is printed in each Location. ------
    "sil-bareli-2018-rathwi-pauri-amalwadi": gn(21.4064, 75.20351, "10690073", "Amalvadi, Chopda tahsil, Jalgaon"),
    "sil-bareli-2018-rathwi-pauri-segwi": gn(21.63439, 75.10678, "11023085", "Segvi, Niwali tahsil, Barwani"),
    "sil-bareli-2018-rathwi-bareli-tharadpura": area(21.8235, 75.6109, "OSM Khargone", "Khargone district (Tharadpura not located)"),
    "sil-bareli-2018-rathwi-bareli-udainagar": gn(22.53942, 76.20485, "1253990", "Udainagar, Bagli tahsil, Dewas"),
    "sil-bareli-2018-rathwi-bareli-chiklia": gn(21.83947, 74.94076, "10828905", "Chiklia, Barwani"),
    "sil-bareli-2018-rathwi-chenpur": area(21.80017, 76.13266, "GeoNames 10492984 Jhirniya", "Jhirniya tahsil HQ, Khargone"),
    "sil-bareli-2018-rathwi-dongargaon": gn(21.62665, 76.37105, "10475546", "Dongargaon, Pandhana tahsil, Khandwa"),
    "sil-bareli-2018-bhilali-bodugam": area(22.29278, 74.40906, "GeoNames 12682340 Alirajpur", "Alirajpur tahsil"),
    "sil-bareli-2018-bhili-punyawat": gn(22.48472, 74.2714, "10687002", "Puniawat, Alirajpur tahsil"),
    "sil-bareli-2018-bhili-anjhera": gn(22.55783, 75.11849, "1278783", "Amjhera, Gandhwani tahsil, Dhar"),
    "sil-bareli-2018-bhilali-anjhera": gn(22.55783, 75.11849, "1278783", "Amjhera, Gandhwani tahsil, Dhar (same village as the Bhili list)"),
    "sil-bareli-2018-bhilali-mandwi": area(21.80017, 76.13266, "GeoNames 10492984 Jhirniya", "Jhirniya tahsil HQ, Khargone"),
    "sil-bareli-2018-bhilali-navalpura": gn(21.67591, 75.11134, "11023112", "Nawalpura, Sendhwa tahsil, Barwani"),
    "sil-bareli-2018-bhilali-agar": gn(22.40272, 74.81414, "10812558", "Agar, Bagh tahsil, Dhar"),
    "sil-bareli-2018-bhilali-udaigadh": osm(22.53258, 74.53693, "node Udaigarh", "Udaigarh, Jobat tahsil, Alirajpur"),
    "sil-bareli-2018-bhilali-kattivada": gn(22.48078, 74.14972, "1267498", "Kathiwara (Kattivada), Alirajpur"),
    "sil-bareli-2018-parya-bhilali-bhorwada": gn_approx(21.83444, 75.25153, "10690097", "Bharvada, Rajpur tahsil, Barwani; spelling variant of Bhorwada"),
    "sil-bareli-2018-bhili-piplia": gn_approx(22.83987, 74.55865, "10808583", "Piplia, Jhabua tahsil (nearest of several Piplias)"),
    "sil-bareli-2018-bhili-kharod": area(22.85157, 74.30923, "GeoNames 12681448 Dohad", "Dahod tahsil, Gujarat"),
    "sil-bareli-2018-bhilali-aspai": area(22.29278, 74.40906, "GeoNames 12682340 Alirajpur", "Alirajpur tahsil"),
    "sil-bareli-2018-rathawi-mankodi": area(22.08249, 74.03053, "GeoNames 12681457 Kavant", "Kawant tahsil, Chhota Udaipur"),
    "sil-bareli-2018-palya-choutharya": gn_approx(21.85438, 75.09645, "11022942", "Chautria, Rajpur tahsil, Barwani; spelling variant of Choutharya"),
    "sil-bareli-2018-palya-natvada": gn(21.38942, 74.94905, "10829315", "Natvada, Shirpur tahsil, Dhule"),
    "sil-bareli-2018-bareli-pauri-shahana": gn(21.60973, 74.74595, "10828179", "Shahana, Shahada tahsil, Nandurbar"),
    "sil-bareli-2018-bareli-pauri-mandvi": area(21.82436, 74.21805, "GeoNames 1273051 Dhadgaon", "Dhadgaon tahsil HQ, Nandurbar"),
    "sil-bareli-2018-bareli-pauri-khadki": gn(21.67539, 74.68919, "10828123", "Khadki, Pansemal tahsil, Barwani"),
    "sil-bareli-2018-ahirani-dhule": gn_approx(20.92623, 74.7609, "10688637", "Bhokar (Wadi Bhokar), Dhule tahsil"),
    "sil-nimadi-2012-bhilkheda-bhilala": gn(22.07048, 74.87835, "10687348", "Bhilkheda, Barwani"),
    "sil-nimadi-2012-awlia-dhar-bhilala": area(22.41667, 75.4, "GeoNames 1262076 Nalcha", "Nalcha tahsil HQ, Dhar"),
    "sil-nimadi-2012-khajuri-bhilala": gn(21.96954, 75.26653, "10690079", "Khajuri, Thikri tahsil, Barwani"),
    "sil-nimadi-2012-maheshwar-bhilala": gn(22.17592, 75.58715, "1264385", "Maheshwar, Khargone"),
    "sil-malvi-2009-bhandikhali-bhil": area(22.66553, 74.97736, "GeoNames 1256090 Sardarpur", "Sardarpur tahsil HQ, Dhar"),
    "sil-dhule-2013-vasavi-kelpada": osm(21.20965, 73.93484, "node Kelpada", "Kelpada, Navapur taluka, Nandurbar (formerly northern Dhule)"),
    "sil-dhule-2013-vasavi-dhanoura": gn_approx(21.56901, 74.28748, "10698534", "Dhanora, Nandurbar (formerly northern Dhule); spelling variant"),
    "sil-dhule-2013-vasavi-digiamba": osm(21.56931, 73.95880, "node Digiamba", "Digiamba, Akkalkuwa taluka, Nandurbar (formerly northern Dhule)"),
    "sil-dhule-2013-vasavi-amoda": area(21.37, 74.2, "GeoNames 7626542 Nandurbar", "Nandurbar district (formerly northern Dhule; Amoda ambiguous)"),
    "sil-dhule-2013-noiri-mundalwad": gn_approx(21.22377, 74.93833, "10829400", "Mudavad, Shirpur tahsil, Dhule; spelling variant of Mundalwad"),
    "sil-dhule-2013-noiri-astamba": area(21.82436, 74.21805, "GeoNames 1273051 Dhadgaon", "Dhadgaon tahsil (Astamba hill area), Nandurbar"),
    "sil-dhule-2013-pauri-bhusha": area(21.82436, 74.21805, "GeoNames 1273051 Dhadgaon", "Dhadgaon tahsil, Nandurbar"),
    "sil-dhule-2013-rathwi-kangai": area(21.37, 74.2, "GeoNames 7626542 Nandurbar", "Nandurbar district (formerly northern Dhule)"),
    "sil-noira-2015-noiri-chillare": gn_approx(21.3846, 75.0085, "10690064", "Chilar, Shirpur tahsil, Dhule; spelling variant of Chillare"),
    "sil-noira-2015-noiri-pannali": gn(21.71156, 74.67651, "10828094", "Pannali, Pansemal tahsil, Barwani"),
    "sil-noira-2015-noiri-gomon": area(21.70056, 74.00019, "OSM Akkalkuwa Taluka", "Akkalkuwa taluka, Nandurbar"),
    "sil-noira-2015-dungra-bhili-mathwad": osm(22.01439, 74.22112, "node Mathwad", "Mathwad, Sondwa tahsil, Alirajpur"),
    "sil-noira-2015-dungra-bhili-ambadungar": area(22.08249, 74.03053, "GeoNames 12681457 Kavant", "Kawant tahsil, Chhota Udaipur"),
    "sil-noira-2015-kotli-narayanpur": gn_approx(21.37276, 74.18121, "10578677", "Narayanpur near Nandurbar town; Papiner not located"),
    "sil-noira-2015-kotli-taradi": gn_approx(21.60686, 74.39357, "10698487", "Tarhadi, Shahada tahsil; spelling variant of Taradi"),
    "sil-noira-2015-gujari-taradi": gn_approx(21.60686, 74.39357, "10698487", "Tarhadi, Shahada tahsil; spelling variant of Taradi"),
    "sil-noira-2015-korku-tukaithad": area(21.37276, 76.52599, "GeoNames 12680477 Khaknar", "Khaknar block, Burhanpur"),
    "sil-noira-2015-nihali-jamod": gn(21.05194, 76.53464, "1269406", "Jalgaon Jamod, Buldhana"),
    "sil-noira-2015-korku-tembhi": p(21.07, 76.50, "C", "manual-map", "Nihali village Tembi near Jalgaon Jamod", "approximate"),

    # =============================================================================================
    # Dialects that had NO coordinates at all (476 rows on 2026-09-17). Sites with a named village
    # or town get a gazetteer point; sites naming only a tahsil/block get that unit (C). Pure
    # regional or register labels ("Uttar Pradesh", "Standard Bangla list", Zoller "Western")
    # stay blank on purpose.
    # --- SIL ESR 2009-011 Malvi (tahsil printed per site) ----------------------------------------
    "sil-malvi-2009-thillorkhurd-ujjaini": gn(22.6153, 75.95363, "10689647", "Tillor Khurd, Indore"),
    "sil-malvi-2009-kumardi-ujjaini": gn_approx(23.00652, 76.29105, "10752799", "Kumari, Sonkatch; spelling variant of Kumardi"),
    "sil-malvi-2009-samapura-gond": gn_approx(22.91029, 77.1216, "10496464", "Samanpura, Ichhawar; spelling variant of Samapura"),
    "sil-malvi-2009-harsodan-ujjaini": area(23.1793, 75.7849, "Ujjain town", "Ujjain tahsil"),
    "sil-malvi-2009-chandukhedi-ujjaini": area(23.1793, 75.7849, "Ujjain town", "Ujjain tahsil"),
    "sil-malvi-2009-nain-ujjaini": area(23.4564, 75.4177, "Nagda town", "Nagda tahsil, Ujjain"),
    "sil-malvi-2009-koyal-ujjaini": area(23.4867, 75.6608, "Mahidpur town", "Mahidpur tahsil, Ujjain"),
    "sil-malvi-2009-rojdi-ujjaini": area(22.7196, 75.8577, "Indore city", "Indore tahsil"),
    "sil-malvi-2009-jokhar-ujjaini": area(23.2591, 76.1454, "Maksi town", "Maksi tahsil, Shajapur"),
    "sil-malvi-2009-bercha-ujjaini": area(22.9676, 76.0534, "Dewas city", "Dewas (respondent resident there)"),
    "sil-malvi-2009-moondikhedi-ujjaini": area(23.0176, 76.7221, "Ashta town", "Ashta tahsil, Sehore"),
    "sil-malvi-2009-lojithara-rajwadi": area(23.3315, 75.0367, "Ratlam city", "Ratlam tahsil"),
    "sil-malvi-2009-bhimakhedi-rajwadi": area(23.6377, 75.1262, "Jaora town", "Jaora tahsil, Ratlam"),
    "sil-malvi-2009-kishorpura-rajwadi": area(24.0148, 75.3535, "Sitamau town", "Sitamau tahsil, Mandsaur"),
    "sil-malvi-2009-bhunyakhedi-rajwadi": area(24.0768, 75.0693, "Mandsaur city", "Mandsaur tahsil"),
    "sil-malvi-2009-jesingpura-rajwadi": area(24.4764, 74.8624, "Neemuch city", "Neemuch tahsil"),
    "sil-malvi-2009-bhandia-rajwadi": area(24.4756, 75.1447, "Manasa town", "Manasa tahsil, Neemuch"),
    "sil-malvi-2009-jhadmu-umadwadi": area(24.0235, 76.3773, "Zirapur town", "Zirapur tahsil, Rajgarh"),
    "sil-malvi-2009-semlikakad-umadwadi": area(24.0414, 76.5807, "Khilchipur town", "Khilchipur tahsil, Rajgarh"),
    "sil-malvi-2009-sagpur-umadwadi": area(23.7076, 77.0932, "Narsinghgarh town", "Narsinghgarh tahsil, Rajgarh"),
    "sil-malvi-2009-paldyabana-umadwadi": area(23.7076, 77.0932, "Narsinghgarh town", "Narsinghgarh tahsil, Rajgarh"),
    "sil-malvi-2009-mungavali-umadwadi": area(23.2032, 77.0844, "Sehore town", "Sehore tahsil"),
    "sil-malvi-2009-harnauda-sondhwadi": area(24.4118, 75.6274, "Gangdhar town", "Gangdhar tahsil, Jhalawar"),
    "sil-malvi-2009-narana-sondhwadi": area(24.0316, 76.0367, "Pirawa town", "Pirawa tahsil, Jhalawar"),
    "sil-malvi-2009-adakhedi-sondhwadi": area(24.0316, 76.0367, "Pirawa town", "Pirawa tahsil, Jhalawar"),
    "sil-malvi-2009-era-sondhwadi": area(24.4147, 76.5745, "Bhawani Mandi", "Pachpahar tahsil, Jhalawar"),
    "sil-malvi-2009-deevdi-sondhwadi": area(24.5426, 76.1675, "Jhalrapatan town", "Jhalrapatan tahsil, Jhalawar"),
    "sil-malvi-2009-jamli-sondhwadi": area(23.7124, 76.0157, "Agar town", "Agar tahsil"),
    "sil-malvi-2009-kalwar-gond": area(22.6684, 76.7413, "Kannod town", "Kannod tahsil, Dewas"),
    # --- SIL 2022 Bagheli -------------------------------------------------------------------------
    "sil-bagheli-2022-amarkantak": gn(22.67486, 81.75908, "1278905", "Amarkantak"),
    "sil-bagheli-2022-karchana": gn(25.28608, 81.93687, "1267806", "Karchana, Prayagraj"),
    "sil-bagheli-2022-baikanthpur": gn(24.72768, 81.40975, "1277770", "Baikunthpur, Sirmour tahsil, Rewa"),
    "sil-bagheli-2022-dewara": gn(24.68631, 81.98819, "10533094", "Dewara, Hanumana tahsil, Rewa"),
    "sil-bagheli-2022-singpur": gn(23.20939, 81.41907, "1256139", "Singpur, Sohagpur tahsil, Shahdol"),
    "sil-bagheli-2022-silpari": gn(24.63929, 81.61456, "10532479", "Silpari, Rewa"),
    "sil-bagheli-2022-katkon": gn_approx(24.63285, 80.61077, "10538710", "Kathkone, Nagod tahsil; spelling variant of Katkon"),
    "sil-bagheli-2022-chawari": gn_approx(24.20516, 81.71172, "10534401", "Chhawari, Sidhi; spelling variant of Chawari"),
    "sil-bagheli-2022-parasawar": gn_approx(24.48019, 82.13898, "10551185", "Paraswar, Deosar tahsil; spelling variant of Parasawar"),
    "sil-bagheli-2022-mahdeiya": gn_approx(24.20957, 82.56064, "10551656", "Mahdeiyan, Singrauli; spelling variant of Mahdeiya"),
    "sil-bagheli-2022-keoti": p(24.83, 81.38, "C", "manual-map", "Keoti, Rewa district", "Keoti village (near Keoti falls); approximate"),
    "sil-bagheli-2022-dabhaura": area(24.9836, 81.6393, "Teonthar town", "Teonthar tahsil, Rewa"),
    "sil-bagheli-2022-sunwari": area(24.2652, 80.7611, "Maihar town", "Maihar tahsil, Satna"),
    "sil-bagheli-2022-domahai": area(24.8165, 80.6167, "Majhgawan", "Majhgawan tahsil, Satna"),
    "sil-bagheli-2022-janakpur": area(23.7883, 82.1133, "Bharatpur (Koriya)", "Bharatpur tahsil, Koriya"),
    "sil-bagheli-2022-lodha": area(23.5258, 80.8369, "Umaria town", "Umaria tahsil"),
    "sil-bagheli-2022-kotasiv-prathapsing": area(24.9165, 82.3241, "Lalganj (Mirzapur)", "Lalganj tahsil, Mirzapur"),
    "sil-bagheli-2022-semara": area(23.6836, 81.389, "Jaisinghnagar", "Jaisinghnagar tahsil, Shahdol"),
    # --- SIL 1985 Kullu survey ---------------------------------------------------------------------
    "sil-kullu-1985-manali": gn(32.26076, 77.18786, "12682516", "Manali"),
    "sil-kullu-1985-maraur": gn(31.78865, 77.53787, "1263550", "Maraur, Banjar"),
    "sil-kullu-1985-bathad": gn(31.59961, 77.48081, "1276706", "Bathad, Banjar"),
    "sil-kullu-1985-manikaran": gn(32.0266, 77.36074, "1263721", "Manikaran"),
    "sil-kullu-1985-ani": gn(31.4498, 77.41281, "12682525", "Ani"),
    "sil-kullu-1985-kullu": p(31.9579, 77.1095, "B", "manual-map", "Kullu town", "Kullu HQ"),
    "sil-kullu-1985-jibhi": p(31.59, 77.35, "C", "manual-map", "Jibhi, Banjar tehsil", "approximate"),
    "sil-kullu-1985-shangarh": p(31.68, 77.37, "C", "manual-map", "Shangarh, Banjar tehsil", "approximate"),
    "sil-kullu-1985-garsah": p(31.90, 77.23, "C", "manual-map", "Garsa, Kullu tehsil", "approximate"),
    "sil-kullu-1985-churla": area(31.9579, 77.1095, "Kullu town", "Kullu tehsil (Lag valley)"),
    "sil-kullu-1985-loren": area(31.9579, 77.1095, "Kullu town", "Kullu tehsil"),
    "sil-kullu-1985-raila": area(31.9579, 77.1095, "Kullu town", "Kullu tehsil"),
    "sil-kullu-1985-bhutti": area(31.9579, 77.1095, "Kullu town", "Kullu tehsil (Lag valley)"),
    "sil-kullu-1985-shalwar": area(31.6383, 77.3431, "Banjar town", "Banjar tehsil"),
    "sil-kullu-1985-chinninal": area(31.6383, 77.3431, "Banjar town", "Banjar tehsil"),
    "sil-kullu-1985-sidua": area(31.6383, 77.3431, "Banjar town", "Banjar tehsil"),
    # --- Varenkamp 2024 Ho field lists (1989) -----------------------------------------------------
    "sil-ho-2024-hbg": gn(22.72127, 85.50261, "11686498", "Nakti, Bandgaon"),
    "sil-ho-2024-hsu": area(21.83525, 85.0772, "GeoNames 12681027 Lahunipara", "Lahunipara block, Sundargarh"),
    "sil-ho-2024-hsa": area(21.539, 85.00849, "GeoNames 1277101 Barkot", "Barkot, Deogarh (historical Sambalpur)"),
    "sil-ho-2024-hdh": area(21.43354, 85.19184, "GeoNames 1260696 Pal Lahara", "Pallahara, Angul (historical Dhenkanal)"),
    "sil-ho-2024-hop": area(21.37152, 86.54672, "GeoNames 12681034 Oupada", "Oupada block, Balasore"),
    "sil-ho-2024-hth": area(21.7215, 86.1163, "Thakurmunda", "Thakurmunda block, Mayurbhanj"),
    "sil-ho-2024-hka": area(21.7614, 85.9762, "Karanjia", "Karanjia, Mayurbhanj"),
    "sil-ho-2024-hke": area(21.6318, 85.5969, "Keonjhar town", "Keonjhar district"),
    "sil-ho-2024-hch": area(22.5548, 85.8175, "Chaibasa", "Chaibasa, West Singhbhum"),
    "sil-ho-2024-hcu": area(20.4625, 85.883, "Cuttack", "Cuttack district"),
    "sil-ho-2024-hjo": area(21.9583, 86.05, "Jashipur", "Jashipur, Mayurbhanj"),
    "sil-ho-2024-hra": area(22.265, 86.1735, "Rairangpur", "Rairangpur, Mayurbhanj"),
    "sil-ho-2024-hba": area(21.9322, 86.7517, "Baripada", "Baripada, Mayurbhanj"),
    "sil-ho-2024-hni": area(21.462, 86.7677, "Nilgiri (Balasore)", "Nilgiri, Balasore"),
    # --- Nimadi (ESR 2012 / ESR 2018-011) ----------------------------------------------------------
    "sil-bareli-2018-nimadi-khargone": gn(21.82306, 75.61028, "8739996", "Khargone city"),
    "sil-nimadi-2012-khargone-general": gn(21.82306, 75.61028, "8739996", "Khargone city"),
    "sil-bareli-2018-nimadi-awlia": gn(21.95789, 75.89032, "11031400", "Avalia (Awlia), Khandwa"),
    "sil-nimadi-2012-awlia-khandwa-balai": gn(21.95789, 75.89032, "11031400", "Avalia (Awlia), Khandwa"),
    "sil-bareli-2018-nimadi-ashapur": gn(22.28088, 75.56169, "10689287", "Asapur, Maheshwar tahsil"),
    "sil-nimadi-2012-sonipura-balai": gn(21.85715, 75.66218, "10706435", "Sonipura, Khargone"),
    "sil-nimadi-2012-sonipura-patidar": gn(21.85715, 75.66218, "10706435", "Sonipura, Khargone"),
    "sil-nimadi-2012-sirpur-melgav-obc": gn(21.81647, 76.62303, "10477153", "Sirpur, Khalwa tahsil, Khandwa"),
    "sil-nimadi-2012-balkhad-brahmin": area(22.128, 75.6118, "Kasrawad", "Kasrawad tahsil, Khargone"),
    "sil-nimadi-2012-jajamkhedi-obc": area(22.2386, 75.0885, "Manawar", "Manawar tahsil, Dhar"),
    "sil-nimadi-2012-rupkheda-brahmin": area(22.2565, 76.0399, "Barwaha", "Barwaha tahsil, Khargone"),
    "sil-nimadi-2012-kupdol-badgav-darbar": area(21.82306, 75.61028, "Khargone city", "Khargone tahsil"),
    # --- SIL 2021 Amri Karbi ---------------------------------------------------------------------
    "sil-amri-karbi-2021-paboi-misamari": gn(26.80614, 92.59768, "1262991", "Missamari, Sonitpur"),
    "sil-amri-karbi-2021-mikirgaon": gn(26.22612, 92.75657, "1263054", "Mikirgaon, Nagaon"),
    "sil-amri-karbi-2021-holanki": area(27.1478, 93.7359, "Yupia", "Papum Pare district, Arunachal Pradesh"),
    "sil-amri-karbi-2021-hajarongpi": area(25.8434, 93.4311, "Diphu", "East Karbi Anglong"),
    "sil-amri-karbi-2021-sermansingner": area(25.8434, 93.4311, "Diphu", "East Karbi Anglong"),
    "sil-amri-karbi-2021-rongtheang": area(25.8434, 93.4311, "Diphu", "East Karbi Anglong"),
    "sil-amri-karbi-2021-sardoka-ingti": area(25.8434, 93.4311, "Diphu", "East Karbi Anglong"),
    "sil-amri-karbi-2021-amguri-wka": area(25.8686, 92.6417, "Hamren", "West Karbi Anglong"),
    "sil-amri-karbi-2021-langhemphi": area(25.8686, 92.6417, "Hamren", "West Karbi Anglong"),
    "sil-amri-karbi-2021-umrinti": area(25.8686, 92.6417, "Hamren", "West Karbi Anglong"),
    "sil-amri-karbi-2021-bankri": area(25.8686, 92.6417, "Hamren", "West Karbi Anglong"),
    "sil-amri-karbi-2021-sunajoli": area(27.236, 94.1035, "North Lakhimpur", "Lakhimpur district"),
    "sil-amri-karbi-2021-amguri-kamrup": area(26.1758, 91.6675, "Amingaon", "Kamrup district"),
    "sil-amri-karbi-2021-maina-kharong": area(26.1758, 91.6675, "Amingaon", "Kamrup district"),
    "sil-amri-karbi-2021-plasha": area(25.9023, 91.8794, "Nongpoh", "Ri Bhoi district"),
    # --- Bhumij (Varenkamp field lists) -----------------------------------------------------------
    "bhumij-mundari1989-udala": gn(21.58523, 86.58529, "12683748", "Udala, Mayurbhanj"),
    "bhumij1989-baigodia": area(21.37152, 86.54672, "GeoNames 12681034 Oupada", "Oupada block, Balasore (same village as the Ho list)"),
    "zoller-mu-bhumij": region(22.2155, 86.2322, "Rairangpur, Mayurbhanj", "Bhumij core area; regional label"),
    # --- JLSR 2022-014 Korwa / Kodaku -------------------------------------------------------------
    "sil-kodaku-2005-jamuniatanr": gn(23.96304, 83.77275, "10744322", "Jamuniatanr, Ranka, Garhwa"),
    "sil-kodaku-2005-dhengura": gn(23.88487, 83.77151, "10744331", "Dhengura, Ranka, Garhwa"),
    "sil-kodaku-2005-chainpur": gn(24.98264, 83.41936, "12682988", "Chainpur (Bhabhua), Kaimur, Uttar Pradesh border"),
    "sil-kodaku-2004-sagardinwa": area(24.05, 84.22, "Chainpur (Palamu)", "Chainpur block, Palamu"),
    "sil-kodaku-2005-jhaleria": area(23.6136, 83.6088, "Balrampur (CG)", "Balrampur district, Chhattisgarh"),
    "sil-kodaku-2005-chilma": area(23.6136, 83.6088, "Balrampur (CG)", "Balrampur district, Chhattisgarh"),
    "sil-kodaku-2005-kodakupara": area(23.3838, 83.3286, "Pratappur", "Pratappur block, Surajpur"),
    "sil-kodaku-2005-tharki": area(23.3949, 83.1235, "Rajpur (Balrampur)", "Rajpur block"),
    "sil-kodaku-2005-baikanthpur": area(23.7936, 83.3536, "Wadrafnagar", "Wadrafnagar block, Balrampur"),
    "sil-korwa-2004-chilma": area(23.3949, 83.1235, "Rajpur (Balrampur)", "Rajpur block"),
    "sil-korwa-2004-dhaneshpur": area(23.2789, 83.3934, "Kusmi", "Kusmi block, Balrampur"),
    "sil-korwa-2005-gaseband": area(22.9494, 83.7727, "Bagicha", "Bagicha block, Jashpur"),
    "sil-korwa-2004-harrapat": area(22.906, 83.965, "Manora", "Manora block, Jashpur"),
    "sil-korwa-2004-bladerpat": area(23.081, 83.665, "Sanna", "Sanna, Jashpur"),
    "sil-korwa-2004-kirkima": area(23.183, 83.4118, "Lundra", "Lundra block, Surguja"),
    "sil-korwa-2004-musakhoel": area(23.12, 83.195, "Ambikapur", "Ambikapur, Surguja"),
    "sil-korwa-2004-rakkaya": area(23.5735, 83.029, "Shankargarh", "Shankargarh block, Balrampur"),
    "sil-korwa-2005-sardih": area(22.3595, 82.7501, "Korba", "Korba district"),
    # --- SIL Bonda / Didayi surveys ----------------------------------------------------------------
    "sil-bonda-didayi-1997-chitrakonda-l-didayi": gn(18.21675, 82.13531, "12683860", "Chitrakonda"),
    "sil-bonda-didayi-1997-biapada-u-didayi": area(18.21675, 82.13531, "Chitrakonda", "Chitrakonda block, Malkangiri"),
    "sil-bonda-didayi-1997-kaluguda-u-didayi": area(18.21675, 82.13531, "Chitrakonda", "Chitrakonda block, Malkangiri"),
    "sil-bonda-didayi-1997-orapadar-u-didayi": area(18.21675, 82.13531, "Chitrakonda", "Chitrakonda block, Malkangiri"),
    "sil-bonda-didayi-1997-oringi-l-didayi": area(18.21675, 82.13531, "Chitrakonda", "Chitrakonda block, Malkangiri"),
    "sil-bonda-didayi-1997-rasabeda-l-bonda": area(18.4946, 82.2851, "GeoNames 12683855 Mudulipada", "Khairput block (Bonda hills)"),
    "sil-bonda-didayi-1997-kendhuguda-l-bonda": area(18.4946, 82.2851, "GeoNames 12683855 Mudulipada", "Khairput block (Bonda hills)"),
    "sil-bonda-didayi-1997-kadamguda-l-bonda": area(18.4946, 82.2851, "GeoNames 12683855 Mudulipada", "Khairput block (Bonda hills)"),
    "sil-bonda-didayi-1997-dumripada-u-bonda": area(18.4946, 82.2851, "GeoNames 12683855 Mudulipada", "Khairput block (Bonda hills)"),
    "sil-bonda-further-2002-podeiguda-u-bonda": area(18.4946, 82.2851, "GeoNames 12683855 Mudulipada", "Khairput block (Bonda hills)"),
    "sil-bonda-further-2002-bondapada-u-bonda": area(18.4946, 82.2851, "GeoNames 12683855 Mudulipada", "Khairput block (Bonda hills)"),
    # --- SIL 1985 Korku ------------------------------------------------------------------------------
    "sil-korku-1985-chikli-ruma": gn_approx(21.89246, 77.61961, "10508913", "Chikhli, Betul; identification approximate"),
    "sil-korku-1985-khanapur-ruma": gn_approx(21.92547, 78.15348, "10212593", "Khanapur, Betul; identification approximate"),
    "sil-korku-1985-bagdara-ruma": gn_approx(21.65029, 77.60115, "10509078", "Bagdara, Betul; identification approximate"),
    "sil-korku-1985-amdhana-mawasi": gn_approx(22.36095, 78.82865, "13428119", "Amdhana, Chhindwara; identification approximate"),
    "sil-korku-1985-warsari-ruma": area(21.9059, 77.9026, "Betul", "Betul district (Ruma Korku area)"),
    "sil-korku-1985-moragao-bouriya": area(21.9059, 77.9026, "Betul", "Betul district"),
    "sil-korku-1985-lahi-bouriya": area(21.9059, 77.9026, "Betul", "Betul district"),
    "sil-korku-1985-khamalpur-bondoy": area(22.7475, 77.7302, "Hoshangabad", "Hoshangabad district"),
    # --- JLSR 2021-029 Koya ----------------------------------------------------------------------------
    "sil-koya-1985-jaganathapuram": gn_approx(17.23717, 80.78987, "10769735", "Jagannathapuram, Khammam; nearest of several"),
    "sil-koya-1985-chintoor": gn(17.76118, 81.37954, "12684186", "Chintoor"),
    "sil-koya-1985-podia": gn(18.18915, 81.52113, "12683856", "Podia, Malkangiri"),
    "sil-koya-1985-utnoor": gn(19.37241, 78.70714, "12681258", "Utnoor, Adilabad"),
    "sil-koya-1985-malakanagiri": gn(18.3479, 81.8871, "7627182", "Malkangiri town"),
    "sil-koya-1985-bhamani-gondi": area(18.8477, 79.9612, "Sironcha", "Sironcha taluka (Bhamani village not located)"),
    "sil-koya-1985-bhamani-madia": area(18.8477, 79.9612, "Sironcha", "Sironcha taluka (Bhamani village not located)"),
    # --- SIL 2007 Desia (Koraput) ---------------------------------------------------------------------
    "sil-desia-2007-souraguda-soura": gn(18.87116, 82.55821, "10777510", "Souraguda, Jeypore"),
    "sil-desia-2007-gagnapur-poroja": area(18.8563, 82.5716, "Jeypore", "Jeypore block, Koraput"),
    "sil-desia-2007-potenda-rona": area(18.5697, 82.8093, "Lamtaput", "Lamtaput block, Koraput"),
    "sil-desia-2007-ghumar-rona": area(19.1416, 82.3226, "Kotpad", "Kotpad block, Koraput"),
    "sil-desia-2007-sourakundi-bhotra": area(19.1416, 82.3226, "Kotpad", "Kotpad block, Koraput"),
    "sil-desia-2007-sabhapatiguda-gaud": area(18.3479, 81.8871, "Malkangiri", "Malkangiri district"),
    "sil-desia-2007-kantigad-gaud": area(19.0479, 82.5511, "Boriguma", "Boriguma block, Koraput"),
    "sil-desia-2007-kakalpoda-bod-mali": area(19.0479, 82.5511, "Boriguma", "Boriguma block, Koraput"),
    "sil-desia-2007-gumalput-gadaba": area(19.0479, 82.5511, "Boriguma", "Boriguma block, Koraput"),
    "sil-desia-2007-aunli-bhotra": area(19.0479, 82.5511, "Boriguma", "Boriguma block, Koraput"),
    "sil-desia-2007-jujhari-kamar": area(19.0479, 82.5511, "Boriguma", "Boriguma block, Koraput"),
    "sil-desia-2007-konda-maliguda-bod-mali": area(18.926, 82.931, "Laxmipur", "Laxmipur block, Koraput"),
    "sil-desia-2007-patta-maliguda-san-mali": area(18.926, 82.931, "Laxmipur", "Laxmipur block, Koraput"),
    "sil-desia-2007-burja-dom": area(18.926, 82.931, "Laxmipur", "Laxmipur block, Koraput"),
    "sil-desia-2007-dame-side-dom": area(18.5658, 82.965, "Pottangi", "Pottangi block, Koraput"),
    "sil-desia-2007-bodgaon-dhulia": area(18.5658, 82.965, "Pottangi", "Pottangi block, Koraput"),
    "sil-desia-2007-chhatrabor-harijan": area(19.2166, 82.4931, "Papadahandi", "Papadahandi block, Nabarangpur"),
    "sil-desia-2007-gemelput-mania": area(18.67, 82.79, "Nandapur", "Nandapur block, Koraput"),
    "sil-desia-2007-sindhiguda-bonda": area(18.4946, 82.2851, "GeoNames 12683855 Mudulipada", "Khairput block, Malkangiri"),
    # --- SIL ESR 2012-015 Kurumba control lists and sites ---------------------------------------------
    "sil-kurumba-1985-kannada-bangalore": p(12.9716, 77.5946, "B", "manual-map", "Bengaluru", "Bangalore city"),
    "sil-kurumba-1976-pudukkottai": gn(10.35, 78.9, "1259298", "Pudukkottai"),
    "sil-kurumba-1984-madapalli": gn(12.49056, 78.60101, "11348417", "Madappalli, Tirupattur"),
    "sil-kurumba-1985-karmadai-kurumba": gn(11.24058, 76.96009, "1267869", "Karamadai"),
    "sil-kurumba-1985-karmadai-vakkaliga": gn(11.24058, 76.96009, "1267869", "Karamadai"),
    "sil-kurumba-1985-kurumbapalayam": gn(10.8478, 77.0651, "11558035", "Kurumbapalayam, Coimbatore"),
    "sil-kurumba-1985-kalangal": gn(10.9923, 77.13798, "11557958", "Kalangal, Palladam"),
    "sil-kurumba-1985-masinagudi-jennu": gn(11.56816, 76.64715, "7731765", "Masinagudi"),
    "sil-kurumba-1985-maddur-betta": gn(11.77915, 76.55897, "1264593", "Maddur, Gundlupet"),
    "sil-kurumba-1985-tamil-madras": gn(13.08784, 80.27847, "1264527", "Chennai"),
    "sil-kurumba-1985-kolar": gn_approx(13.28773, 78.34199, "11322567", "Basavanahalli, Kolar; spelling variant of Basavanatha"),
    "sil-kurumba-1985-chitradurga": gn_approx(14.21738, 76.37303, "10883708", "Malavvanahatti, Chitradurga; spelling variant"),
    "sil-kurumba-1985-badaga-arvenu": gn_approx(11.40802, 76.87359, "11461356", "Arevenu, Kotagiri; spelling variant of Arvenu"),
    "sil-kurumba-1985-kotagiri-alu": p(11.35, 76.80, "B", "manual-cross-reference", "existing Jambu point palakkad-alukurumba-banigudisola", "Banigudisole, Kotagiri"),
    "sil-kurumba-1984-belavarthy": area(12.5266, 78.215, "Krishnagiri", "Krishnagiri taluk"),
    "sil-kurumba-1984-kurumbatheru": area(12.5266, 78.215, "Krishnagiri", "Krishnagiri taluk"),
    "sil-kurumba-1985-beerajjanur": area(12.5266, 78.215, "Krishnagiri", "Krishnagiri taluk"),
    "sil-kurumba-1984-buringi": area(12.495, 78.573, "Tirupattur", "Tirupattur taluk"),
    "sil-kurumba-1984-thangiyadikuppam": area(12.7495, 78.3417, "Kuppam", "Kuppam taluk, Chittoor"),
    # --- Census/People's Linguistic Survey chapter localities ------------------------------------------
    "census-tamil-nadu-1": gn(11.46986, 76.68907, "12684098", "Udhagamandalam"),
    "census-tamil-nadu-2": gn(11.46986, 76.68907, "12684098", "Udhagamandalam"),
    "census-tamil-nadu-5": gn(11.46986, 76.68907, "12684098", "Udhagamandalam"),
    "census-tamil-nadu-9": gn(11.46986, 76.68907, "12684098", "Udhagamandalam"),
    "census-tamil-nadu-4": gn(12.83359, 79.78225, "7646149", "Kanchipuram"),
    "census-tamil-nadu-6": gn_approx(13.12758, 79.25748, "1259416", "Ponnai, Vellore (one of the two named localities)"),
    "census-tamil-nadu-8": gn(11.34, 77.55, "8223995", "Erode district"),
    "census-tamil-nadu-10": area(11.5048, 77.2384, "Sathyamangalam", "Hasanur/Sathyamangalam, Erode"),
    "census-uttar-pradesh-2": gn(26.93864, 81.3274, "1277159", "Barabanki"),
    "census-uttar-pradesh-7": gn(28.66535, 77.43915, "1271308", "Ghaziabad"),
    "census-uttar-pradesh-8": gn(28.54702, 77.38838, "10265206", "Salarpur Khadar, Noida"),
    "census-uttar-pradesh-10": gn(25.54277, 79.81235, "1266937", "Kharela, Mahoba"),
    "census-uttar-pradesh-11": gn(26.87449, 81.03477, "1274161", "Chinhat, Lucknow"),
    "census-uttar-pradesh-3": gn(27.51842, 80.85633, "1255971", "Sitapur"),
    "census-uttar-pradesh-6": gn(27.18222, 79.05211, "1264293", "Mainpuri"),
    "census-uttar-pradesh-9": area(25.9226, 82.201, "Patti", "Patti tehsil, Pratapgarh"),
    "census-khortha-sikaripara": gn(24.23855, 87.47413, "1256317", "Shikaripara, Dumka"),
    "census-danuwar-0": gn(27.6437, 85.63998, "12095779", "Panchkhal, Kavrepalanchok"),
    "census-danuwar-1": gn(27.72859, 85.68126, "7995552", "Bhimtar, Sindhupalchok"),
    "census-danuwar-2": gn(27.5813, 85.3014, "7813632", "Dukuchhap, Lalitpur"),
    "census-danuwar-3": gn(27.0888, 86.0838, "7810332", "Hatpate, Sindhuli"),
    "census-danuwar-4": gn(27.24048, 85.1477, "12096150", "Nijgadh, Bara"),
    "census-tharu-5": gn(26.44202, 87.27613, "12096209", "Biratnagar (Morang)"),
    "census-tharu-6": gn(27.03226, 85.00264, "1283272", "Kalaiya (Bara)"),
    "census-tharu-7": gn(26.74253, 86.34984, "12095522", "Siraha"),
    "census-kisan-0": area(26.52545, 88.07948, "GeoNames 12096010 Bhadrapur", "Jhapa district"),
    # --- KEED regional Kannada labels ----------------------------------------------------------------------
    "keed_my": gn(12.23, 76.42, "1262322", "Mysore district"),
    "keed_kumta": p(14.4262, 74.4098, "B", "manual-map", "Kumta", "Kumta town, Uttara Kannada"),
    "keed_bellary": p(15.1394, 76.9214, "B", "manual-map", "Ballari", "Bellary city"),
    "keed_ck": region(14.43, 75.9, "Davanagere", "Central Karnataka; regional label"),
    "keed_nk": region(15.37, 75.14, "Dharwad", "Northern Karnataka; regional label"),
    "keed_sk": region(12.23, 76.42, "Mysore", "Southern Karnataka; regional label"),
    "keed_smhr": region(16.3333, 74.75, "Belgaum", "Southern Maratha country; regional label"),
    # --- SIL 1996 Eastern Gujari -----------------------------------------------------------------------------
    "sil-eastern-gujari-1996-udhampur": gn(33.0, 75.16667, "1253957", "Udhampur"),
    "sil-eastern-gujari-1996-jammu": gn(32.73528, 74.86167, "1269321", "Jammu"),
    "sil-eastern-gujari-1996-chamba": gn(32.57147, 76.10229, "1274849", "Chamba"),
    "sil-eastern-gujari-1996-nalagarh": gn(31.07116, 76.70431, "12685362", "Nalagarh"),
    "sil-eastern-gujari-1996-dehra-dun": gn(30.32443, 78.03392, "1273313", "Dehradun"),
    "sil-eastern-gujari-1996-haldwani": gn(29.14086, 79.71308, "12682626", "Haldwani"),
    "sil-eastern-gujari-1996-rampur": gn(31.44943, 77.63087, "1258596", "Rampur Bushahr"),
    "sil-eastern-gujari-1996-kotdwara": gn(29.74612, 78.52219, "1266014", "Kotdwara"),
    # --- JLSR 2024-011 Haryanvi ---------------------------------------------------------------------------------
    "sil-haryanvi-2024-hrt": gn(28.8333, 76.6667, "1258077", "Rohtak"),
    "sil-haryanvi-2024-hjn": gn(29.5, 76.25, "1268908", "Jind"),
    "sil-haryanvi-2024-hft": gn(29.51285, 75.43344, "12682421", "Fatehabad"),
    "sil-haryanvi-2024-htr": gn(28.23629, 76.95759, "12682462", "Taoru"),
    "sil-haryanvi-2024-hlh": gn(28.53054, 75.7678, "7646704", "Loharu"),
    "sil-haryanvi-2024-hng": area(30.479, 77.1315, "Naraingarh", "Naraingarh tehsil, Ambala"),
    # --- Konda Dora, Adi, Tharu, Dhurwa, misc -----------------------------------------------------------------
    "sil-konda-dora-1987-visakh": gn_approx(17.85305, 82.91521, "10770856", "Lakshmipuram, Paderu area; nearest of several"),
    "sil-konda-dora-1987-koraput": area(18.5658, 82.965, "Pottangi", "Pottangi block, Koraput"),
    "sil-adi-2015-ashing-ningging": gn(28.9533, 94.83321, "1261317", "Ningging, Upper Siang"),
    "sil-adi-2015-bori-bogu-payum": gn_approx(28.53355, 94.66434, "1275377", "Bogu, Payum circle, West Siang"),
    "sil-adi-2015-ramo-ngorlung": area(28.0669, 95.326, "Pasighat", "East Siang district"),
    "sil-adi-2015-minyong-rayang": area(28.0669, 95.326, "Pasighat", "East Siang district"),
    "sil-adi-2015-padam-siluk": area(28.0669, 95.326, "Pasighat", "East Siang district"),
    "sil-adi-2015-pailibo-irgo": area(28.1661, 94.7642, "Aalo", "West Siang district"),
    "sil-adi-2015-bokar-manigong": area(28.6086, 94.1487, "Mechuka", "Mechuka circle, Shi Yomi"),
    "sil-adi-2015-shimong-mobuk": area(28.6415, 95.0189, "Yingkiong", "Upper Siang district"),
    "sil-adi-2015-milang-village": area(28.6415, 95.0189, "Yingkiong", "Upper Siang district"),
    "Tharu-RNS-Sisana": gn_approx(28.98371, 79.69612, "10827298", "Sisauna, Sitarganj; spelling variant of Sisana"),
    "Tharu-RNS-Sisaikhara": area(28.9291, 79.7002, "Sitarganj", "Sitarganj tehsil"),
    "sil-dhurwa-2021-tiriya": gn(18.91293, 82.19341, "10774229", "Tiria, Bastar"),
    "sil-dhurwa-2021-nethanar": gn(18.87918, 82.05993, "10774196", "Netanar, Bastar"),
    "sil-dhurwa-2021-dharba": gn(18.86983, 81.87016, "1273496", "Darba, Bastar"),
    "sil-dhurwa-2021-kukanar": area(18.3923, 81.6613, "Sukma", "Sukma district"),
    "muduga_chindakki": area(11.1057095, 76.65, "OSM Agali", "Agali (Attappady block HQ), Palakkad"),
    "kurux2011-B-gabindanagar": gn(26.05333, 88.45519, "7487295", "Gobindanagar, Thakurgaon"),
    "kurux2011-D-lohanipara": gn(25.57685, 89.10268, "1196014", "Lohanipara, Badarganj"),
    "kurux2011-E-dulhapur": gn(25.52284, 89.31968, "7890326", "Rameshwarpara, Mithapukur"),
    "kurux2011-C-boldipukur": area(25.5389, 89.2833, "Mithapukur", "Mithapukur upazila, Rangpur"),
    "kurux2011-A-dima": area(26.27, 88.19, "Islampur (Uttar Dinajpur)", "Dima, West Bengal; approximate district area"),
    "kochbd2011-b-nokshi": p(25.19, 90.07, "C", "manual-map", "Koch report figure 4; Nokshi lies on the border in Jhenaigati", "approximate"),
    "kochbd2011-q-uttor-nokshi": p(25.19, 90.07, "C", "manual-map", "Koch report figure 4; Uttor Nokshi adjoins Nokshi", "approximate"),
    "kochbd2011-c-kholchanda": NALITABARI,
    "kochbd2011-l-bharatpur": DURGAPUR,
    "kochbd2011-m-nalchapra": KALMAKANDA,
    "kochbd2011-r-chandabhoi": area(25.1936, 90.2116, "Dalu", "Dalu, West Garo Hills"),
    "angika_omnagar": gn(26.44202, 87.27613, "12096209", "Biratnagar"),
    "angika_darahiya": gn(26.44202, 87.27613, "12096209", "Biratnagar"),
    "angika_pokhariya": gn(26.48057, 87.28293, "7969994", "Pokhariya, Morang"),
    "angika_chhitha": gn(26.5493, 87.2006, "7832377", "Chhitaha, Sunsari"),
    "angika_pandittol": gn(26.469, 87.1933, "7832037", "Amahibelaha, Sunsari"),
    "majhi_manthali": gn(27.40898, 86.05571, "12095795", "Manthali, Ramechhap"),
    "majhi_rajagaun": gn(27.38358, 85.99763, "7981731", "Rajgaun, Ramechhap"),
    "majhi_seleghat": gn(27.35527, 85.98803, "7981615", "Seleghat, Ramechhap"),
    "majhi_sitkha": gn_approx(27.70665, 86.1298, "7945213", "Sitka, Ramechhap; identification approximate"),
    "vedda_dambani": gn(7.5357, 81.1645, "7420127", "Dambana"),
    "vedda_nilgala": gn(7.2236, 81.2869, "8303667", "Nilgala"),
    "vedda_tamankaduwa": gn(7.87737, 80.98691, "11983812", "Thamankaduwa"),
    "vedda_unuwatura_bubula": gn(7.5467, 81.3443, "1225485", "Unuwaturabubula"),
    "vedda_bandaraduwa": gn_approx(7.44775, 81.53314, "11989933", "Bandaradoowa, Ampara; identification approximate"),
    "sil-pahari-pothwari-2010-gho": gn(33.88064, 73.34173, "1178479", "Ghora Gali, Murree"),
    "sil-pahari-pothwari-2010-dew": gn(34.00402, 73.47036, "1407757", "Dewal, Murree"),
    "sil-pahari-pothwari-2010-ayu": gn(34.02872, 73.40741, "10988777", "Ayubia"),
    "sil-pahari-pothwari-2010-koh": gn(34.11615, 73.48771, "1173510", "Kohala"),
    "sil-pahari-pothwari-2010-tha": gn(34.23183, 73.35197, "1163629", "Thandiani"),
    "sil-pahari-pothwari-2010-lor": gn(33.8901, 73.28297, "1171870", "Lora, Abbottabad"),
    "sil-pahari-pothwari-2010-osi": gn(34.01446, 73.48194, "10988780", "Osia, Murree"),
    "sil-pahari-pothwari-2010-muz": gn(34.37002, 73.47082, "1169607", "Muzaffarabad"),
    "sil-pahari-pothwari-2010-dun": gn(34.05474, 73.41209, "10988757", "Dunga Gali"),
    "sil-pahari-pothwari-2010-guj": gn(33.25411, 73.30433, "1177682", "Gujar Khan"),
    "sil-pahari-pothwari-2010-mir": p(33.1478, 73.7519, "B", "manual-map", "Mirpur, Azad Kashmir", "Mirpur city"),
    "sil-pahari-pothwari-2010-bha": p(33.7345, 73.169, "B", "manual-map", "Bhara Kahu, Islamabad", "Bhara Kahu"),
    "sil-pahari-pothwari-2010-mos": area(33.907, 73.3943, "Murree", "Murree tehsil"),
    "sil-pahari-pothwari-2010-nil": area(33.9793, 73.7754, "Bagh", "Bagh district, Azad Kashmir"),
    # --- Zoller 2023 variety labels that are toponyms (regional display points) ----------------------------
    "ghatage-kudali1965": gn(15.86703, 73.67021, "12681320", "Vengurla"),
    "zoller-m-kudali": region(15.86703, 73.67021, "Vengurla", "Kudali; southern Konkan"),
    "zoller-m-kasargod": region(12.52616, 75.12132, "Kasaragod", "Kasargod Marathi"),
    "zoller-m-cochin": region(9.98763, 76.2335, "Kochi", "Cochin Marathi"),
    "zoller-m-berari": region(20.9333, 77.75, "Amravati", "Berar region"),
    "zoller-m-nagpuri": region(21.14631, 79.08491, "Nagpur", "Nagpur region"),
    "ghatage-konkani1963": gn(12.99592, 74.92089, "12022498", "Mangalore"),
    "zoller-ko-southkanara": region(12.99592, 74.92089, "Mangalore", "South Kanara"),
    "zoller-ko-kankon": region(15.00896, 74.13058, "Canacona", "Kankon (Canacona) taluka, Goa"),
    "zoller-k-kishtwari": region(33.52958, 76.01462, "Kishtwar", "Kishtwar"),
    "zoller-wpah-khashi": region(33.52958, 76.01462, "Kishtwar", "Khashi (Kishtwar area)"),
    "zoller-srk-multani": region(30.19679, 71.47824, "Multan", "Multan"),
    "zoller-rj-ajmeri": region(26.2859, 74.8642, "Ajmer", "Ajmer"),
    "zoller-rj-dang": region(26.4986, 77.0192, "Karauli", "Dang tract (Karauli–Sawai Madhopur)"),
    "zoller-garh-tihriyali": region(30.37103, 78.41559, "Tehri", "Tehri Garhwal"),
    "zoller-garh-nagpuriya": region(30.67372, 79.65522, "Joshimath", "Nagpur patti, Chamoli"),
    "zoller-garh-rathi": region(30.06424, 78.73122, "Pauri", "Rath region, Pauri Garhwal"),
    "zoller-garh-bangani": region(31.11534, 78.04361, "Mori", "Bangan, Uttarkashi"),
    "zoller-wpah-bangani": region(31.11534, 78.04361, "Mori", "Bangan, Uttarkashi"),
    "zoller-wpah-bushahari": region(31.44943, 77.63087, "Rampur Bushahr", "Bushahr"),
    "zoller-wpah-shimlasiraji": region(31.44943, 77.63087, "Rampur Bushahr", "Shimla Siraj (Rampur area)"),
    "zoller-wpah-kotguru": region(31.29764, 77.49678, "Kotgarh", "Kotgarh"),
    "zoller-wpah-kotkhai": region(31.13476, 77.56504, "Kotkhai", "Kotkhai"),
    "zoller-wpah-padri": region(33.26, 76.16, "Paddar (Gulabgarh)", "Padar, Kishtwar"),
    "zoller-wpah-outersiraji": region(31.4498, 77.41281, "Ani", "Outer Siraj (Ani)"),
    "zoller-wpah-himachali": region(31.10442, 77.16662, "Shimla", "Himachali; state capital display point"),
    "zoller-kiuth-handuri": region(31.07116, 76.70431, "Nalagarh", "Handur (Nalagarh)"),
    "zoller-kjl-maikoti": gn(28.67437, 82.88019, "1283092", "Maikot, Rukum"),
    "zoller-kjl-nisi": gn_approx(28.4342, 83.2189, "7823936", "Nisi, Baglung"),
    "zoller-sh-gures": region(34.65912, 74.79084, "Gurez", "Gurez valley"),
    "zoller-sh-gultar": region(34.66467, 75.51672, "Gultari", "Gultari"),
    "zoller-mai-duber": region(35.04057, 72.89581, "Duber Bazar", "Duber valley, Kohistan"),
    "zoller-mai-shatoti": region(35.52602, 73.54928, "Shatial", "Shatial, Kohistan"),
    "zoller-pas-darrainur": gn(34.70654, 70.58589, "7053266", "Darah-ye Nur"),
    "zoller-pas-degano": p(34.6458, 70.9008, "B", "manual-cross-reference", "existing Jambu point deg", "Gorayk/Degano"),
    "zoller-ishk-zebaki": region(36.52, 71.34, "Zebak", "Zebak district, Badakhshan"),
    "zoller-bshk-dir": region(35.2058, 71.8756, "Dir", "Dir town"),
    "zoller-l-tinauli": region(33.99783, 72.93493, "Haripur", "Tinauli (Haripur area)"),
    "zoller-sa-singhbhum": region(22.54776, 85.81745, "Chaibasa", "Singhbhum"),
    "zoller-b-chittagong": region(22.4875, 91.96333, "Chittagong", "Chittagong"),
    "zoller-b-rohingya": region(21.03525, 92.3683, "Maungdaw", "Rohingya; Maungdaw, Rakhine"),
    "zoller-oss-digor": region(43.15666, 44.15629, "Digora", "Digor"),
    "zoller-svan-lashkhi": region(42.78888, 42.72228, "Lentekhi", "Lashkhi (Lentekhi)"),
    "zoller-bahnar-pleiku": region(13.98333, 108.0, "Pleiku", "Pleiku"),
    "zoller-centralnicobarese-nancowry": region(7.9484, 93.39469, "Nancowry", "Nancowry"),
    "zoller-romvlax-burgenland": region(47.28971, 16.20595, "Oberwart", "Burgenland"),
    "zoller-prk-purik": region(34.55765, 76.12622, "Kargil", "Purik (Kargil)"),
    "zoller-kabardian-baslen": region(44.83182, 41.38647, "Uspenskoye", "Baslen (Besleney), Uspensky district"),
    "zoller-khmu-yuan": region(20.9486, 101.40188, "Luang Namtha", "Khmu Yuan"),
    "zoller-kryz-kryts": region(41.20, 48.30, "Qriz village, Quba", "Kryts; approximate"),
    "zoller-rombalk-greece": p(39.07, 21.82, "C", "manual-centroid", "country centroid", "Greece"),
    "zoller-gy-sweden": p(62.0, 15.0, "C", "manual-centroid", "country centroid", "Sweden"),
    "zoller-gy-norway": p(61.0, 9.0, "C", "manual-centroid", "country centroid", "Norway"),
    "zoller-gy-finland": p(64.0, 26.0, "C", "manual-centroid", "country centroid", "Finland"),
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_atomic(path: Path, fields: list[str], rows: list[dict[str, str]], lineterminator: str = "\n") -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator=lineterminator, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(check: bool) -> None:
    fields, rows = read_csv(DIALECTS)
    by_id = {row["ID"]: row for row in rows}
    missing = sorted(set(DECISIONS) - set(by_id))
    if missing:
        raise SystemExit(f"unknown dialect IDs: {missing}")

    changed = 0
    for ident, (lat, lon, quality, _method, _source, note) in DECISIONS.items():
        row = by_id[ident]
        if (row["Latitude"], row["Longitude"], row["Quality"]) == (lat, lon, quality):
            continue
        changed += 1
        if check:
            print(f"{ident:<42} {row['Latitude']:>11},{row['Longitude']:<11} -> {lat:>11},{lon:<11} {quality}  {note[:60]}")
        row["Latitude"], row["Longitude"], row["Quality"] = lat, lon, quality
    print(f"{changed} dialect rows change")
    if check:
        return

    write_atomic(DIALECTS, fields, rows, lineterminator="\r\n")  # the registry is committed with CRLF rows
    dfields, drows = read_csv(DECISIONS_TABLE)
    table = {row["ID"]: row for row in drows}
    for ident, (lat, lon, _quality, method, source, note) in DECISIONS.items():
        table[ident] = {"ID": ident, "Latitude": lat, "Longitude": lon, "Method": method, "Source": source, "Note": note}
    write_atomic(DECISIONS_TABLE, dfields, list(table.values()))
    print(f"wrote {DIALECTS} and {len(table)} rows in {DECISIONS_TABLE}")


if __name__ == "__main__":
    main(check="--check" in sys.argv[1:])
