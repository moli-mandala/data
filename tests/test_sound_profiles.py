import csv
import glob
import os
import sys
import unicodedata
from pathlib import Path

from segments.tokenizer import Tokenizer

DATA_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(DATA_DIR))

from utils import mapping


# These imports already supply a canonical CLTS/IPA column alongside the source
# transcription, so routing them through a Jambu source-orthography profile
# would discard upstream analysis rather than add normalization.
EXCLUDED_FILES = {
    "20260813-tagin-puroik.csv",
}

CHECKED_PROFILE_FILES = {
    "house": [
        "20220913-dhivehi.csv", "20220913-gawri.csv", "20220913-kalkoti.csv",
        "20220913-khetrani.csv", "20220913-kholosi.csv", "20220913-konkani.csv",
        "20220913-kundalshahi.csv", "20220913-kvari.csv", "20220913-zadjali.csv",
        "20230403-arora.csv", "20230519-punjabi.csv", "20230524-sindhic.csv",
        "20230705-pashai.csv", "20260726-paranavitana-sigiri.csv",
        "20260810-tulpule-old-marathi.csv", "20260813-wolf-kota.csv",
    ],
    "drasi": ["20260725-drasi.csv"],
    "yoshioka": ["20260726-yoshioka-eastern-burushaski.csv"],
    "gandhari": ["20260805-gandhari-org.csv"],
    "kullui": ["20260813-kullui-org.csv"],
    "toda": ["20260813-bhaskararao-toda.csv"],
    "rabha": ["20260813-rabha.csv"],
    "yamphu": ["20260813-yamphu.csv"],
    "eastern-magar": ["20260813-eastern-magar.csv"],
    "western-tamang": ["20260813-western-tamang.csv"],
    "humla": ["20260813-humla.csv"],
    "gurung": ["20260813-gurung.csv"],
    "dotyali": ["20260813-dotyali.csv"],
    "kudiya": ["20260813-kudiya.csv"],
    "brahui": ["20260813-ali-kobayashi-brahui.csv"],
    "southworth-marathi": ["20260818-southworth-marathi.csv"],
    "hajong-survey": ["20260813-hajong.csv"],
    "santali-cluster": ["20260813-santali-cluster.csv"],
    "sampang": ["20260813-sampang.csv"],
    "mewahang": ["20260813-mewahang.csv"],
    "chhulung": ["20260813-chhulung.csv"],
    "magahi-survey": ["20260813-magahi.csv"],
    "ghatage": ["20260817-ghatage-marati-kasargod.csv"],
    "vaagri": ["20220913-vaagri.csv"],
    "nihali": [
        "20260817-mundlay-nihali.csv",
        "20260817-nagaraja-nihali-wiktionary.csv",
        "20260817-nihali-database-bhattacharya.csv",
        "20260817-nihali-database-konow.csv",
    ],
    "badaga-hockings": ["20260818-hockings-badaga.csv"],
}


def convert(profile: str, value: str) -> str:
    tokenizer = Tokenizer(f"conversion/{profile}.txt")
    return unicodedata.normalize(
        "NFC",
        tokenizer(unicodedata.normalize("NFC", value), column="IPA")
        .replace(" ", "")
        .replace("#", " "),
    )


def test_every_installed_source_has_an_explicit_sound_profile():
    profiles = {
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob("conversion/*.txt")
    }
    for path in sorted(glob.glob("data/other/forms/*.csv")):
        basename = os.path.basename(path)
        if basename in EXCLUDED_FILES:
            continue
        with open(path, encoding="utf-8", newline="") as stream:
            first_row = next(csv.reader(stream), None)
        if first_row is None:
            # The historical Schmidt placeholder is empty, but its later replacement uses the
            # same filename key and is explicitly registered below.
            assert mapping["schmidt"] == "schmidt-kashmiri"
            continue
        key = os.path.splitext(basename)[0].split("-")[1]
        if first_row[7].split("[", 1)[0] in {"shackle", "shackle-auto"}:
            profile = "cdial"
        elif first_row[7].split("[", 1)[0] == "liljegren-hindukush":
            profile = "liljegren-hindukush"
        else:
            profile = mapping.get(key, key)
        assert profile in profiles, f"{basename} has no sound profile"


def test_preservation_profile_repairs_only_known_legacy_notation():
    assert convert("house", "iⁿdо̄li") == "iⁿdōli"
    assert convert("house", "ic̣ī") == "iʦ̣ī"
    assert convert("house", "nat̪he") == "nathe"
    assert convert("house", "-abi (obl.)") == "-abi (obl.)"


def test_new_source_profiles_cover_source_specific_transcription():
    assert convert("ghatage", "tã:ŋkɨ") == "tā̃ŋkɨ"
    assert convert("ghatage", "pɛ:ṇṭɛ") == "pɛ̄ṇṭɛ"
    assert convert("ghatage", "goṭṭe") == "goṭṭe"
    assert convert("ghatage", "ǰagrutɛ") == "jagrutɛ"
    assert convert("southworth-marathi", "phaḷ") == "pʰaḷ"
    assert convert("southworth-marathi", "āi") == "āī"
    assert convert("southworth-marathi", "māṇḍi") == "māṇḍī"
    assert convert("southworth-marathi", "niṭ") == "nīṭ"
    assert convert("southworth-marathi", "bāḷant(iṇ)") == "bāḷant(īṇ)"
    assert convert("southworth-marathi", "ḍokə") == "ḍokə̄"
    assert convert("southworth-marathi", "buṭṭ@") == "buṭṭ@"
    assert convert("ghatage", "tilače te:lɨ") == "tilace tēlɨ"
    assert convert("vaagri", "iga:ri") == "igāri"
    assert convert("vaagri", "iJalbiJal") == "iẓalbiẓal"
    assert convert("vaagri", "uba:Sa") == "ubāśa"
    assert convert("vaagri", "phu:k") == "pʰūk"
    assert convert("vaagri", "be:Ra:d#") == "bēʀādɨ"
    assert convert("lsi", "tʃʰiː") == "cʰī"
    assert convert("lsi", "pʰʌn̪tʃʰ") == "pʰancʰ"
    assert convert("lsi", "prʌː¹²") == "prā¹²"
    assert convert("nihali", "caːgo") == "cāgo"
    assert convert("nihali", "ãːpo") == "ā̃po"
    assert convert("nihali", "dhāblā") == "dʰāblā"
    assert convert("nihali", "aɖɖo") == "aḍḍo"
    assert convert("nihali", "aᵑgarako") == "aⁿgarako"
    assert convert("nihali", "chhirī") == "cʰirī"
    assert convert("nihali", "ʈoːl") == "ṭōl"
    assert convert("drasi", "ó:ʃ") == "ṓś"
    assert convert("drasi", "ʧhúp") == "cʰúp"
    assert convert("yoshioka", "aabáad") == "ābā̂d"
    assert convert("yoshioka", "aaqhér") == "āqʰér"
    assert convert("gandhari", "aṭ́hi") == "aṭṭʰi"
    assert convert("gandhari", "maj̄a") == "majja"
    assert convert("gandhari", "kiṣ̄a") == "kiṣṇa"
    assert convert("kullui", "dzʰaɽna") == "ʣʰaṛna"
    assert convert("kullui", "rɔng") == "rɔŋg"
    assert convert("toda", "aḏïyi ïḏ") == "aḏɨyi ɨḏ"
    assert convert("toda", "teːsts̱") == "tēsts̱"
    assert convert("rabha", "kɑ́n") == "kā́n"
    assert convert("rabha", "kɑnɡɑnd͡ʒi") == "kāngānjī"
    assert convert("rabha", "tʃɑ̑skɑm") == "cā̑skām"
    assert convert("yamphu", "dʒʌɾa") == "jara"
    assert convert("yamphu", "tsʌŋak̚") == "ʦaŋak̚"
    assert convert("sampang", "tˢʰʌ̃wara") == "ʦʰãvara"
    assert convert("sampang", "pʌmtᶳʱu") == "pamcʰu"
    assert convert("sampang", "dᶽʰara") == "jʰara"
    assert convert("mewahang", "tˢʰebruŋwa") == "ʦʰebruŋva"
    assert convert("mewahang", "mimtᶳʰa") == "mimcʰa"
    assert convert("mewahang", "pɨ:ʔma") == "pɨ̄ʔma"
    assert convert("chhulung", "dzʰarak") == "ʣʰarak"
    assert convert("chhulung", "ŋa?lasi") == "ŋaʔlasi"
    assert convert("chhulung", "hərd̪i") == "hərdi"
    assert convert("magahi-survey", "kəpar") == "kapār"
    assert convert("magahi-survey", "jʰãɽa") == "jʰā̃ṛā"
    assert convert("magahi-survey", "pʰut̺əl") == "pʰūtal"
    assert convert("eastern-magar", "midʒaŋ") == "mijaŋ"
    assert convert("eastern-magar", "tuk̚tʃʲo") == "tuk̚cʸo"
    assert convert("western-tamang", "dʑiu") == "ʣ̣iu"
    assert convert("western-tamang", "tɕⁱam") == "ʦ̣ⁱam"
    assert convert("western-tamang", "ʔa:tɕabel") == "ʔāʦ̣abel"
    assert convert("humla", "tsʲʰerwa") == "ʦʸʰerva"
    assert convert("humla", "tɻ̥a") == "tr̥a"
    assert convert("gurung", "mõɾaa") == "mõrā"
    assert convert("gurung", "tʃʰʲaa") == "cʰʸā"
    assert convert("dotyali", "tʃʰɑti") == "cʰāti"
    assert convert("dotyali", "kɑ̃ɖɑ̃") == "kā̃ḍā̃"
    assert convert("kudiya", "tʃaɭu") == "caḷū"
    assert convert("kudiya", "maᶚe") == "maṛ̆e"
    assert convert("brahui", "zunḍ-ing") == "zunḍ-ing"
    assert convert("brahui", "ḍʰāḍarī") == "ḍʰāḍarī"
    assert convert("badaga-hockings", "Eḍeka:ḍu") == "Eḍekāḍu"
    assert convert("badaga-hockings", "ka:ḷu") == "kāḷu"


def test_new_profiles_cover_every_installed_source_form():
    root = Path("data/other/forms")
    for profile, filenames in CHECKED_PROFILE_FILES.items():
        tokenizer = Tokenizer(f"conversion/{profile}.txt")
        for filename in filenames:
            with (root / filename).open(encoding="utf-8", newline="") as stream:
                for row_number, row in enumerate(csv.reader(stream), 1):
                    source = unicodedata.normalize("NFC", row[2])
                    result = tokenizer(source, column="IPA")
                    assert ("�" in result) == ("�" in source), (
                        filename,
                        row_number,
                        source,
                        result,
                    )


def test_lsi_profile_covers_every_upstream_phonemic_form():
    tokenizer = Tokenizer("conversion/lsi.txt")
    path = Path("data/other/forms/20260813-grierson-lsi.csv")
    with path.open(encoding="utf-8", newline="") as stream:
        for row_number, row in enumerate(csv.reader(stream), 1):
            source = unicodedata.normalize("NFC", row[5])
            result = tokenizer(source, column="IPA")
            assert "�" not in result, (row_number, source, result)
