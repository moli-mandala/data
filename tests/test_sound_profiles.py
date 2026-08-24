import csv
import glob
import importlib.util
import io
import os
import sys
import unicodedata
from pathlib import Path

from segments.tokenizer import Tokenizer

DATA_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(DATA_DIR))

from utils import mapping
import make_cldf


STRAND_SCRIPT = DATA_DIR / "data/other/forms/raw_data/strand.py"
STRAND_SPEC = importlib.util.spec_from_file_location("strand_profile_source", STRAND_SCRIPT)
assert STRAND_SPEC and STRAND_SPEC.loader
strand_source = importlib.util.module_from_spec(STRAND_SPEC)
STRAND_SPEC.loader.exec_module(strand_source)


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
    "emeneau-brahui": ["20260819-emeneau-brahui-1997.csv"],
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
    "nured": ["20260818-nured-org.csv"],
    "buddruss-grangali": ["20260819-buddruss-grangali.csv"],
    "merriam-reconstruction": ["20260718-merriam-dravidian-db.csv"],
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
    assert convert("merriam-reconstruction", "kaṭ-/kaḍ-") == "kaṭ-/kaḍ-"
    assert convert("merriam-reconstruction", "agáḍ-") == "agáḍ-"
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
    assert convert("emeneau-brahui", "bēg̲h̲-") == "bēɣ-"
    assert convert("emeneau-brahui", "hōg̲h̲-") == "hōɣ-"
    assert convert("emeneau-brahui", "taṛifing") == "taṛifing"
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
    assert convert("nured", "ačẽ́") == "ačẽ́"
    assert convert("nured", "puṇḍrë́/-í") == "puṇḍrë́/-í"
    assert convert("buddruss-grangali", "cōr") == "ʦōr"
    assert convert("buddruss-grangali", "čar") == "car"
    assert convert("buddruss-grangali", "ãc̣") == "ãʦ̣"
    assert convert("buddruss-grangali", "kaširə") == "kaśirə"
    assert convert("buddruss-grangali", "goā́t") == "goā́t"
    assert convert("buddruss-grangali", "naṅacə́") == "naŋaʦə́"
    assert convert("buddruss-grangali", "brəṣpā̃re") == "brəṣpā̃re"
    assert convert("magar-2024", "sṳm") == "sṳm"
    assert convert("pyangaun-newar", "ṳ") == "ṳ"
    assert convert("tagin-puroik", "ʃĕandəkhau") == "śeandəkʰau"
    assert convert("rajasthani", "ʂɐgɭaji") == "ṣagḷāyī"
    assert convert("markodi", "nakʰːam") == "nakkʰam"
    assert convert("strand", "uː") == "ū"


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


def test_every_build_routed_form_converts_without_replacement(monkeypatch):
    """Exercise the exact routing and preprocessing used by the complete CLDF build."""
    monkeypatch.setattr(make_cldf, "tqdm", lambda iterable, **_kwargs: iterable)
    files = [
        DATA_DIR / "data/cdial/cdial.csv",
        DATA_DIR / "data/munda/forms.csv",
        DATA_DIR / "data/dedr/dedr_new.csv",
        DATA_DIR / "data/dedr/pdr.csv",
        *sorted((DATA_DIR / "data/other/forms").glob("*.csv")),
        DATA_DIR / "data/dbia/forms.csv",
    ]
    param_counter = {}
    converted = 0
    for file_number, path in enumerate(files):
        errors = io.StringIO()
        _, stats = make_cldf.parse_file(
            str(path.relative_to(DATA_DIR)),
            errors,
            file_num=file_number,
            param_counter=param_counter,
        )
        assert not errors.getvalue(), (path.name, errors.getvalue().splitlines()[:10])
        converted += stats["for_conversion"]

    # Guard against a path-selection mistake turning this into a vacuous test.
    assert converted > 500_000


def test_every_build_routed_parameter_converts_without_replacement():
    checked = 0
    with Path("data/cdial/params.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            headword = (
                row[1]
                .replace("ˊ", "́")
                .replace("`", "̀")
                .replace(" --", "-")
                .replace("-- ", "-")
                .strip(".,;-: ")
                .replace("<? >", "")
                .lower()
                .replace("˜", "̃")
                .split(",", 1)[0]
                .strip()
            )
            if " " in headword or "˚" in headword:
                continue
            result = make_cldf.convertors["cdial"](
                headword.strip("-123456,;"), column="IPA"
            )
            assert "�" not in result, ("cdial", row[0], headword, result)
            checked += 1

    aliases = {"extensions_ia": "cdial", "strand3": "strand"}
    for path in sorted(Path("data/other/params").glob("*.csv")):
        raw_name = path.stem
        convertible = raw_name in make_cldf.convertors or raw_name in aliases
        profile = aliases.get(raw_name, raw_name)
        if not convertible:
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                value = row[2]
                if profile == "strand":
                    if row[1] in {"PNur", "PA"}:
                        value = "*" + value
                    value = value.replace("′", "ʹ").replace("-", "")
                result = make_cldf.convertors[profile](
                    value.strip("-123456,;"), column="IPA"
                )
                assert "�" not in result, (path.name, row[0], value, result)
                checked += 1

    assert checked > 17_000


def test_strand_phonemic_profile_covers_every_legacy_form():
    tokenizer = Tokenizer("data/other/forms/ipa/strand.txt")
    for filename in ("20220913-strand.csv", "20220913-strand2.csv"):
        path = Path("data/other/forms") / filename
        with path.open(encoding="utf-8", newline="") as stream:
            for row_number, row in enumerate(csv.reader(stream), 1):
                source = strand_source.normalize_legacy_stress(row[2])
                result = tokenizer(source, column="IPA")
                assert "�" not in result, (filename, row_number, source, result)
                assert "�" not in row[5], (filename, row_number, row[5])


def test_sound_profiles_have_unique_graphemes():
    for path in sorted(Path("conversion").glob("*.txt")):
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream, delimiter="\t"))
        graphemes = [row[0] for row in rows[1:] if row]
        assert len(graphemes) == len(set(graphemes)), path.name


def test_installed_form_inputs_have_no_replacement_characters():
    paths = [
        Path("data/cdial/cdial.csv"),
        Path("data/munda/forms.csv"),
        Path("data/dedr/dedr_new.csv"),
        Path("data/dedr/pdr.csv"),
        *sorted(Path("data/other/forms").glob("*.csv")),
        Path("data/dbia/forms.csv"),
    ]
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            for row_number, row in enumerate(csv.reader(stream), 1):
                assert all("�" not in value for value in row), (path.name, row_number, row)


def test_legacy_survey_placeholders_and_ocr_intrusions_are_not_forms():
    with Path("data/other/forms/20230521-rajasthani.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rajasthani = list(csv.reader(stream))
    assert all(row[2].casefold() != "no entry" for row in rajasthani)
    assert all(not (row[2].startswith("(") and row[2].endswith(")")) for row in rajasthani)
    small_axe = next(
        row for row in rajasthani
        if row[0] == "mewati_akera" and row[2] == "tʃãʈja" and row[3] == "axe"
    )
    assert (small_axe[2], small_axe[6]) == ("tʃãʈja", "(small)")

    with Path("data/other/forms/20230517-chattisgarhi.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        chattisgarhi = list(csv.reader(stream))
    assert any(row[2] == "ɐnɐ̆̃" for row in chattisgarhi)
    assert all("another" not in row[2] for row in chattisgarhi)


def test_lsi_profile_covers_every_upstream_phonemic_form():
    tokenizer = Tokenizer("conversion/lsi.txt")
    path = Path("data/other/forms/20260813-grierson-lsi.csv")
    with path.open(encoding="utf-8", newline="") as stream:
        for row_number, row in enumerate(csv.reader(stream), 1):
            source = unicodedata.normalize("NFC", row[5])
            result = tokenizer(source, column="IPA")
            assert "�" not in result, (row_number, source, result)
