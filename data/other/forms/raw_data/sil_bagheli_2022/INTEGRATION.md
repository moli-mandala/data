# Shared integration proposal

Shared bibliography, dialect registry, profile routing, audit-registry, census,
and focused metadata integration were applied by the coordinating task on
2026-08-28. The consolidated CLDF build and browser QA remain deferred until
all parallel source packages have landed. The sections below retain the source-
local proposal and expected counts as an audit trail.

## Bibliography

Add to `cldf/sources.bib`:

```bibtex
@techreport{koshy2022bagheli,
  author = {Koshy, Binoy},
  title = {A Sociolinguistic Study of Bagheli Speakers in Madhya Pradesh},
  year = {2022},
  number = {2022-015},
  institution = {SIL International},
  address = {Dallas},
  series = {Journal of Language Survey Reports},
  url = {https://www.sil.org/resources/archives/94596},
  note = {Report created in 2004 and published in 2022; PDF SHA-256 d1424f317dc12fe01d99d33abd917201575487f4de44529678ecce1c282a4627}
}
```

No new `cldf/languages.csv` row is required. All 18 target lects attach to the
existing canonical row `bagheli_lakshman,Bagheli,bagh1251,...,E. Hindi`.

## Dialects

Add these exact rows to `cldf/dialects.csv`. The report names localities but
does not give defensible historical collection coordinates, so latitude and
longitude are blank and quality is C.

```csv
sil-bagheli-2022-dabhaura,dialect:bagheli_lakshman:sil-bagheli-2022-dabhaura:Dabhaura,bagheli_lakshman,sil-bagheli-2022-dabhaura,Dabhaura,bagh1251,,,E. Hindi,"Dabhaura, Theothar tahsil, Rewa district, Madhya Pradesh; source code D",C
sil-bagheli-2022-katkon,dialect:bagheli_lakshman:sil-bagheli-2022-katkon:Katkon%20%28Khadi-Hindi%29,bagheli_lakshman,sil-bagheli-2022-katkon,Katkon (Khadi-Hindi),bagh1251,,,E. Hindi,"Katkon, Nagod tahsil, Satna district, Madhya Pradesh; source label Khadi-Hindi; code K",C
sil-bagheli-2022-amarkantak,dialect:bagheli_lakshman:sil-bagheli-2022-amarkantak:Amarkantak%20%28Pindra-Zamindari%29,bagheli_lakshman,sil-bagheli-2022-amarkantak,Amarkantak (Pindra-Zamindari),bagh1251,,,E. Hindi,"Amarkantak, Pushparajgarh tahsil, Anuppur district, Madhya Pradesh; source label Pindra-Zamindari; code P",C
sil-bagheli-2022-sunwari,dialect:bagheli_lakshman:sil-bagheli-2022-sunwari:Sunwari,bagheli_lakshman,sil-bagheli-2022-sunwari,Sunwari,bagh1251,,,E. Hindi,"Sunwari, Maihar tahsil, Satna district, Madhya Pradesh; source code S",C
sil-bagheli-2022-karchana,dialect:bagheli_lakshman:sil-bagheli-2022-karchana:Karchana%20%28Allahabadi%29,bagheli_lakshman,sil-bagheli-2022-karchana,Karchana (Allahabadi),bagh1251,,,E. Hindi,"Karchana, Allahabad district, Uttar Pradesh; source label Allahabadi; code a",C
sil-bagheli-2022-baikanthpur,dialect:bagheli_lakshman:sil-bagheli-2022-baikanthpur:Baikanthpur,bagheli_lakshman,sil-bagheli-2022-baikanthpur,Baikanthpur,bagh1251,,,E. Hindi,"Baikanthpur, Sirmour tahsil, Rewa district, Madhya Pradesh; source code b",C
sil-bagheli-2022-chawari,dialect:bagheli_lakshman:sil-bagheli-2022-chawari:Chawari,bagheli_lakshman,sil-bagheli-2022-chawari,Chawari,bagh1251,,,E. Hindi,"Chawari, Sidhi tahsil and district, Madhya Pradesh; source code c",C
sil-bagheli-2022-dewara,dialect:bagheli_lakshman:sil-bagheli-2022-dewara:Dewara,bagheli_lakshman,sil-bagheli-2022-dewara,Dewara,bagh1251,,,E. Hindi,"Dewara, Hanumana tahsil, Rewa district, Madhya Pradesh; source code d",C
sil-bagheli-2022-domahai,dialect:bagheli_lakshman:sil-bagheli-2022-domahai:Domahai,bagheli_lakshman,sil-bagheli-2022-domahai,Domahai,bagh1251,,,E. Hindi,"Domahai, Majgama tahsil, Satna district, Madhya Pradesh; source code e",C
sil-bagheli-2022-janakpur,dialect:bagheli_lakshman:sil-bagheli-2022-janakpur:Janakpur%20%28Bakhari%20Boli%29,bagheli_lakshman,sil-bagheli-2022-janakpur,Janakpur (Bakhari Boli),bagh1251,,,E. Hindi,"Janakpur, Bharatpur tahsil, Koriya district, Chhattisgarh; source label Bakhari Boli; code j",C
sil-bagheli-2022-keoti,dialect:bagheli_lakshman:sil-bagheli-2022-keoti:Keoti,bagheli_lakshman,sil-bagheli-2022-keoti,Keoti,bagh1251,,,E. Hindi,"Keoti, Sirmour tahsil, Rewa district, Madhya Pradesh; source code k",C
sil-bagheli-2022-lodha,dialect:bagheli_lakshman:sil-bagheli-2022-lodha:Lodha%20%28Rimahi%20Bagheli%29,bagheli_lakshman,sil-bagheli-2022-lodha,Lodha (Rimahi Bagheli),bagh1251,,,E. Hindi,"Lodha, Umaria tahsil and district, Madhya Pradesh; source label Rimahi Bagheli; code l",C
sil-bagheli-2022-kotasiv-prathapsing,dialect:bagheli_lakshman:sil-bagheli-2022-kotasiv-prathapsing:Kotasiv%20Prathapsing%20%28Mirzapuri%29,bagheli_lakshman,sil-bagheli-2022-kotasiv-prathapsing,Kotasiv Prathapsing (Mirzapuri),bagh1251,,,E. Hindi,"Kotasiv Prathapsing, Lalganj tahsil, Mirzapur district, Uttar Pradesh; source label Mirzapuri; code m",C
sil-bagheli-2022-singpur,dialect:bagheli_lakshman:sil-bagheli-2022-singpur:Singpur%20%28Sohagpuri%29,bagheli_lakshman,sil-bagheli-2022-singpur,Singpur (Sohagpuri),bagh1251,,,E. Hindi,"Singpur, Sohagpur tahsil, Shahdol district, Madhya Pradesh; source label Sohagpuri; code n",C
sil-bagheli-2022-parasawar,dialect:bagheli_lakshman:sil-bagheli-2022-parasawar:Parasawar,bagheli_lakshman,sil-bagheli-2022-parasawar,Parasawar,bagh1251,,,E. Hindi,"Parasawar, Devsar tahsil, Sidhi district, Madhya Pradesh; source code p",C
sil-bagheli-2022-semara,dialect:bagheli_lakshman:sil-bagheli-2022-semara:Semara,bagheli_lakshman,sil-bagheli-2022-semara,Semara,bagh1251,,,E. Hindi,"Semara, Jaisingh-Nagar tahsil, Shahdol district, Madhya Pradesh; source code r",C
sil-bagheli-2022-silpari,dialect:bagheli_lakshman:sil-bagheli-2022-silpari:Silpari,bagheli_lakshman,sil-bagheli-2022-silpari,Silpari,bagh1251,,,E. Hindi,"Silpari, Rewa tahsil and district, Madhya Pradesh; source code s",C
sil-bagheli-2022-mahdeiya,dialect:bagheli_lakshman:sil-bagheli-2022-mahdeiya:Mahdeiya%20%28Singraulihi%29,bagheli_lakshman,sil-bagheli-2022-mahdeiya,Mahdeiya (Singraulihi),bagh1251,,,E. Hindi,"Mahdeiya, Singrauli tahsil, Sidhi district, Madhya Pradesh; source also prints Thurua/Dabhaura inconsistently; code t",C
```

## Profile and routing

- Add `sil-bagheli` to the profile-name allowlist in `make_cldf.py`.
- Route citation key `koshy2022bagheli` to
  `conversion/sil-bagheli.txt` and enable conversion in the source-key routing
  block of `make_cldf.py` (and any parallel route table in `utils.py`).
- The profile has complete coverage of all 5,828 installed forms. Preserve the
  manually checked source IPA as `Original`; use the converted value only for
  Jambu display form.
- No etymology, cognacy, borrowing, derivation, or graph edges are claimed.

## Commands and expected results

```sh
python data/other/forms/raw_data/sil_bagheli_2022/import_bagheli.py --install
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest --with segments pytest -q tests/test_sil_bagheli_2022.py
make all
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected source-local results: 2,284 manually transcribed lexical response
lines, 6,111 expanded lexical response occurrences, 3,990 manually reviewed
conceptual cells (3,933 lexically attested, 10 `by name` only, and 47 genuinely
blank), 24 excluded non-lexical `by name` occurrences, 5,828 installed Bagheli
source rows, 5,829 compiled Bagheli forms after item 173 site `m` `je,e` is
expanded into its two printed alternatives, 283 excluded Hindi-control
occurrences, two excluded unassigned lines, and 6,184 total audit rows. The
shared build, full suite, and browser QA are intentionally deferred here.
