# Adi 2015 shared source-specific integration record

The source-local package is complete: 2,763 manually reviewed conceptual cells
yield 2,770 lossless response rows and an exhaustive 2,763-row audit. The exact
rows and source-specific bibliography, language, dialect and profile routes
are installed. The consolidated CLDF/full build, global audit, browser
database and commit remain deferred.

The independent evidence was frozen before shared integration in
`preintegration_manifest.json`: PDF SHA-256
`8e1500383a02445252a3eb6973a1b011fabea71eb25ad79fc43ba5b78bd1135c`,
manual-bundle SHA-256
`a9a1aac22c77c4cf66230c2fa014a7b151cd676db44810744b06748070bd92f0`,
staged/installed SHA-256
`edb29a8f65fea0600e3d54bfcf2adef81fd833c47b619de5cd701bd61df4031c`,
and 22-page render-tree SHA-256
`0746c68daf48349570eb0d37e2d69afb79c22571d69a410484b8437e1efd794c`.

## Installed bibliography

```bibtex
@techreport{padung-sako2015adi,
  author = {Padung, Tutum and Sako, Kara},
  title = {A Brief Survey among the Adi of Arunachal Pradesh},
  institution = {SIL International},
  type = {SIL Electronic Survey Report},
  number = {2015-016},
  year = {2015},
  pages = {1--41},
  url = {https://www.sil.org/resources/archives/69459},
  included = {Appendix B, printed pages 13--34: all 307 prompts across nine Adi-area elicitation lists},
  provenance = {data/other/forms/raw_data/sil_adi_2015/staged_forms.csv; data/other/forms/raw_data/sil_adi_2015/staged_audit.tsv},
  jambu_editor = {Aryaman Arora and OpenAI Codex},
  ocr = {No}
}
```

Form citations are already staged as
`padung-sako2015adi[Appendix B, printed p. N, item I, list CODE]`.
The report's numbered similarity groups are preserved in `staged_audit.tsv`
but deliberately do not become Jambu cognate sets.

## Installed base-language rows

These four language-level Glottolog mappings reflect Glottolog 5.3. They are
not all one language merely because the report groups them under “Adi.”

```csv
MisingPadamMiriMinyong,Mising-Padam-Miri-Minyong,misi1242,27.418,94.69,Other,"Arunachal Pradesh and Assam; parent for the Minyong and Padam lists",C
BoriKarko,Bori-Karko,bori1243,27.62812,94.3538,Other,"Arunachal Pradesh; parent for the Bori and Shimong lists",C
BokarRamo,Bokar-Ramo,boka1249,28.61433,94.066429,Other,"Arunachal Pradesh and adjacent China; parent for the Ramo, Pailibo, Ashing and Bokar lists",C
Milang,Milang,mila1245,28.5539908,95.186127,Other,"Upper Siang district, Arunachal Pradesh; parent for the Milang-village list",C
```

The coordinates above are current Glottolog representative language points,
not 2004 survey-site coordinates, and therefore remain quality C.

## Installed dialect/site rows

The report identifies an elicitation village and consultant variety for every
list. It supplies no coordinates, so the site coordinates remain blank rather
than inheriting a modern language centroid.

```csv
sil-adi-2015-minyong-rayang,dialect:MisingPadamMiriMinyong:sil-adi-2015-minyong-rayang:Rayang%20%28Minyong%29,MisingPadamMiriMinyong,MN,Rayang (Minyong),miny1239,,,Other,"Rayang village, East Siang district, Arunachal Pradesh; Minyong consultant",C
sil-adi-2015-bori-bogu-payum,dialect:BoriKarko:sil-adi-2015-bori-bogu-payum:Bogu%2FPayum%20Circle%20%28Bori%29,BoriKarko,BR,Bogu/Payum Circle (Bori),bori1245,,,Other,"Bogu/Payum Circle, West Siang district, Arunachal Pradesh; Bori consultant",C
sil-adi-2015-ramo-ngorlung,dialect:BokarRamo:sil-adi-2015-ramo-ngorlung:Ngorlung%20%28Ramo%29,BokarRamo,RM,Ngorlung (Ramo),ramo1243,,,Other,"Ngorlung village, East Siang district, Arunachal Pradesh; Ramo consultant",C
sil-adi-2015-milang-village,dialect:Milang:sil-adi-2015-milang-village:Milang%20village%20%28Milang%29,Milang,ML,Milang village (Milang),,,,Other,"Milang village, Upper Siang district, Arunachal Pradesh; Milang consultant",C
sil-adi-2015-pailibo-irgo,dialect:BokarRamo:sil-adi-2015-pailibo-irgo:Irgo%20%28Pailibo%29,BokarRamo,PL,Irgo (Pailibo),pail1243,,,Other,"Irgo village, West Siang district, Arunachal Pradesh; Pailibo consultant",C
sil-adi-2015-ashing-ningging,dialect:BokarRamo:sil-adi-2015-ashing-ningging:Ningging%20%28Ashing%2FBogum%20Bokang%29,BokarRamo,AS,Ningging (Ashing/Bogum Bokang),ashi1243,,,Other,"Ningging village, Upper Siang district, Arunachal Pradesh; Bogum Bokang (Ashing) consultant",C
sil-adi-2015-padam-siluk,dialect:MisingPadamMiriMinyong:sil-adi-2015-padam-siluk:Siluk%20%28Padam%29,MisingPadamMiriMinyong,PD,Siluk (Padam),pada1257,,,Other,"Siluk village, East Siang district, Arunachal Pradesh; Padam consultant",C
sil-adi-2015-shimong-mobuk,dialect:BoriKarko:sil-adi-2015-shimong-mobuk:Mobuk%20%28Shimong%29,BoriKarko,SM,Mobuk (Shimong),shim1250,,,Other,"Mobuk village, Upper Siang district, Arunachal Pradesh; Shimong consultant",C
sil-adi-2015-bokar-manigong,dialect:BokarRamo:sil-adi-2015-bokar-manigong:Manigong%20%28Bokar%29,BokarRamo,BK,Manigong (Bokar),,,,Other,"Manigong village, West Siang district, Arunachal Pradesh; Bokar consultant",C
```

## Profile and routing

- Preserve the source string as `Original` and use the reviewed
  `symbol_inventory.tsv` as the coverage baseline.
- Start from the lossless decisions in `PRESERVATION_PROFILE.md`. Any shared
  display mapping of `tʃ`, `dʒ`, dental marks, or other IPA sequences must be
  explicit, sequence-aware, and tested; source punctuation, spaces, length,
  nasalization, and question marks must survive.
- Route only bibliography key `padung-sako2015adi` through the eventual shared
  profile. Do not use a filename-wide heuristic.
- Keep all nine lists as targets. Exclude the 93 explicit `no entry` cells;
  there are no control lists or unresolved readings.
- The profile inventory contains exactly 42 source symbols and zero
  replacement characters. The literal printed question marks in `huupe?`
  (item 292/RM) and `kapə?` (item 293/AS) are reviewed source punctuation,
  not unresolved transcription, and remain visible.

## Validation and remaining shared work

```sh
python3 data/other/forms/raw_data/sil_adi_2015/import_adi_2015.py \
  --pdf ../tmp/pdfs/adi_2015/silesr2015_016.pdf --stage
python3 data/other/forms/raw_data/sil_adi_2015/preintegration_audit.py
uv run --with pytest --with segments pytest -q tests/test_sil_adi_2015.py \
  data/other/forms/raw_data/sil_adi_2015/manual_chunks/test_*_hand_keyed.py \
  tests/test_sound_profiles.py
```

Expected source-local result: 2,763/2,763 reviewed cells = 2,670 attested +
93 source blanks + 0 ambiguous + 0 illegible; 2,770 staged forms; 2,763 audit
rows; 0 unresolved rows.

Shared source-specific installation, reference, registry and profile gates are
complete. Remaining work is the deliberately deferred consolidated full build,
compiled source-survival and identity checks, global source-audit regeneration,
full repository test suite, and (only if requested) browser refresh/QA.
