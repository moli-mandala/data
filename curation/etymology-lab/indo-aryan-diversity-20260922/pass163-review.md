# Grierson Khetrani: one and ten

Two numeral records in the LSI digital edition are compared with precise western Indo-Aryan forms in Turner. Exact original-spelling rows and keys are in `pass163-source-audit.json`; full articles are in `khetrani-primary010.json`. The physical LSI table image has not been independently checked. The unusual **ḍḍāh** spelling is preserved without emendation and its difference from Turner’s Khetrani *dā* is documented.

| # | Khetrani | Proposed etymology | Evidence |
|---|---|---|---|
| 1 | **hik** ‘one’ | **\*ēkka-**, CDIAL **2462-2** | Turner lists Sindhi *heko*, Lahnda *hikk* and Punjabi *hik-* in this second branch. |
| 2 | **ḍḍāh** ‘ten’ | **dáśa**, CDIAL **6227** | Sindhi *ḍ̠aha* and Lahnda *ḍ̠ā̆h* match the regional numeral; doubled onset awaits image verification. |

Two `reflex` rows affect two records. **cā** ‘what’ remains held, and the exact transcription of **ḍḍāh** remains pending a physical scan check. No source ingestion or rebuild.

`make check-pass`, `make save-pass`, and `make check-sources` passed. Both exact rows appear once in the owning LSI sidecar; the second application changed zero rows. Compiled forms, graph and identity hashes were unchanged. No rebuild.
