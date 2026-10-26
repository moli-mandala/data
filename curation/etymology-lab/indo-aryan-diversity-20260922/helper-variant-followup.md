# Existing attestation parents with saved ancestry

Zargari pass006 originally included the source-explicit čhiv → čhib variant while also assigning čhib → CDIAL 5228. Shared helper etymology_lab.py validate_against_graph rejects the parent because its compiled Status remains unlinked. The ancestry of čhib is now saved, but no rebuild is authorised. The variant remains held with exact target ID and evidence in pass006-decisions.json.

A future focused helper improvement could evaluate donor/variant parents against the combined compiled graph, accepted overlay and current pass, with cycle and identity validation. Do not bypass the shared saver or refresh the compiled data to work around this. First confirm intended helper policy and add focused regression tests before changing it.

Update after pass031: the effective-overlay scratch validation fix resolves the shared-helper limitation. The čhiv variant and seven pass030 derivations are saved and independently verified. No rebuild.
