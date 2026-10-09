"""Dataset preparation and document selection (no torch dependency).

* :mod:`.types`: ``Document`` / ``FieldSpec`` records of prepared splits.
* :mod:`.schema`: flatten nested labels into per-field branches.
* :mod:`.prepare`: raw dataset -> deduplicated train/dev/calibration/test.
* :mod:`.selection`: which documents a run reads, in which order.
* :mod:`.cohorts`: pre-declared, leak-free cohorts for confidence heads.
* :mod:`.serialize`: value <-> plain text (the KevFormat output).
* :mod:`.stats`: split sizes, value types and string features.
"""
