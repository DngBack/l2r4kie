"""Evaluation of extracted values.

* :mod:`.comparator`: exact-match comparison (``text``, the default, and the old ``json``).
* :mod:`.errors`: recognisable failure modes (e.g. box coordinates instead of text).
* :mod:`.extraction`: metrics of prediction rows (EM micro/macro, scalar/array,
  coordinate and truncated rates, row-level metrics of arrays).
* :mod:`.compare`: paired document-bootstrap difference between two prediction sets.
* :mod:`.metrics`: calibration metrics of confidence logits (step 6).
"""
