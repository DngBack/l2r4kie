"""Correctness confidence of extracted values and the review policy built on it.

* :mod:`.features`: trace records of one decoded field (marker states
  ``key``/``decide``/``value``, intermediate layers, token states and
  statistics, value summary) and :class:`~.features.TraceStore` batches.
* :mod:`.heads`: correctness heads (``end`` … ``attention``, new ``query``)
  and training-free heuristics; old r4 heads load too.
* :mod:`.calibration`: identity / temperature / affine, chosen by
  document-grouped cross-validation.
* :mod:`.policy`: review policies, fitted on dev (head selection) or with
  observational lower bounds on risk_validation (frozen threshold).
* :mod:`.cache`: per-cohort trace caches (new and old layouts).
* :mod:`.config`: the confidence run config (``configs/confidence/*.yaml``).
* :mod:`.bundle`: a frozen head + calibration + policy, applied by ``infer``.
"""
