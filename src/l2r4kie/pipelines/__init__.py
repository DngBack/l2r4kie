"""End-to-end procedures built from the data, model and confidence packages.

* :mod:`.inspect`: readable views of model inputs (``show-input``, ``token-stats``).
* :mod:`.infer`: request JSON → response JSON for one document (``infer``).
* :mod:`.predict`: decode and score labelled documents (``evaluate``).
* :mod:`.trace_cache`: decode confidence cohorts once (``cache-traces``).
* :mod:`.head_selection`: train heads, select on dev (``select-heads``).
* :mod:`.finalize`: calibrate, fix thresholds, freeze (``finalize``).
* :mod:`.audit`: one measurement of the frozen heads (``audit``).
"""
