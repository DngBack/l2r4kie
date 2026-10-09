"""Extractor training (LoRA, cross-entropy on branch-packed values).

* :mod:`.config`: strictly validated run config (:class:`~.config.TrainConfig`).
* :mod:`.losses`: chunked value cross-entropy.
* :mod:`.schedule`: learning-rate schedule.
* :mod:`.snapshots`: crash-safe, resumable snapshots.
* :mod:`.trainer`: the training loop, with periodic decode monitoring.
"""
