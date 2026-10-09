"""Dependency-light helpers shared by every stage of the pipeline.

* :mod:`.io`: JSON/JSONL reading and atomic writes.
* :mod:`.fingerprint`: file digests, checkpoint fingerprints, hash buckets.
* :mod:`.config`: YAML configs with ``key=value`` overrides.
* :mod:`.seed`: global RNG seeding (imports torch, so not re-exported here).
"""
