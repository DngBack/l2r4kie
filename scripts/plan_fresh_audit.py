"""Plan a fresh audit cohort from documents no earlier run has used (``docs/results/fresh_audit.md``).

The new plan keeps the development cohorts of ``--base-plan`` (train, dev,
calibration, risk_validation: the frozen heads were fitted on them, and the
audit checks it does not overlap them) and replaces ``audit`` with documents
of the selectable splits that are in none of:

* the base plan's cohorts (the old audit included) or its excluded documents;
* the extractor's training set;
* the documents of the extractor's evaluations (``--eval-provenance``).

Up to ``--per-form`` documents per form, each form's pool shuffled (by id
order) with one RNG seeded by ``--seed``, forms in sorted order. Plan
subsets of the old audit (``*_audit``) are dropped.

    python scripts/plan_fresh_audit.py --output artifacts/kev-r4/confidence/audit2/cohort_plan.json
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from l2r4kie.data.cohorts import CohortPlan, check_disjoint
from l2r4kie.data.selection import Selection
from l2r4kie.train.config import TrainConfig
from l2r4kie.utils.config import load_config
from l2r4kie.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-plan', default='/home/jovyan/bachdx2/l2r4kie/artifacts/token-review-v2/cache/'
                                               'cohort_plan.json')
    parser.add_argument('--extractor', default='configs/extractor/kev_r4.yaml')
    parser.add_argument('--eval-provenance', nargs='*', default=['artifacts/kev-r4/eval-dev/provenance.json',
                                                                 'artifacts/kev-r4/eval-test/provenance.json'])
    parser.add_argument('--per-form', type=int, default=20)
    parser.add_argument('--seed', type=int, default=211)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    base = CohortPlan.load(args.base_plan)
    config = TrainConfig.from_dict(load_config(args.extractor))
    s = config.selection
    selection = Selection(Path(s.prepared), s.seed, s.holdout_percent, s.forms, s.balanced_forms)
    evaluated = {i for p in args.eval_provenance for i in read_json(p)['documents']}
    trained = {d.id for d in selection.documents('train', config.train_documents)}
    used = set().union(*base.cohorts.values()) | set(base.excluded_documents) | evaluated | trained
    pool: dict[str, list[str]] = defaultdict(list)
    for split in (base.train_split, 'dev', 'calibration', 'test'):
        for document in selection.documents(split):
            if document.id not in used:
                pool[document.form].append(document.id)
    rng = random.Random(args.seed)
    audit: list[str] = []
    for form in sorted(pool):
        ids = sorted(pool[form])
        rng.shuffle(ids)
        audit.extend(ids[:args.per_form])
    cohorts = {**{k: v for k, v in base.cohorts.items() if k != 'audit'}, 'audit': audit}
    excluded = sorted(set(base.excluded_documents) | set(base.cohorts['audit']) | evaluated)
    check_disjoint(cohorts, excluded)
    per_form = Counter(i.split('__')[0] for i in audit)
    plan = CohortPlan(
        cohorts=cohorts, seed=args.seed, forms=sorted(pool), train_split=base.train_split,
        excluded_documents=excluded, exclusion_sources=[*base.exclusion_sources, args.base_plan, *args.eval_provenance],
        source_checkpoint=base.source_checkpoint, source_fingerprint=base.source_fingerprint,
        extra={'derived_from': args.base_plan, 'per_form': args.per_form, 'audit_forms': dict(sorted(per_form.items())),
               'note': 'Fresh audit: documents in no earlier cohort, exclusion list, extractor training set or '
                       'extractor evaluation. Development cohorts are the base plan\'s. Only the forms with '
                       'unused documents are covered, so this audit does not replace G4.'})
    plan.save(args.output)
    print({'audit_documents': len(audit), 'forms': len(per_form), 'per_form': dict(per_form),
           'pool': sum(map(len, pool.values())), 'excluded': len(excluded)})


if __name__ == '__main__':
    main()
