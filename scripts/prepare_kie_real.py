"""Split the chat-style KIE datasets (IDP-Training-Pipeline/Data/kie) into train/dev/test.

Documents sharing a page image (the v1/v2 prompt variants of one scan, duplicated
scans) form one group, and splits are drawn per group, so no page is in two
splits. Per dataset, test takes ``min(--test-per-dataset, --test-fraction of the
groups)`` groups, preferring groups holding a document of ``--prefer`` (the
sample the synthetic-trained models were evaluated on, so they stay comparable);
dev takes up to ``--dev-per-dataset`` of the rest; train keeps everything else.
Test and dev keep one prompt variant (v1); train keeps all variants.

    python scripts/prepare_kie_real.py --output artifacts/kie-real \
        --prefer artifacts/kie-external/sample50.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from l2r4kie.data.conversations import iter_dataset
from l2r4kie.utils.fingerprint import sha256_file
from l2r4kie.utils.io import read_jsonl, write_json, write_jsonl

DATA = '/home/jovyan/scratch/ntvinh/IDP-Training-Pipeline/Data/kie'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=DATA)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prefer', help='documents JSONL whose documents go to test first')
    parser.add_argument('--test-per-dataset', type=int, default=50)
    parser.add_argument('--test-fraction', type=float, default=0.5)
    parser.add_argument('--dev-per-dataset', type=int, default=3)
    parser.add_argument('--max-train-pages', type=int, default=6, help='drop longer training documents (memory)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    documents = []
    for directory in sorted(p for p in Path(args.data).iterdir() if (p / 'image_label_details.json').is_file()):
        documents.extend(iter_dataset(directory, variant=None))
    pages = sorted({p for d in documents for p in d.pages})
    with ThreadPoolExecutor(16) as pool:
        digest = dict(zip(pages, pool.map(sha256_file, pages), strict=True))

    # Union-find over shared page hashes.
    parent = list(range(len(documents)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for i, d in enumerate(documents):
        for p in d.pages:
            j = owner.setdefault(digest[p], i)
            parent[find(i)] = find(j)
    members = defaultdict(list)
    for i in range(len(documents)):
        members[find(i)].append(i)
    group_of = {}
    for ids in members.values():
        name = min(documents[i].id for i in ids)
        for i in ids:
            group_of[i] = name
    documents = [replace(d, group_id=group_of[i], image_sha256=tuple(digest[p] for p in d.pages))
                 for i, d in enumerate(documents)]

    preferred = {r['id'] for r in read_jsonl(args.prefer)} if args.prefer else set()
    # A group belongs to the dataset of the document it is named after, even when
    # the same scan also appears in another dataset.
    form_of = {d.id: d.form for d in documents}
    by_dataset = defaultdict(lambda: defaultdict(list))
    for d in documents:
        by_dataset[form_of[d.group_id]][d.group_id].append(d)
    splits = {'train': [], 'dev': [], 'test': []}
    report = {}
    for dataset, groups in sorted(by_dataset.items()):
        rng = random.Random(f'{args.seed}:{dataset}')
        names = sorted(groups)
        rng.shuffle(names)
        names.sort(key=lambda g: not any(d.id in preferred for d in groups[g]))  # stable: preferred first
        n_test = min(args.test_per_dataset, int(len(names) * args.test_fraction))
        n_dev = min(args.dev_per_dataset, (len(names) - n_test) // 10)
        chosen = {'test': names[:n_test], 'dev': names[n_test:n_test + n_dev], 'train': names[n_test + n_dev:]}
        counts = Counter()
        for split, group_names in chosen.items():
            for g in group_names:
                for d in groups[g]:
                    variant = re.search(r'__v(\d+)$', d.id)
                    if split != 'train' and variant and variant.group(1) != '1':
                        continue
                    if split == 'train' and len(d.pages) > args.max_train_pages:
                        counts['train_dropped_long'] += 1
                        continue
                    splits[split].append(d)
                    counts[f'{split}_documents'] += 1
                    counts[f'{split}_fields'] += len(d.fields)
        counts['groups'] = len(names)
        counts['test_preferred'] = sum(any(d.id in preferred for d in groups[g]) for g in chosen['test'])
        report[dataset] = dict(sorted(counts.items()))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for split, docs in splits.items():
        write_jsonl(output / f'{split}.jsonl', [d.to_json() for d in docs])
    leaked = {h for d in splits['train'] for h in d.image_sha256} & {h for d in splits['test'] for h in d.image_sha256}
    totals = {s: {'documents': len(v), 'fields': sum(len(d.fields) for d in v)} for s, v in splits.items()}
    write_json(output / 'report.json', {'data': args.data, 'seed': args.seed, 'args': vars(args), 'totals': totals,
                                        'train_test_shared_pages': len(leaked), 'datasets': report})
    print(json.dumps(totals), 'shared pages train/test:', len(leaked))


if __name__ == '__main__':
    main()
