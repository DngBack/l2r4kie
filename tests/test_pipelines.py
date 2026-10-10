"""End-to-end tests of the confidence pipelines on a fake extractor (CPU, seconds).

``cache-traces`` → ``select-heads`` → ``finalize`` → ``audit`` run on decode
results whose states carry a planted correctness signal, then ``infer`` with
the frozen bundle must reproduce the audited confidences exactly (serving
parity: same features, same head code).
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from l2r4kie.confidence.bundle import ConfidenceBundle
from l2r4kie.confidence.config import ConfidenceConfig, GridEntry, HeadTraining
from l2r4kie.data.cohorts import CohortPlan
from l2r4kie.data.selection import Selection
from l2r4kie.data.types import Document, FieldSpec
from l2r4kie.model.decode import TOKEN_STATS, DecodeResult, Signals, Trace
from l2r4kie.pipelines import infer as infer_module
from l2r4kie.pipelines import trace_cache as trace_cache_module
from l2r4kie.pipelines.audit import audit, bootstrap_review, paired_systems
from l2r4kie.pipelines.finalize import finalize
from l2r4kie.pipelines.head_selection import select_heads
from l2r4kie.pipelines.trace_cache import cache_traces, resolve_cohorts
from l2r4kie.utils.io import read_json, read_jsonl, write_jsonl

H = 16
CLOSE = 1
SPLITS = {'train': 40, 'dev': 24, 'calibration': 24, 'risk_validation': 30, 'audit': 20}
FIELDS = 6


class FakeTokenizer:
    """One token per character; the close marker decodes to a tag."""

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:  # noqa: ARG002
        return '<close>' if ids == [CLOSE] else chr(ids[0])


def fake_extractor() -> SimpleNamespace:
    return SimpleNamespace(core=SimpleNamespace(visual=torch.nn.Identity()),
                           format=SimpleNamespace(tokenizer=FakeTokenizer()), dtype=torch.float32,
                           config=SimpleNamespace(max_value_tokens=32, max_pixels=100, close='box_end'))


def fake_extract(targets: dict[tuple[str, str], str]):  # noqa: ANN201
    """An ``extract`` whose outputs depend only on (page, field): wrong values have shifted states."""

    def extract(extractor, pages, requests, max_value_tokens=None, trace=False, layers=()):  # noqa: ANN001, ANN202, ARG001
        extractor.core.visual(torch.zeros(1))  # the one vision forward
        page = Path(pages[0]).name
        results = []
        for request in requests:
            target = targets[(page, request.id)]
            g = torch.Generator().manual_seed(zlib.crc32(f'{page}{request.id}'.encode()))
            draw = float(torch.rand(1, generator=g))
            if draw < .06:
                signals = Signals(torch.randn(H, generator=g), torch.randn(H, generator=g), None)
                results.append(DecodeResult(request.id, None, target[:2], 'truncated', signals))
                continue
            wrong = draw < .36
            text = target + 'x' if wrong else target
            shift = -1.5 if wrong else 1.5
            n = len(text)
            hidden = torch.randn(n + 1, H, generator=g)
            hidden[:, 0] += shift
            stats = torch.rand(n + 1, len(TOKEN_STATS), generator=g)
            stats[:, 0] = -stats[:, 0] * (3 if wrong else 1)
            value = torch.randn(H, generator=g)
            value[0] += shift
            deep = torch.randn(3, len(layers), H, generator=g) if layers else None
            if deep is not None:
                deep[2, :, 1] += shift
            signals = Signals(torch.randn(H, generator=g), torch.randn(H, generator=g), value, deep)
            tokens = (*(ord(c) for c in text), CLOSE)
            results.append(DecodeResult(request.id, text, text, 'ok', signals,
                                        Trace(tokens, hidden, stats) if trace else None))
        return results

    return extract


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Cohorts of fake documents decoded into a trace cache."""
    documents: dict[str, list[Document]] = {}
    targets: dict[tuple[str, str], str] = {}
    for split, count in SPLITS.items():
        documents[split] = []
        for i in range(count):
            id_ = f'{split}-{i}'
            fields = tuple(FieldSpec(f'/f{j}', f'field {j}', f'v{i}{j}' * (1 + j % 3), 'scalar')
                           for j in range(FIELDS))
            page = tmp_path / 'pages' / f'{id_}.png'
            page.parent.mkdir(exist_ok=True)
            page.write_bytes(b'png')
            documents[split].append(Document(id_, id_, f'form{i % 3}', (str(page),), fields, ('0' * 64,)))
            targets.update({(page.name, f.id): f.value for f in fields})
    extract = fake_extract(targets)
    monkeypatch.setattr(trace_cache_module, 'extract', extract)
    monkeypatch.setattr(infer_module, 'extract', extract)
    cache = tmp_path / 'cache'
    extractor = fake_extractor()
    for split in SPLITS:
        cache_traces(extractor, documents[split], cache, split, layers=(3,), max_trace_tokens=4,
                     provenance={'source_fingerprint': 'fp'}, part_size=7)
    plan = tmp_path / 'plan.json'
    CohortPlan({s: [d.id for d in docs] for s, docs in documents.items()}, 1, ['form0', 'form1', 'form2'],
               extra={'fresh_audit': [d.id for d in documents['audit'][:10]]}).save(plan)
    heads = HeadTraining(seeds=(1,), steps=40, eval_steps=(20, 40), batch_size=32, pair_batch=8,
                         min_field_count=4, grid=(
                             GridEntry(mode=('hybrid', 'attention'), l2=(.001,), rank=(0., .3)),
                             GridEntry(mode=('hybrid',), signals=(('value', 'value@3'),), l2=(.001,), rank=(0.,)),
                             GridEntry(mode=('query',), l2=(.001,), rank=(0.,), prior=(True,))))
    config = ConfidenceConfig('extractor.yaml', 'adapter', str(plan), str(cache), str(tmp_path / 'selected'),
                              device='cpu', layers=(3,), max_trace_tokens=4, target_error_recall=.8,
                              risk_replicates=200, heads=heads)
    return SimpleNamespace(config=config, documents=documents, extractor=extractor, tmp=tmp_path)


def test_cache_traces_writes_rows_records_and_merges_parts(run: SimpleNamespace) -> None:
    cache = Path(run.config.cache)
    summary = read_json(cache / 'train.summary.json')
    assert summary['fields'] == SPLITS['train'] * FIELDS and summary['valid'] < summary['fields']
    assert summary['layers'] == [3] and summary['source_fingerprint'] == 'fp'
    assert not list(cache.glob('*.part*.pt'))
    data = torch.load(cache / 'train.pt', weights_only=True)
    record = data['traces'][0]
    assert record['layers'].shape == (3, 1, H) and len(record['tokens']) <= 4
    # Rerunning with other settings is refused; with the same settings the cache is reused.
    with pytest.raises(ValueError, match='different run'):
        cache_traces(run.extractor, run.documents['train'], cache, 'train', layers=(), max_trace_tokens=4,
                     provenance={'source_fingerprint': 'fp'}, part_size=7)


def test_select_finalize_audit_and_serving_parity(run: SimpleNamespace) -> None:
    config = run.config
    out = Path(config.output)
    selection = select_heads(config, 'fp')
    families = set(selection['families'])
    assert {'min_log_probability', 'mean_log_probability'} <= families
    assert any(f.startswith('query') and f.endswith('+prior') for f in families)
    assert selection['primary'] in families and (out / 'trials.json').is_file()
    with pytest.raises(ValueError, match='already holds'):
        select_heads(config)

    dry = finalize(out, dry_run=True, replicates=200)
    assert not (out / 'frozen.json').exists() and set(dry['candidates']) == families
    finalize(out, replicates=200, layers=config.layers, max_trace_tokens=config.max_trace_tokens)
    frozen = read_json(out / 'frozen.json')
    assert frozen['source_fingerprint'] == 'fp' and not frozen['audit_seen']
    with pytest.raises(ValueError, match='already frozen'):
        finalize(out)

    baseline = run.tmp / 'baseline.jsonl'
    rows = torch.load(Path(config.cache) / 'audit.pt', weights_only=True)['rows']
    write_jsonl(baseline, ({**r, 'needs_review': True} for r in rows))
    report = audit(out, cohort_plan=config.cohort_plan, baseline_predictions=baseline)
    primary = report['candidates'][frozen['primary']]
    assert set(primary['subsets']) == {'fresh_audit'}
    assert primary['subsets']['fresh_audit']['documents'] == 10
    assert primary['versus_baseline']['baseline_review_rate'] == 1.
    assert primary['versus_baseline']['review_rate_change'] <= 0
    assert set(report['paired_versus_primary']) == families - {frozen['primary']}
    with pytest.raises(ValueError, match='already audited'):
        audit(out)

    # Serving: the same document through infer gives the audited confidences and review decisions.
    folder = out / 'heads' / frozen['primary']
    bundle = ConfidenceBundle.load(folder)
    bundle.check_source('fp')
    with pytest.raises(ValueError, match='trained on extractor'):
        bundle.check_source('other')
    audited = {(r['document_id'], r['field_id']): r for r in read_jsonl(folder / 'audit_predictions.jsonl')}
    document = run.documents['audit'][3]
    request = {'pages': list(document.pages),
               'fields': [{'id': f.id, 'description': f.description} for f in document.fields]}
    response = infer_module.infer(run.extractor, request, bundle=bundle)
    assert response['calibrated']
    reviewed = {item['id'] for item in response['review']}
    for field in document.fields:
        expected = audited[(document.id, field.id)]
        if expected['confidence'] is None:
            assert response['confidence'][field.id] is None
        else:
            assert response['confidence'][field.id] == pytest.approx(expected['confidence'], abs=1e-6)
        assert (field.id in reviewed) == expected['needs_review']
    assert response['document_needs_review'] == bool(reviewed)
    json.dumps(response)


def test_audit_refuses_a_changed_bundle(run: SimpleNamespace) -> None:
    out = Path(run.config.output)
    select_heads(run.config)
    finalize(out, replicates=100)
    frozen = read_json(out / 'frozen.json')
    policy = Path(frozen['candidates'][frozen['primary']]['folder']) / 'review_policy.json'
    policy.write_text(policy.read_text().replace('"threshold"', '"threshold" ', 1))
    with pytest.raises(ValueError, match='changed after freezing'):
        audit(out)


def test_audit_intervals_and_paired_systems() -> None:
    rows = [{'document_id': f'd{i // 5}', 'field_id': f'/{i % 5}', 'correct': i % 3 != 0} for i in range(100)]
    flags = [not r['correct'] or i % 7 == 0 for i, r in enumerate(rows)]
    intervals = bootstrap_review(rows, flags, replicates=200)
    assert intervals['error_recall'] == [1., 1.]
    assert intervals['review_rate'][0] < sum(flags) / 100 < intervals['review_rate'][1]
    paired = paired_systems(rows, flags, rows, [True] * 100, replicates=200)
    assert paired['baseline_review_rate'] == 1. and paired['review_rate_change'] < 0
    assert paired['review_rate_change_95'][1] < 0
    with pytest.raises(ValueError, match='share no'):
        paired_systems(rows, flags, [{**rows[0], 'document_id': 'x'}], [True])


def test_resolve_cohorts_refuses_extractor_training_documents(tmp_path: Path) -> None:
    for split in ('train', 'dev', 'calibration', 'test'):
        docs = [Document(f'{split}-{i}', f'{split}-{i}', 'f', (), (FieldSpec('/a', '/a', 'x', 'scalar'),), ())
                for i in range(4)]
        write_jsonl(tmp_path / f'{split}.jsonl', (d.to_json() for d in docs))
    selection = Selection(tmp_path)
    plan = tmp_path / 'plan.json'
    CohortPlan({'train': ['train-0'], 'dev': ['dev-0', 'dev-1']}, 1, ['f']).save(plan)
    assert [d.id for d in resolve_cohorts(plan, selection, ['dev'])['dev']] == ['dev-0', 'dev-1']
    with pytest.raises(ValueError, match='extractor training documents'):
        resolve_cohorts(plan, selection, ['train'])
    with pytest.raises(KeyError, match='audit'):
        resolve_cohorts(plan, selection, ['audit'])
