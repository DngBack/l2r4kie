"""Tests for ``l2r4kie.confidence``: features, trace store, heads, calibration, caches, config."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from l2r4kie.confidence.cache import load_cache, save_cache
from l2r4kie.confidence.calibration import IDENTITY, calibrate, fit_calibration, select_calibration
from l2r4kie.confidence.config import ConfidenceConfig, GridEntry, HeadTraining
from l2r4kie.confidence.features import (TraceStore, content_mask, parse_signal, signal_vector, summarize,
                                         summary_names, trace_record, value_kind)
from l2r4kie.confidence.heads import (MODES, ConfidenceHead, HeadConfig, HeuristicHead, load_head,
                                      load_legacy_head, save_head, score_head)
from l2r4kie.model.decode import TOKEN_STATS, DecodeResult, Signals, Trace

H = 8
K = len(TOKEN_STATS)


def decoded(n: int, text: str = 'abc', seed: int = 0, layers: int = 0) -> DecodeResult:
    """A closed decode result with ``n`` value tokens and random states."""
    g = torch.Generator().manual_seed(seed)
    stats = torch.rand(n + 1, K, generator=g)
    stats[:, 0] = -stats[:, 0]  # log-probabilities are negative
    signals = Signals(torch.randn(H, generator=g), torch.randn(H, generator=g), torch.randn(H, generator=g),
                      torch.randn(3, layers, H, generator=g) if layers else None)
    trace = Trace(tuple(range(n + 1)), torch.randn(n + 1, H, generator=g), stats)
    return DecodeResult('/a', text, text, 'ok', signals, trace)


def records(count: int = 12, layers: int = 0) -> list[dict]:
    """Trace records of varied lengths (one empty value)."""
    out = []
    for i in range(count):
        n = i % 4
        out.append(trace_record(decoded(n, 'x' * n, seed=i, layers=layers), ['p'] * n, 'scalar'))
    return out


# ------------------------------------------------------------------ features

def test_summary_has_one_value_per_name_and_kind_one_hot() -> None:
    stats = torch.tensor([[-.1, .2, .9, .5, -3.], [-2., .8, .1, .4, -1.]])
    close = torch.tensor([-.05, .1, .95, .6, -.05])
    summary = summarize(stats, close, [True, True], '12', 'scalar')
    names = summary_names()
    assert summary.shape == (len(names),) == (27,)
    assert summary[names.index('min_log_probability')] == pytest.approx(-2.)
    assert summary[names.index('max_close_log_probability')] == pytest.approx(-1.)
    assert summary[names.index('log1p_tokens')] == pytest.approx(math.log(3))
    assert summary[names.index('kind_numeric')] == 1 and summary[names.index('kind_text')] == 0


def test_empty_value_summary_keeps_the_close_decision() -> None:
    close = torch.tensor([-.3, .1, .5, .2, -.3])
    summary = summarize(torch.zeros(0, K), close, [], '', 'scalar')
    names = summary_names()
    assert summary[names.index('close_log_probability')] == pytest.approx(-.3)
    assert summary[names.index('kind_empty')] == 1


@pytest.mark.parametrize(('text', 'kind', 'expected'), [
    ('', 'scalar', 'empty'), ('  ', 'scalar', 'empty'), ('12/03/2024', 'scalar', 'numeric'),
    ('1.000.000', 'scalar', 'numeric'), ('Hà Nội', 'scalar', 'text'), ('[]', 'array', 'array')])
def test_value_kind(text: str, kind: str, expected: str) -> None:
    assert value_kind(text, kind) == expected


def test_content_mask_drops_whitespace_and_json_syntax() -> None:
    assert content_mask(['Hà', ' ', 'Nội'], 'scalar') == [True, False, True]
    assert content_mask(['[{"', 'a', '":', ' "', 'x', '"}]'], 'array') == [False, True, False, False, True, False]
    assert content_mask(['[', ']'], 'array') == [True, True]  # nothing to pool otherwise


def test_trace_record_keeps_the_lowest_probability_tokens_in_order() -> None:
    result = decoded(6)
    stats = result.trace.stats
    stats[:6, 0] = torch.tensor([-.1, -5., -.2, -4., -.3, -3.])
    record = trace_record(result, list('abcdef'), 'scalar', max_tokens=3)
    assert record['length'] == 6
    torch.testing.assert_close(record['stats'][:, 0], torch.tensor([-5., -4., -3.]))
    torch.testing.assert_close(record['tokens'].float(), result.trace.hidden[[1, 3, 5]].bfloat16().float())
    # The summary still covers all six tokens.
    assert record['summary'][summary_names().index('min_log_probability')] == pytest.approx(-5.)
    torch.testing.assert_close(record['close'], stats[-1])


def test_trace_record_needs_a_closed_value() -> None:
    result = decoded(2)
    open_ = DecodeResult('/a', None, 'ab', 'truncated', Signals(result.signals.key, result.signals.decide, None),
                         result.trace)
    with pytest.raises(ValueError, match='closed'):
        trace_record(open_, ['a', 'b'], 'scalar')
    with pytest.raises(ValueError, match='pieces'):
        trace_record(result, ['a'], 'scalar')


def test_signals_and_layers() -> None:
    assert parse_signal('value') == ('value', None)
    assert parse_signal('key@14') == ('key', 14)
    for bad in ('end', 'value@x', 'h_value'):
        with pytest.raises(ValueError):
            parse_signal(bad)
    record = trace_record(decoded(2, layers=2), ['a', 'b'], 'scalar')
    torch.testing.assert_close(signal_vector(record, 'decide@21', (7, 21)), record['layers'][1, 1])
    with pytest.raises(KeyError, match='layer 14'):
        signal_vector(record, 'value@14', (7, 21))


# ------------------------------------------------------------------ store

def test_store_batches_flat_tokens_with_padding() -> None:
    recs = records(8)
    store = TraceStore(recs, ('value', 'key'))
    assert len(store) == 8 and store.hidden_size == H and store.stats_size == K
    batch = store.batch([3, 0, 2])
    assert batch['tokens'].shape == (3, 3, H)
    assert batch['valid'].tolist() == [[True] * 3, [True, False, False], [True, True, False]]
    torch.testing.assert_close(batch['tokens'][0], recs[3]['tokens'])
    torch.testing.assert_close(batch['tokens'][2, :2], recs[2]['tokens'])
    assert not batch['tokens'][1, 1:].any()  # padding row
    # The empty value gets one pseudo-token: its value state, zero statistics.
    torch.testing.assert_close(batch['tokens'][1, 0].float(), recs[0]['value'].bfloat16().float())
    assert not batch['stats'][1, 0].any()
    torch.testing.assert_close(batch['vectors']['key'][0], recs[3]['key'])
    assert 'tokens' not in store.batch([0], tokens=False)


def test_store_normalizers_use_content_tokens_with_floors() -> None:
    store = TraceStore(records(8))
    values = store.normalizers(('value',))
    assert values['vector_mean'].shape == (1, H)
    assert (values['vector_std'] >= .1).all() and (values['stats_std'] >= .05).all()
    with pytest.raises(ValueError):
        TraceStore([])


# ------------------------------------------------------------------ heads

@pytest.mark.parametrize('mode', MODES)
def test_every_mode_scores_and_round_trips(mode: str, tmp_path: Path) -> None:
    recs = records(10, layers=1)
    signals = ('value', 'key@3') if mode != 'query' else ('value',)
    store = TraceStore(recs, (*signals, 'key'), layers=(3,))
    config = HeadConfig(mode=mode, signals=signals, width=4, field_keys=('/a',))
    torch.manual_seed(0)
    head = ConfidenceHead(config, H, store.summary_size, K)
    head.normalize(store)
    with torch.no_grad():
        for p in head.parameters():
            p.normal_()
    logits = score_head(head, store, ['/a'] * 10, batch_size=3)
    assert logits.shape == (10,) and torch.isfinite(logits).all()
    loaded = load_head(save_head(head, tmp_path / mode))
    torch.testing.assert_close(score_head(loaded, store, ['/a'] * 10), logits)
    assert json.loads((tmp_path / mode / 'head_config.json').read_text())['hidden_size'] == H


def test_family_names_tell_heads_apart() -> None:
    assert HeadConfig('attention').family == 'attention[value]'
    assert HeadConfig('attention', signals=('value', 'key')).family != HeadConfig('attention').family
    assert HeadConfig('attention', field_keys=('/a',)).family.endswith('+prior')
    assert HeadConfig('query').required_signals() == ('value', 'key')


def test_field_prior_needs_field_ids() -> None:
    store = TraceStore(records(4))
    head = ConfidenceHead(HeadConfig('hybrid', field_keys=('/a',)), H, store.summary_size, K)
    with pytest.raises(ValueError, match='field ids'):
        score_head(head, store)
    # Unknown fields fall back to index 0: no learned offset.
    assert head.field_indices(['/a', '/b']).tolist() == [1, 0]


def test_heuristics_rank_by_token_probability() -> None:
    recs = records(8)
    store = TraceStore(recs)
    minimum = score_head(HeuristicHead(HeadConfig('min_log_probability', signals=())), store)
    for record, score in zip(recs, minimum, strict=True):
        values = torch.cat((record['stats'][:, 0], record['close'][:1]))
        assert score == pytest.approx(float(values.min()))
    mean = score_head(HeuristicHead(HeadConfig('mean_log_probability', signals=())), store)
    record = recs[3]
    assert mean[3] == pytest.approx(float(torch.cat((record['stats'][:, 0], record['close'][:1])).mean()), rel=1e-5)


def test_legacy_head_loads_with_renamed_buffers(tmp_path: Path) -> None:
    store = TraceStore(records(6))
    torch.manual_seed(1)
    head = ConfidenceHead(HeadConfig('attention', width=4), H, store.summary_size, K)
    head.normalize(store)
    state = head.state_dict()
    state['mean'], state['std'] = state.pop('vector_mean')[0], state.pop('vector_std')[0]
    torch.save(state, tmp_path / 'head.pt')
    (tmp_path / 'head_config.json').write_text(json.dumps({'kind': 'token', 'mode': 'attention', 'width': 4}))
    legacy = load_legacy_head(tmp_path).head
    torch.testing.assert_close(score_head(legacy, store), score_head(head, store))
    (tmp_path / 'head_config.json').write_text(json.dumps({'kind': 'linear'}))
    with pytest.raises(ValueError, match='token'):
        load_legacy_head(tmp_path)


# ------------------------------------------------------------------ calibration

def test_temperature_calibration_recovers_the_scale() -> None:
    g = torch.Generator().manual_seed(0)
    true = torch.randn(4000, generator=g) * 2
    labels = torch.bernoulli(torch.sigmoid(true), generator=g).tolist()
    fitted = fit_calibration(true * 3, labels, 'temperature')
    assert fitted['temperature'] == pytest.approx(3, rel=.1) and fitted['bias'] == 0
    torch.testing.assert_close(calibrate(true, IDENTITY), true.double())


def test_select_calibration_reports_every_family_and_groups_documents() -> None:
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(400, generator=g)
    labels = torch.bernoulli(torch.sigmoid(2 * logits + 1), generator=g).tolist()
    documents = [f'd{i // 10}' for i in range(400)]
    result = select_calibration(logits, labels, documents)
    assert set(result['cv_nll']) == {'identity', 'temperature', 'affine'}
    assert result['method'] == 'affine' and result['documents'] == 40
    with pytest.raises(ValueError, match='align'):
        select_calibration(logits, labels[:-1], documents)


# ------------------------------------------------------------------ cache and config

def test_cache_round_trip_and_scores(tmp_path: Path) -> None:
    recs = records(3)
    rows = [{'document_id': 'd', 'form': 'f', 'field_id': f'/{i}', 'status': 'ok', 'correct': bool(i % 2),
             'split': 'dev', 'trace_index': i} for i in range(3)]
    rows.append({'document_id': 'd', 'form': 'f', 'field_id': '/t', 'status': 'truncated', 'correct': False,
                 'split': 'dev', 'trace_index': -1})
    save_cache(tmp_path / 'dev.pt', rows, recs, {'split': 'dev', 'source_fingerprint': 'x'})
    cache = load_cache(tmp_path, 'dev')
    assert not cache.legacy and cache.labels == [0., 1., 0.] and cache.documents == {'d'}
    scores = cache.scores(torch.tensor([0., 100., -100.]))
    assert scores[0] == pytest.approx(.5) and scores[3] is None


def test_legacy_cache_is_renamed(tmp_path: Path) -> None:
    record = {'end': torch.ones(H), 'tokens': torch.zeros(2, H, dtype=torch.bfloat16),
              'token_stats': torch.zeros(2, 4), 'mask': torch.ones(2, dtype=torch.bool), 'summary': torch.zeros(21)}
    torch.save({'features': None, 'traces': [record], 'metadata': {'split': 'dev'},
                'rows': [{'document_id': 'd', 'field_id': '/a', 'status': 'ok', 'correct': True,
                          'split': 'dev', 'feature_index': 0}]}, tmp_path / 'dev.pt')
    cache = load_cache(tmp_path, 'dev')
    assert cache.legacy and cache.rows[0]['trace_index'] == 0
    torch.testing.assert_close(cache.records[0]['value'], torch.ones(H))
    assert cache.records[0]['stats'].shape == (2, 4)


def test_config_is_strict_and_checks_layers() -> None:
    base = {'extractor': 'e.yaml', 'adapter': 'a', 'cohort_plan': 'p.json', 'cache': 'c', 'output': 'o'}
    config = ConfidenceConfig.from_dict({**base, 'layers': [14], 'heads': {'grid': [
        {'mode': ['attention', 'query'], 'signals': [['value', 'value@14']], 'prior': [True, False]}]}})
    assert config.heads.signals() == ('value', 'value@14', 'key')
    assert sum(1 for e in config.heads.grid for _ in e.combinations()) == 2 * 2 * 2 * 2
    assert ConfidenceConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(ValueError, match='Unknown key'):
        ConfidenceConfig.from_dict({**base, 'layer': [14]})
    with pytest.raises(ValueError, match='heads.grid'):
        ConfidenceConfig.from_dict({**base, 'heads': {'grid': [{'mode': ['end'], 'width': 3}]}})
    with pytest.raises(ValueError, match='needs layer 21'):
        ConfidenceConfig.from_dict({**base, 'heads': {'grid': [{'mode': ['end'], 'signals': [['value@21']]}]}})
    with pytest.raises(ValueError, match='mode'):
        GridEntry(mode=('linear',))
    with pytest.raises(ValueError, match='eval_steps'):
        HeadTraining(steps=10, eval_steps=(20,))


def test_shipped_config_loads() -> None:
    from l2r4kie.utils.config import load_config

    config = ConfidenceConfig.from_dict(load_config(Path(__file__).parents[1] / 'configs/confidence/kev.yaml', []))
    assert set(config.heads.signals()) == {'value', 'key', 'decide', 'value@14', 'value@21'}
    assert config.heads.seeds == (107, 108, 109)
