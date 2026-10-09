"""Tests for ``l2r4kie.data.cohorts``: exclusions, disjoint draws, plan files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from l2r4kie.cli import main
from l2r4kie.data.cohorts import (
    CohortPlan, CohortSpec, check_disjoint, load_exclusions, plan_cohorts, plan_from_config,
)
from l2r4kie.data.selection import Selection
from l2r4kie.data.types import Document, FieldSpec
from l2r4kie.utils.io import write_json, write_jsonl

OLD_REPO = Path('/home/jovyan/bachdx2/l2r4kie')
R4_CONFIG = Path(__file__).parents[1] / 'configs' / 'cohorts' / 'r4_winner.yaml'


def doc(id: str, form: str) -> Document:
    return Document(id, id, form, (), (FieldSpec('/a', '/a', 'x', 'scalar'),), ())


@pytest.fixture
def selection(tmp_path: Path) -> Selection:
    """Three forms with 20 documents in every split (and no train reserve)."""
    for split in ('train', 'dev', 'calibration', 'test'):
        docs = [doc(f'{split}-{form}-{i}', form) for form in ('a', 'b', 'c') for i in range(20)]
        write_jsonl(tmp_path / f'{split}.jsonl', (d.to_json() for d in docs))
    return Selection(tmp_path)


SPEC = CohortSpec(seed=107, train_per_type=5, per_type=2, calibration_per_type=3, risk_per_type=4)


def test_cohort_sizes_and_disjointness(selection: Selection) -> None:
    plan = plan_cohorts(selection, SPEC)
    assert {k: len(v) for k, v in plan.cohorts.items()} == {
        'train': 15, 'dev': 6, 'calibration': 9, 'risk_validation': 12, 'audit': 6}
    check_disjoint(plan.cohorts)
    assert plan.forms == ['a', 'b', 'c']
    assert all(i.startswith('train-') for i in plan.cohorts['risk_validation'])  # risk_split='train'
    assert all(i.startswith('test-') for i in plan.cohorts['audit'])


def test_risk_from_calibration_follows_calibration_cohort(selection: Selection) -> None:
    plan = plan_cohorts(selection, CohortSpec(seed=107, train_per_type=5, calibration_per_type=3,
                                              risk_per_type=4, risk_split='calibration'))
    assert all(i.startswith('calibration-') for i in plan.cohorts['risk_validation'])
    check_disjoint(plan.cohorts)


def test_plan_is_deterministic_and_seeded(selection: Selection) -> None:
    assert plan_cohorts(selection, SPEC) == plan_cohorts(selection, SPEC)
    other = plan_cohorts(selection, CohortSpec(seed=108, train_per_type=5))
    assert other.cohorts['train'] != plan_cohorts(selection, SPEC).cohorts['train']


def test_exclusions_and_excluded_forms_are_respected(selection: Selection, tmp_path: Path) -> None:
    used = [f'train-a-{i}' for i in range(20)] + ['dev-b-0', 'test-c-1']
    write_jsonl(tmp_path / 'used.jsonl', [{'id': i} for i in used[:-1]] + [{'document_id': used[-1]}, {'step': 3}])
    exclusions = load_exclusions([tmp_path / 'used.jsonl'])
    assert exclusions.ids == frozenset(used)
    plan = plan_cohorts(selection, CohortSpec(seed=1, train_per_type=5, exclude_forms=('b',)), exclusions)
    every = {i for ids in plan.cohorts.values() for i in ids}
    assert not every & set(used)
    assert not any('-b-' in i for i in every)
    assert plan.forms == ['c']  # form a has no train documents left, form b is excluded
    assert plan.excluded_documents == sorted(used)
    assert plan.exclusion_sources == [str(tmp_path / 'used.jsonl')]


def test_missing_exclusion_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='nope.jsonl'):
        load_exclusions([tmp_path / 'nope.jsonl'])


def test_previous_plan_as_exclusion(selection: Selection, tmp_path: Path) -> None:
    first = plan_cohorts(selection, SPEC)
    first.save(tmp_path / 'first.json')
    second = plan_cohorts(selection, CohortSpec(seed=9, train_per_type=5), load_exclusions([tmp_path / 'first.json']))
    first_ids = {i for ids in first.cohorts.values() for i in ids}
    assert not first_ids & {i for ids in second.cohorts.values() for i in ids}


def test_check_disjoint() -> None:
    with pytest.raises(ValueError, match='Leaking cohorts: a/b'):
        check_disjoint({'a': ['1', '2'], 'b': ['2']})
    with pytest.raises(ValueError, match='previously used'):
        check_disjoint({'a': ['1']}, ['1'])


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match='train_split'):
        CohortSpec(seed=1, train_per_type=1, train_split='dev')
    with pytest.raises(ValueError, match='risk_split'):
        CohortSpec(seed=1, train_per_type=1, risk_split='test')


def test_plan_json_round_trip_and_resolve(selection: Selection, tmp_path: Path) -> None:
    plan = plan_cohorts(selection, SPEC)
    path = plan.save(tmp_path / 'plan.json')
    loaded = CohortPlan.load(path)
    assert loaded == plan
    resolved = loaded.resolve(selection)
    assert {k: [d.id for d in v] for k, v in resolved.items()} == plan.cohorts
    write_json(tmp_path / 'old.json', {**plan.to_json(), 'note': 'kept'})
    assert CohortPlan.load(tmp_path / 'old.json').extra['note'] == 'kept'


def test_plan_from_config_rejects_unknown_keys(selection: Selection) -> None:
    base = {'selection': {'prepared': str(selection.prepared)}, 'cohorts': {'seed': 1, 'train_per_type': 2}}
    assert plan_from_config(base).cohorts['train']
    with pytest.raises(ValueError, match='Unknown cohort config keys'):
        plan_from_config({**base, 'exclusion': []})
    with pytest.raises(TypeError):
        plan_from_config({**base, 'cohorts': {'seed': 1, 'train_per_type': 2, 'per_form': 3}})


def test_cli_never_overwrites_a_plan(selection: Selection, tmp_path: Path) -> None:
    config = tmp_path / 'c.yaml'
    config.write_text(f'selection:\n  prepared: {selection.prepared}\ncohorts:\n  seed: 1\n  train_per_type: 2\n')
    output = tmp_path / 'plan.json'
    assert main(['plan-cohorts', '--config', str(config), '--output', str(output)]) == 0
    before = output.read_bytes()
    assert main(['plan-cohorts', '--config', str(config), '--output', str(output), '--set', 'cohorts.seed=2']) == 1
    assert output.read_bytes() == before


@pytest.mark.skipif(not OLD_REPO.is_dir(), reason='old repository artifacts not available')
def test_reproduces_r4_winner_plan() -> None:
    from l2r4kie.utils.config import load_config

    old = json.loads((OLD_REPO / 'artifacts/token-review-winner/cache/cohort_plan.json').read_text())
    plan = plan_from_config(load_config(R4_CONFIG))
    for key in ('cohorts', 'excluded_documents', 'forms', 'seed', 'train_split', 'source_fingerprint', 'risk_origin'):
        assert plan.to_json()[key] == old[key], key

