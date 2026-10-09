"""Tests for ``l2r4kie.data``: label flattening, prepare, selection and stats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from l2r4kie.cli import main
from l2r4kie.data.prepare import assign_split, group_duplicates, prepare
from l2r4kie.data.schema import escape_pointer, iter_branches
from l2r4kie.data.selection import Selection, interleave_forms
from l2r4kie.data.stats import document_stats, string_features, value_type
from l2r4kie.data.types import Document, FieldSpec
from l2r4kie.utils.io import read_json, read_jsonl, write_jsonl

OLD_PREPARED = Path('/home/jovyan/bachdx2/l2r4kie/artifacts/data')
RAW_DATA = Path('/home/jovyan/bachdx2/data/kie-all-v1')


# ----------------------------------------------------------------------- helpers

def make_raw(root: Path, docs: list[dict[str, Any]]) -> Path:
    """Write a tiny raw dataset.

    Each doc: ``{'form', 'sample', 'pages': {name: bytes}, 'label': {...},
    'missing': [page names not to write]}``.
    """
    for folder in ('images', 'kie-labels', 'schemas'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    rows = []
    for doc in docs:
        for name, content in doc['pages'].items():
            if name not in doc.get('missing', ()):
                (root / 'images' / name).write_bytes(content)
        label_file = f"{doc['form']}__{doc['sample']}.fields.json"
        (root / 'kie-labels' / label_file).write_text(json.dumps(doc['label']))
        rows.append({'form': doc['form'], 'sample': doc['sample'], 'pages': list(doc['pages']), 'fields': label_file})
    write_jsonl(root / 'manifest.jsonl', rows)
    return root


def doc(id: str, form: str = 'f', group_id: str | None = None) -> Document:
    return Document(id, group_id or id, form, (), (FieldSpec('/a', '/a', 'x', 'scalar'),), ())


# ------------------------------------------------------------------------ schema

def test_escape_pointer() -> None:
    assert escape_pointer('a/b~c') == 'a~1b~0c'
    assert escape_pointer('~1') == '~01'  # '~' first, so a literal '~1' round-trips


def test_iter_branches_flattens_objects_and_keeps_arrays_atomic() -> None:
    label = {'a/b': {'x~y': '01'}, 'rows': [{'name': 'a'}, {'name': 'b'}], 'ok': True}
    descriptions = {'a/b': {'x~y': 'Mã số'}, 'rows': 'Danh sách', 'ok': {'unexpected': 'tree'}}
    assert list(iter_branches(label, descriptions)) == [
        FieldSpec('/a~1b/x~0y', 'Mã số', '01', 'scalar'),
        FieldSpec('/rows', 'Danh sách', [{'name': 'a'}, {'name': 'b'}], 'array'),
        FieldSpec('/ok', '/ok', True, 'scalar'),  # non-string description falls back to the path
    ]


def test_document_json_round_trip_keeps_key_order() -> None:
    record = {'id': 'f__1', 'group_id': 'f__1', 'form': 'f', 'pages': ['/p.png'],
              'fields': [{'id': '/a', 'description': 'A', 'value': ['x'], 'kind': 'array'}], 'image_sha256': ['ab']}
    document = Document.from_json(record)
    assert document.fields[0].kind == 'array'
    assert json.dumps(document.to_json()) == json.dumps(record)


# ----------------------------------------------------------------------- prepare

def test_duplicate_images_cannot_cross_splits(tmp_path: Path) -> None:
    raw = make_raw(tmp_path / 'raw', [
        {'form': 'lift-f', 'sample': str(i), 'pages': {f'{i}.png': b'same-page'}, 'label': {'fields': {'a': 'b'}}}
        for i in range(8)
    ])
    output = tmp_path / 'derived'
    report = prepare(raw, output)
    nonempty = [s for s in ('train', 'dev', 'calibration', 'test') if read_jsonl(output / f'{s}.jsonl')]
    assert len(nonempty) == 1
    assert report['documents'] == 8
    assert report['duplicate_page_occurrences'] == 7
    assert {r['group_id'] for r in read_jsonl(output / f'{nonempty[0]}.jsonl')} == {'lift-f__0'}


def test_group_duplicates_is_transitive_through_multi_page_documents() -> None:
    manifest = [{'form': 'f', 'sample': 'c', 'pages': ['p1']},
                {'form': 'f', 'sample': 'b', 'pages': ['p2', 'p3']},  # bridges a and c
                {'form': 'f', 'sample': 'a', 'pages': ['p4']},
                {'form': 'f', 'sample': 'z', 'pages': ['p5']}]
    hashes = {'p1': 'h1', 'p2': 'h1', 'p3': 'h2', 'p4': 'h2', 'p5': 'h5'}
    assert group_duplicates(manifest, hashes) == ['f__a', 'f__a', 'f__a', 'f__z']


def test_rejections_unprinted_fields_and_descriptions(tmp_path: Path) -> None:
    raw = make_raw(tmp_path / 'raw', [
        {'form': 'lift-f', 'sample': 'ok', 'pages': {'ok.png': b'1'},
         'label': {'fields': {'a': 'x', 'b': 'y'}, 'unprinted': ['/b']}},
        {'form': 'lift-f', 'sample': 'nopage', 'pages': {'np.png': b'2'}, 'missing': ['np.png'],
         'label': {'fields': {'a': 'x'}}},
        {'form': 'lift-f', 'sample': 'empty', 'pages': {'e.png': b'3'},
         'label': {'fields': {'a': 'x'}, 'unprinted': ['/a']}},
        {'form': 'lift-f', 'sample': 'badlabel', 'pages': {'b.png': b'4'}, 'label': {'no_fields': {}}},
    ])
    (raw / 'schemas' / 'f.descriptions.json').write_text(json.dumps({'a': 'Trường A'}))
    report = prepare(raw, tmp_path / 'out')
    assert report['documents'] == 1
    assert {r['id']: r['reason'] for r in report['rejected']} == {
        'lift-f__nopage': 'Missing image', 'lift-f__empty': 'No usable fields', 'lift-f__badlabel': "'fields'"}
    [record] = [r for s in ('train', 'dev', 'calibration', 'test') for r in read_jsonl(tmp_path / 'out' / f'{s}.jsonl')]
    assert record['fields'] == [{'id': '/a', 'description': 'Trường A', 'value': 'x', 'kind': 'scalar'}]
    assert record['pages'] == [str((raw / 'images' / 'ok.png').resolve())]
    assert read_json(tmp_path / 'out' / 'report.json') == report


def test_prepare_refuses_output_inside_source(tmp_path: Path) -> None:
    raw = make_raw(tmp_path / 'raw', [])
    with pytest.raises(ValueError, match='outside'):
        prepare(raw, raw / 'derived')


def test_assign_split_proportions_and_seed() -> None:
    splits = [assign_split(f'g{i}', 42) for i in range(5000)]
    shares = {s: splits.count(s) / len(splits) for s in ('train', 'dev', 'calibration', 'test')}
    assert shares['train'] == pytest.approx(0.80, abs=0.02)
    assert shares['test'] == pytest.approx(0.06, abs=0.015)
    assert splits != [assign_split(f'g{i}', 43) for i in range(5000)]


@pytest.mark.skipif(not (RAW_DATA.is_dir() and OLD_PREPARED.is_dir()), reason='real dataset not available')
def test_prepare_reproduces_old_splits_byte_for_byte(tmp_path: Path) -> None:
    prepare(RAW_DATA, tmp_path)
    for name in ('train.jsonl', 'dev.jsonl', 'calibration.jsonl', 'test.jsonl', 'report.json'):
        assert (tmp_path / name).read_bytes() == (OLD_PREPARED / name).read_bytes(), name


# --------------------------------------------------------------------- selection

def write_prepared(root: Path, splits: dict[str, list[Document]]) -> Path:
    for split in ('train', 'dev', 'calibration', 'test'):
        write_jsonl(root / f'{split}.jsonl', (d.to_json() for d in splits.get(split, [])))
    return root


def test_train_holdout_is_a_disjoint_partition(tmp_path: Path) -> None:
    prepared = write_prepared(tmp_path, {'train': [doc(str(i)) for i in range(400)]})
    selection = Selection(prepared, holdout_percent=15)
    fit = {d.id for d in selection.documents('train')}
    reserve = {d.id for d in selection.documents('train_reserve')}
    assert not fit & reserve and len(fit | reserve) == 400 and 30 < len(reserve) < 90
    assert len(Selection(prepared).documents('train')) == 400
    assert Selection(prepared).documents('train_reserve') == []


def test_holdout_is_per_group(tmp_path: Path) -> None:
    prepared = write_prepared(tmp_path, {'train': [doc(f'{g}-{k}', group_id=str(g)) for g in range(100) for k in range(3)]})
    reserve = Selection(prepared, holdout_percent=30).documents('train_reserve')
    groups = {d.group_id for d in reserve}
    assert len(reserve) == 3 * len(groups)  # all members of a reserved group move together


def test_selection_filters_shuffles_balances_and_limits(tmp_path: Path) -> None:
    docs = [doc(f'a{i}', 'a') for i in range(6)] + [doc(f'b{i}', 'b') for i in range(2)] + [doc('c0', 'c')]
    prepared = write_prepared(tmp_path, {'train': docs, 'dev': docs})
    balanced = Selection(prepared, seed=1, balanced_forms=True)
    order = [d.form for d in balanced.documents('train')]
    assert sorted(order[:3]) == ['a', 'b', 'c'] and order[-4:] == ['a'] * 4
    assert balanced.documents('train', limit=2) == balanced.documents('train')[:2]
    assert {d.form for d in Selection(prepared, forms=('b',)).documents('dev')} == {'b'}
    assert Selection(prepared, seed=1).documents('dev') == Selection(prepared, seed=1).documents('dev')
    assert Selection(prepared, seed=1).documents('dev') != Selection(prepared, seed=2).documents('dev')


def test_interleave_forms() -> None:
    docs = [doc('a1', 'a'), doc('a2', 'a'), doc('b1', 'b'), doc('a3', 'a'), doc('c1', 'c')]
    assert [d.id for d in interleave_forms(docs)] == ['a1', 'b1', 'c1', 'a2', 'a3']


def test_selection_from_config_ignores_other_keys(tmp_path: Path) -> None:
    config = {'prepared': str(tmp_path), 'seed': 7, 'holdout_percent': 15, 'forms': [], 'lr': 1e-4}
    assert Selection.from_config(config) == Selection(tmp_path, seed=7, holdout_percent=15)


# ------------------------------------------------------------------------- stats

@pytest.mark.parametrize(('value', 'expected'), [
    ('x', 'string'), ('', 'empty_string'), (True, 'bool'), (None, 'null'), (3, 'number'),
    ([], 'empty_array'), ([{'a': 1}], 'array_of_objects'), (['a'], 'array_of_scalars'), ({}, 'object'),
])
def test_value_type(value: Any, expected: str) -> None:
    assert value_type(value) == expected


def test_string_features() -> None:
    assert string_features('0123') == ['leading_zero_digits']
    assert string_features('0') == []
    assert string_features(' a\nb') == ['multiline', 'surrounding_whitespace']
    assert string_features('x <|box_end|>') == ['angle_brackets', 'special_token_like']


def test_document_stats_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stats = document_stats([doc('a', 'f'), doc('b', 'g')])
    assert stats == {'documents': 2, 'pages': 0, 'fields': 2, 'forms': 2,
                     'value_types': {'string': 2}, 'string_features': {}}
    prepared = write_prepared(tmp_path, {'dev': [doc('a')]})
    assert main(['data-stats', '--prepared', str(prepared)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['splits']['dev']['documents'] == 1 and printed['total']['documents'] == 1


def test_cli_prepare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = make_raw(tmp_path / 'raw', [
        {'form': 'f', 'sample': '1', 'pages': {'1.png': b'1'}, 'label': {'fields': {'a': 'x'}}}])
    assert main(['prepare', '--data', str(raw), '--output', str(tmp_path / 'out')]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['documents'] == 1 and 'forms' not in printed
