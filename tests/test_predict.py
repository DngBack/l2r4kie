"""Tests for ``l2r4kie.pipelines.predict``: rows, provenance and resume.

The decode itself is tested in ``test_decode.py``; here ``extract`` is
replaced by a fake that answers from a table and records its calls.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from l2r4kie.cli import main
from l2r4kie.data.types import Document, FieldRequest, FieldSpec
from l2r4kie.model.decode import DecodeResult, Signals
from l2r4kie.pipelines import predict as predict_module
from l2r4kie.pipelines.predict import evaluation_documents, predict
from l2r4kie.utils.io import read_json, read_jsonl, write_json

ANSWERS = {'/name': ('Nguyễn Văn A', 'ok'), '/year': ('1990', 'ok'), '/code': ('(1,2),(3,4)', 'ok'),
           '/note': ('x x', 'truncated')}


def document(id_: str) -> Document:
    fields = (FieldSpec('/name', 'Name', 'Nguyễn Văn A', 'scalar'), FieldSpec('/year', 'Year', 1990, 'scalar'),
              FieldSpec('/code', 'Code', 'AB', 'scalar'), FieldSpec('/note', 'Note', 'n', 'scalar'))
    return Document(id_, id_, id_.split('__')[0], ('page.png',), fields, ('0' * 64,))


DOCUMENTS = [document('f__1'), document('f__2'), document('g__1')]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """Replace ``extract``; the returned list records the field ids of each call."""
    seen: list[tuple[str, ...]] = []

    def fake_extract(extractor: Any, pages: Sequence[str], requests: Sequence[FieldRequest], *,
                     max_value_tokens: int) -> list[DecodeResult]:
        assert max_value_tokens == 7
        seen.append(tuple(r.id for r in requests))
        signals = Signals(None, None, None)  # type: ignore[arg-type]
        return [DecodeResult(r.id, text if status == 'ok' else None, text, status, signals)
                for r in requests for text, status in [ANSWERS[r.id]]]

    monkeypatch.setattr(predict_module, 'extract', fake_extract)
    return seen


EXTRACTOR: Any = SimpleNamespace(config=SimpleNamespace(max_value_tokens=7, max_pixels=100, close='box_end',
                                                        dtype='torch.float32'))


def test_predict_writes_scored_rows_and_metrics(tmp_path: Path, calls: list) -> None:
    metrics = predict(EXTRACTOR, DOCUMENTS, tmp_path, fields_per_document=3, provenance={'adapter': 'a'})
    assert calls == [('/name', '/year', '/code')] * 3
    rows = read_jsonl(tmp_path / 'predictions.jsonl')
    assert [r['document_id'] for r in rows] == ['f__1'] * 3 + ['f__2'] * 3 + ['g__1'] * 3
    assert rows[1] == {'document_id': 'f__1', 'form': 'f', 'field_id': '/year', 'kind': 'scalar', 'target': 1990,
                       'prediction': '1990', 'text': '1990', 'status': 'ok', 'correct': True}
    assert metrics == read_json(tmp_path / 'metrics.json')
    assert metrics['exact_match'] == 2 / 3 and metrics['coordinate_rate'] == 1 / 3
    provenance = read_json(tmp_path / 'provenance.json')
    assert provenance['documents'] == ['f__1', 'f__2', 'g__1'] and provenance['max_value_tokens'] == 7


def test_json_comparator_flags(tmp_path: Path, calls: list) -> None:
    predict(EXTRACTOR, DOCUMENTS[:1], tmp_path, fields_per_document=2, comparator='json')
    assert [r['correct'] for r in read_jsonl(tmp_path / 'predictions.jsonl')] == [True, False]


def test_resume_redoes_only_incomplete_documents(tmp_path: Path, calls: list) -> None:
    first = predict(EXTRACTOR, DOCUMENTS, tmp_path, fields_per_document=4)
    predictions = tmp_path / 'predictions.jsonl'
    lines = predictions.read_text(encoding='utf-8').splitlines(keepends=True)
    # Crash while writing f__2: two of its rows made it, the second torn in half.
    predictions.write_text(''.join(lines[:5]) + lines[5][:10], encoding='utf-8')
    calls.clear()
    second = predict(EXTRACTOR, DOCUMENTS, tmp_path, fields_per_document=4)
    assert calls == [('/name', '/year', '/code', '/note')] * 2
    rows = read_jsonl(predictions)
    assert [r['document_id'] for r in rows[::4]] == ['f__1', 'f__2', 'g__1'] and len(rows) == 12
    assert second == first and second['truncated_rate'] == 1 / 4
    calls.clear()
    predict(EXTRACTOR, DOCUMENTS, tmp_path, fields_per_document=4)
    assert calls == []


def test_a_different_run_cannot_reuse_the_directory(tmp_path: Path, calls: list) -> None:
    predict(EXTRACTOR, DOCUMENTS[:1], tmp_path, fields_per_document=2, provenance={'adapter': 'a'})
    with pytest.raises(ValueError, match=r"\['adapter'\]"):
        predict(EXTRACTOR, DOCUMENTS[:1], tmp_path, fields_per_document=2, provenance={'adapter': 'b'})
    with pytest.raises(ValueError, match='fields_per_document'):
        predict(EXTRACTOR, DOCUMENTS[:1], tmp_path, fields_per_document=3, provenance={'adapter': 'a'})
    other = tmp_path / 'other'
    other.mkdir()
    (other / 'predictions.jsonl').write_text(json.dumps({'document_id': 'f__1'}) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='without provenance'):
        predict(EXTRACTOR, DOCUMENTS[:1], other)


def test_evaluation_documents_follow_the_file_order(tmp_path: Path) -> None:
    selection: Any = SimpleNamespace(documents=lambda split, limit=None: DOCUMENTS[:limit])
    plan = tmp_path / 'plan.json'
    write_json(plan, {'documents': ['g__1', 'f__1']})
    assert [d.id for d in evaluation_documents(selection, 'dev', plan)] == ['g__1', 'f__1']
    assert [d.id for d in evaluation_documents(selection, 'dev', plan, limit=1)] == ['g__1']
    assert [d.id for d in evaluation_documents(selection, 'dev', limit=2)] == ['f__1', 'f__2']
    write_json(plan, ['f__1', 'h__9'])
    with pytest.raises(KeyError, match='1 documents'):
        evaluation_documents(selection, 'dev', plan)


def test_cli_rescores_and_compares_prediction_files(tmp_path: Path, calls: list,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    predict(EXTRACTOR, DOCUMENTS, tmp_path / 'b', fields_per_document=2)
    baseline = [dict(r, prediction='?') if r['field_id'] == '/year' else r
                for r in read_jsonl(tmp_path / 'b' / 'predictions.jsonl')]
    (tmp_path / 'a.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in baseline), encoding='utf-8')
    capsys.readouterr()
    assert main(['evaluate-file', '--predictions', str(tmp_path / 'a.jsonl'), '--output',
                 str(tmp_path / 'm.json')]) == 0
    assert json.loads(capsys.readouterr().out)['exact_match'] == 1 / 2
    assert read_json(tmp_path / 'm.json')['by_form']['f']['count'] == 4
    assert main(['compare', '--a', str(tmp_path / 'a.jsonl'), '--b', str(tmp_path / 'b' / 'predictions.jsonl'),
                 '--draws', '50', '--kind', 'scalar']) == 0
    result = json.loads(capsys.readouterr().out)
    assert (result['baseline_exact_match'], result['candidate_exact_match'], result['delta']) == (0.5, 1.0, 0.5)
