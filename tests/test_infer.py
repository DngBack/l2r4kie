"""Tests for ``l2r4kie.pipelines.infer``: request validation and response schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from l2r4kie.cli import main
from l2r4kie.data.types import FieldRequest
from l2r4kie.pipelines.infer import build_response, parse_request


@pytest.fixture
def page(tmp_path: Path) -> Path:
    path = tmp_path / 'page.png'
    path.write_bytes(b'not read by parse_request')
    return path


def test_parse_request_defaults_and_relative_pages(page: Path) -> None:
    pages, requests = parse_request({'pages': ['page.png'], 'fields': [
        {'id': '/a'}, {'id': '/b', 'description': 'B', 'kind': 'array'}]}, root=page.parent)
    assert pages == [page]
    assert requests == [FieldRequest('/a', '/a'), FieldRequest('/b', 'B', 'array')]


@pytest.mark.parametrize(('request_', 'error'), [
    ({'fields': [{'id': '/a'}]}, 'pages'),
    ({'pages': ['page.png'], 'fields': []}, 'fields'),
    ({'pages': ['page.png'], 'fields': [{'description': 'no id'}]}, '"id"'),
    ({'pages': ['page.png'], 'fields': [{'id': '/a', 'kind': 'number'}]}, 'kind'),
    ({'pages': ['page.png'], 'fields': [{'id': '/a'}, {'id': '/a'}]}, 'Duplicate'),
])
def test_parse_request_rejects_malformed_requests(page: Path, request_: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_request(request_, root=page.parent)


def test_parse_request_reports_missing_pages(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='missing.png'):
        parse_request({'pages': ['missing.png'], 'fields': [{'id': '/a'}]}, root=tmp_path)


def test_response_keeps_the_old_schema() -> None:
    from l2r4kie.model.decode import DecodeResult, Signals

    signals = Signals(*([None] * 3))  # type: ignore[arg-type]
    response = build_response([DecodeResult('/a', 'x', 'x', 'ok', signals),
                               DecodeResult('/b', None, '(1,2', 'truncated', signals)])
    assert response == {'result': {'/a': 'x', '/b': None}, 'confidence': {'/a': None, '/b': None},
                        'status': {'/a': 'ok', '/b': 'truncated'}, 'calibrated': False}
    json.dumps(response)


def test_cli_infer_has_device_and_precision_options() -> None:
    with pytest.raises(SystemExit) as exit_:
        main(['infer', '--help'])
    assert exit_.value.code == 0
