"""Test page for the fine-tuned extractors (Qwen3.5-4B / 9B trained on artifacts/kie-real).

Upload page images, or pick a held-out test document (its ground truth is then
scored), give the fields, and see the assembled JSON, per-field values and the
timings of the request. Every request is appended to
``artifacts/kie-web/requests.jsonl`` (timings, tokens, statuses; uploads and
full responses are kept next to it); the Log tab summarises them per model.

Fields can be given as

* a Data/kie prompt (lines ``- "Họ và tên": ...``, ``- "Thuốc" (mảng object): ...``),
* a JSON description tree (``{"Họ và tên": "...", "Thuốc": [{"__desc": "..."}]}``), or
* a JSON list (``[{"id": "/so_tien", "description": "...", "kind": "scalar"}]``).

The 4B model is the default and is loaded at start. Its adapter is the final
one when training has finished, else the latest snapshot; a newer adapter is
picked up at the next request. Only stdlib HTTP (no web framework)::

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \\
        /home/jovyan/scratch/home-migrate/bachdx2/venvs/l2r4kie-q35/bin/python scripts/kie_web.py --port 7860
"""

from __future__ import annotations

import argparse
import base64
import binascii
import gc
import io
import json
import statistics
import threading
import time
import traceback
import uuid
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch
from PIL import Image

from l2r4kie.data.conversations import field_descriptions
from l2r4kie.data.schema import iter_branches
from l2r4kie.data.types import Document, FieldRequest
from l2r4kie.eval.comparator import is_correct
from l2r4kie.model.decode import decode_prefix
from l2r4kie.model.extractor import Extractor, ExtractorConfig

REPO = Path(__file__).resolve().parents[1]
RUNS = {'4b': ('Qwen/Qwen3.5-4B', REPO / 'artifacts/kev-real-4b'),
        '9b': ('Qwen/Qwen3.5-9B', REPO / 'artifacts/kev-real-9b')}
TEST = REPO / 'artifacts/kie-real/test.jsonl'
PAGE = Path(__file__).with_name('kie_web.html')
MAX_VALUE_TOKENS = 4096  # the evaluation decode budget (and the training value budget)


# ------------------------------------------------------------------ models

def resolve_adapter(run: Path) -> tuple[Path, str]:
    """The run's final adapter, else its latest snapshot's, with a label."""
    if (run / 'adapter').is_dir():
        summary = run / 'training_summary.json'
        steps = json.loads(summary.read_text()).get('steps') if summary.is_file() else None
        return run / 'adapter', f'final{f" ({steps} steps)" if steps else ""}'
    latest = run / 'latest_training.json'
    if not latest.is_file():
        raise FileNotFoundError(f'{run} has no adapter yet')
    record = json.loads(latest.read_text())
    return run / record['path'] / 'adapter', f'snapshot step {record["step"]} (training not finished)'


def is_final(adapter: Path) -> bool:
    """Whether ``adapter`` is a run's final adapter (not a snapshot's)."""
    return adapter.parent.parent.name != 'snapshots'


class Models:
    """Loaded extractors keyed by (model, merge), at most ``keep`` at a time (least recently used dropped)."""

    def __init__(self, device: str, keep: int = 2) -> None:
        self.device, self.keep = device, keep
        self.loaded: OrderedDict[tuple[str, bool], dict[str, Any]] = OrderedDict()

    def get(self, name: str, merge: bool) -> tuple[dict[str, Any], float]:
        """The extractor for ``name`` and the seconds spent loading it now (0 when cached)."""
        model, run = RUNS[name]
        adapter, label = resolve_adapter(run)
        key = (name, merge)
        entry = self.loaded.get(key)
        # Switch only to a final adapter: while training runs, a snapshot appears every few minutes.
        if entry is not None and (entry['adapter'] == adapter or not is_final(adapter)):
            self.loaded.move_to_end(key)
            return entry, 0.0
        if entry is not None:  # training finished since this one was loaded
            del self.loaded[key]
        while len(self.loaded) >= self.keep:
            self.loaded.popitem(last=False)
        gc.collect()
        torch.cuda.empty_cache()
        trained = json.loads((run / 'config.json').read_text())
        config = ExtractorConfig(model=model, device=self.device, max_pixels=trained['format']['max_pixels'],
                                 max_value_tokens=MAX_VALUE_TOKENS, close=trained['format']['close'],
                                 merge_adapter=merge)
        start = time.perf_counter()
        extractor = Extractor.load(config, adapter)
        seconds = time.perf_counter() - start
        entry = {'extractor': extractor, 'adapter': adapter, 'label': label, 'model': model}
        self.loaded[key] = entry
        return entry, seconds

    def status(self) -> dict[str, Any]:
        out = {}
        for name, (model, run) in RUNS.items():
            try:
                adapter, label = resolve_adapter(run)
            except FileNotFoundError as error:
                out[name] = {'model': model, 'available': False, 'label': str(error)}
                continue
            loaded = [{'merge': merge, 'label': e['label'], 'stale': e['adapter'] != adapter and is_final(adapter)}
                      for (n, merge), e in self.loaded.items() if n == name]
            out[name] = {'model': model, 'available': True, 'label': label, 'adapter': str(adapter.relative_to(REPO)),
                         'loaded': loaded}
        return out


# ------------------------------------------------------------------ fields

def _skeleton(tree: Any) -> Any:
    """A label-shaped value for a description tree: objects recurse, arrays are [], the rest ''."""
    if isinstance(tree, dict):
        return {k: _skeleton(v) for k, v in tree.items() if k != '__desc'}
    return [] if isinstance(tree, list) else ''


def parse_fields(text: str) -> list[FieldRequest]:
    """Field requests from a Data/kie prompt, a JSON description tree or a JSON list (see the module doc)."""
    text = text.strip()
    if not text:
        raise ValueError('No fields given')
    if text[0] in '[{':
        spec = json.loads(text)
        if isinstance(spec, list):
            requests = []
            for item in spec:
                if not isinstance(item, dict) or not str(item.get('id', '')).strip():
                    raise ValueError(f'Each field needs an "id": {item!r}')
                # Training ids are JSON pointers ('/a/b'); the id is part of the prompt.
                fid = item['id'] if item['id'].startswith('/') else f'/{item["id"]}'
                kind = item.get('kind', 'scalar')
                if kind not in ('scalar', 'array'):
                    raise ValueError(f'{fid}: kind must be scalar or array')
                requests.append(FieldRequest(fid, str(item.get('description') or fid), kind))
        elif isinstance(spec, dict):
            requests = [FieldRequest(f.id, f.description, f.kind) for f in iter_branches(_skeleton(spec), spec)]
        else:
            raise ValueError('JSON fields must be an object or a list')
    else:
        tree = field_descriptions(text)
        if tree is None:
            raise ValueError('No field list found: use lines like - "Họ và tên": description')
        requests = [FieldRequest(f.id, f.description, f.kind) for f in iter_branches(_skeleton(tree), tree)]
    ids = [r.id for r in requests]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate field ids: {sorted({i for i in ids if ids.count(i) > 1})}')
    if not requests:
        raise ValueError('No fields given')
    return requests


def assemble(values: dict[str, Any]) -> dict[str, Any]:
    """Nested JSON from pointer ids ('/a/b' → {"a": {"b": ...}})."""
    root: dict[str, Any] = {}
    for fid, value in values.items():
        *parents, leaf = [p.replace('~1', '/').replace('~0', '~') for p in fid.lstrip('/').split('/')]
        node = root
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return root


# ------------------------------------------------------------------ app

class App:
    def __init__(self, device: str, out: Path) -> None:
        self.models = Models(device)
        self.device = device
        self.out = out
        (out / 'uploads').mkdir(parents=True, exist_ok=True)
        (out / 'responses').mkdir(parents=True, exist_ok=True)
        self.log_path = out / 'requests.jsonl'
        self.gpu = threading.Lock()
        self.log_lock = threading.Lock()
        self.busy: str | None = None
        self.test = {d['id']: d for d in map(json.loads, TEST.open())} if TEST.is_file() else {}

    def preload(self, name: str) -> None:
        with self.gpu:
            self.busy = f'loading {name.upper()}'
            try:
                self.models.get(name, False)
            except Exception:  # noqa: BLE001 - shown in the status line, the server keeps running
                traceback.print_exc()
            finally:
                self.busy = None

    def save_uploads(self, rid: str, pages: list[str]) -> list[Path]:
        paths = []
        for i, data in enumerate(pages):
            try:
                raw = base64.b64decode(data.split(',', 1)[-1], validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValueError(f'Page {i + 1} is not base64 image data') from error
            with Image.open(io.BytesIO(raw)) as image:
                fmt = (image.format or 'png').lower()
                image.verify()
            path = self.out / 'uploads' / rid / f'page{i + 1}.{fmt}'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            paths.append(path)
        return paths

    def run(self, body: dict[str, Any]) -> dict[str, Any]:
        received = time.perf_counter()
        rid = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
        name, merge = body.get('model', '4b'), bool(body.get('merge'))
        if name not in RUNS:
            raise ValueError(f'Unknown model {name!r}')
        requests = parse_fields(body.get('fields', ''))
        document = self.test.get(body.get('doc_id') or '')
        if document is not None:
            pages = [Path(p) for p in document['pages']]
            truth = {f['id']: f['value'] for f in document['fields']}
        else:
            if not body.get('pages'):
                raise ValueError('Upload at least one page image, or pick a test document')
            pages, truth = self.save_uploads(rid, body['pages']), {}
        record: dict[str, Any] = {'id': rid, 'time': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                  'model': name, 'merge': merge,
                                  'source': f'test:{document["id"]}' if document else 'upload',
                                  'pages': len(pages), 'fields': len(requests),
                                  'arrays': sum(r.kind == 'array' for r in requests)}
        with self.gpu:
            waited = time.perf_counter() - received
            self.busy = f'{name.upper()} · {len(pages)} pages · {len(requests)} fields'
            try:
                entry, load_seconds = self.models.get(name, merge)
                extractor: Extractor = entry['extractor']
                cuda = torch.cuda.is_available() and self.device.startswith('cuda')
                if cuda:
                    torch.cuda.reset_peak_memory_stats(self.device)
                start = time.perf_counter()
                prefix = extractor.encode_prefix(pages)
                if cuda:
                    torch.cuda.synchronize(self.device)
                encoded = time.perf_counter()
                results = decode_prefix(extractor, prefix, requests, max_value_tokens=MAX_VALUE_TOKENS)
                if cuda:
                    torch.cuda.synchronize(self.device)
                done = time.perf_counter()
                peak = torch.cuda.max_memory_allocated(self.device) / 2**30 if cuda else None
            finally:
                self.busy = None
        tokenizer = extractor.format.tokenizer
        rows = []
        for request, result in zip(requests, results, strict=True):
            tokens = len(tokenizer(result.text, add_special_tokens=False)['input_ids']) + (result.stop == 'close')
            row = {'id': request.id, 'description': request.description, 'kind': request.kind,
                   'value': result.value, 'text': result.text, 'status': result.status, 'stop': result.stop,
                   'tokens': tokens}
            if request.id in truth:
                row['target'] = truth[request.id]
                row['correct'] = is_correct({'status': result.status, 'prediction': result.value,
                                             'target': truth[request.id]})
            rows.append(row)
        decode_seconds, generated = done - encoded, sum(r['tokens'] for r in rows)
        timings = {'queue': waited, 'load': load_seconds, 'prefix': encoded - start, 'decode': decode_seconds,
                   'infer': done - start, 'total': time.perf_counter() - received}
        scored = [r for r in rows if 'correct' in r]
        record.update({
            'adapter': entry['label'], 'prefix_tokens': int(prefix['input_ids'].shape[1]),
            'timings': {k: round(v, 3) for k, v in timings.items()},
            'generated_tokens': generated, 'longest_field_tokens': max(r['tokens'] for r in rows),
            'tokens_per_s': round(generated / decode_seconds, 2) if decode_seconds else None,
            'fields_per_s': round(len(rows) / timings['infer'], 2),
            'peak_gib': round(peak, 2) if peak is not None else None,
            'status': dict(Counter(r['status'] for r in rows)),
            'exact_match': round(sum(r['correct'] for r in scored) / len(scored), 4) if scored else None,
            'scored_fields': len(scored)})
        response = {'record': record, 'fields': rows,
                    'json': assemble({r['id']: r['value'] for r in rows}),
                    'truth': assemble({r['id']: r['target'] for r in scored}) if scored else None}
        (self.out / 'responses' / f'{rid}.json').write_text(json.dumps(response, ensure_ascii=False, indent=1))
        with self.log_lock, self.log_path.open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        return response

    def log_error(self, body: dict[str, Any], error: Exception) -> None:
        record = {'time': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'model': body.get('model'),
                  'source': f'test:{body["doc_id"]}' if body.get('doc_id') else 'upload',
                  'error': f'{type(error).__name__}: {error}'}
        with self.log_lock, self.log_path.open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    def logs(self, limit: int = 200) -> dict[str, Any]:
        records = [json.loads(line) for line in self.log_path.open()] if self.log_path.is_file() else []
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            if 'timings' in r:
                groups.setdefault(f'{r["model"].upper()}{" merged" if r.get("merge") else ""}', []).append(r)
        summary = []
        for key, rs in sorted(groups.items()):
            infer = [r['timings']['infer'] for r in rs]
            decode = sum(r['timings']['decode'] for r in rs)
            summary.append({'model': key, 'requests': len(rs), 'median_infer_s': round(statistics.median(infer), 2),
                            'mean_infer_s': round(statistics.mean(infer), 2), 'max_infer_s': round(max(infer), 2),
                            'median_prefix_s': round(statistics.median(r['timings']['prefix'] for r in rs), 3),
                            'fields_per_s': round(sum(r['fields'] for r in rs) / sum(infer), 2),
                            'tokens_per_s': round(sum(r['generated_tokens'] for r in rs) / decode, 1) if decode else None})
        return {'summary': summary, 'records': records[-limit:][::-1]}


# ------------------------------------------------------------------ http

def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quieter: no successful requests
            if len(args) == 3 and str(args[1]).startswith(('2', '3')):
                return
            super().log_message(fmt, *args)

        def send(self, code: int, payload: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def json(self, value: Any, code: int = 200) -> None:
            self.send(code, json.dumps(value, ensure_ascii=False).encode(), 'application/json; charset=utf-8')

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            path = url.path.rstrip('/').rsplit('/', 1)[-1] if url.path.strip('/') else ''
            if path in ('', 'index.html'):
                self.send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
            elif path == 'status':
                self.json({'busy': app.busy, 'models': app.models.status(), 'device': app.device})
            elif path == 'datasets':
                self.json(sorted(Counter(d['form'] for d in app.test.values()).items()))
            elif path == 'samples':
                self.json([{'id': d['id'], 'pages': len(d['pages']), 'fields': len(d['fields'])}
                           for d in app.test.values() if d['form'] == query.get('dataset')])
            elif path == 'doc' and query.get('id') in app.test:
                d = app.test[query['id']]
                self.json({'id': d['id'], 'form': d['form'], 'pages': len(d['pages']),
                           'fields': [{'id': f['id'], 'description': f['description'], 'kind': f['kind']}
                                      for f in d['fields']]})
            elif path == 'page' and query.get('doc') in app.test:
                pages = app.test[query['doc']]['pages']
                i = int(query.get('i', 0))
                if not 0 <= i < len(pages):
                    return self.json({'error': 'no such page'}, 404)
                with Image.open(pages[i]) as image:  # downscaled preview
                    image = image.convert('RGB')
                    image.thumbnail((1400, 1400))
                    buffer = io.BytesIO()
                    image.save(buffer, 'JPEG', quality=85)
                self.send(200, buffer.getvalue(), 'image/jpeg')
            elif path == 'logs':
                self.json(app.logs())
            else:
                self.json({'error': 'not found'}, 404)

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.rstrip('/').endswith('/run'):
                return self.json({'error': 'not found'}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            except json.JSONDecodeError:
                return self.json({'error': 'request body is not JSON'}, 400)
            try:
                self.json(app.run(body))
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
                app.log_error(body, error)
                self.json({'error': str(error)}, 400)
            except Exception as error:  # noqa: BLE001 - reported to the page and logged
                traceback.print_exc()
                app.log_error(body, error)
                if isinstance(error, torch.OutOfMemoryError):
                    torch.cuda.empty_cache()
                self.json({'error': f'{type(error).__name__}: {error}'}, 500)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default=str(REPO / 'artifacts/kie-web'), help='request log, uploads, responses')
    parser.add_argument('--preload', default='4b', help='model to load at start ("" for none)')
    args = parser.parse_args()
    app = App(args.device, Path(args.output))
    if args.preload:
        threading.Thread(target=app.preload, args=(args.preload,), daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f'serving on http://{args.host}:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
