"""Extractor training loop: LoRA on Qwen2-VL with branch-packed teacher forcing.

One step is one document: its prefix (all pages) plus ``fields_per_document``
sampled field branches, packed with block-causal isolation
(:func:`~l2r4kie.model.packing.pack`) and trained with cross-entropy on value
tokens and the close marker (:func:`~l2r4kie.train.losses.value_loss`).

Outputs in the run directory:

* ``config.json``: the run config;
* ``train.jsonl``: one record per step (loss, accuracies, lr, memory, time);
* ``monitor.jsonl``: periodic greedy decoding of a few documents
  (exact match, coordinate rate, truncation rate), when enabled;
* ``snapshots/`` + ``latest_training.json``: resumable state;
* ``adapter/``: the final adapter; ``training_summary.json``.

Changes from the old ``application.train``: no confidence head (it was in the
optimizer and got pointless weight decay), no synthetic negatives (their
generation consumed the field RNG), and the device comes from the config.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, TextIO

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..data.selection import Selection
from ..data.types import Document, FieldRequest
from ..eval.comparator import correct
from ..eval.errors import is_coordinates
from ..model.decode import extract
from ..model.extractor import Extractor
from ..model.format import EncodedBranch, KevFormat, load_processor
from ..model.packing import pack, prefix_positions
from ..utils.io import read_jsonl, write_json
from ..utils.seed import seed_everything
from .config import TrainConfig
from .losses import value_loss
from .schedule import lr_multiplier
from .snapshots import latest_snapshot, restore_state, save_adapter, save_snapshot


def build_model(config: TrainConfig, adapter: Path | str | None = None, base: torch.nn.Module | None = None,
                processor: Any = None) -> Extractor:
    """Wrap the base model with a trainable LoRA adapter.

    Args:
        config: Run config.
        adapter: Existing adapter directory to continue training (a snapshot's
            ``adapter/`` or ``init_adapter``); ``None`` creates a fresh LoRA.
        base: Base ``Qwen2VLForConditionalGeneration``; loaded from
            ``config.model`` when ``None`` (tests pass a tiny model).
        processor: Its processor; loaded when ``None``.
    """
    from peft import LoraConfig, PeftModel, get_peft_model

    settings = config.extractor()
    processor = processor if processor is not None else load_processor(config.model, config.format.max_pixels)
    fmt = KevFormat(processor.tokenizer, config.format.close)
    if base is None:
        from transformers import Qwen2VLForConditionalGeneration

        base = Qwen2VLForConditionalGeneration.from_pretrained(config.model, dtype=settings.dtype,
                                                               attn_implementation='sdpa')
    if adapter is not None:
        path = Path(adapter)
        model = PeftModel.from_pretrained(base, str(path / 'adapter' if (path / 'adapter').is_dir() else path),
                                          is_trainable=True)
    else:
        m = fmt.markers
        markers = [m.key_open, m.key_close, m.value_open, m.value_close] if config.trainable_markers else None
        lora = config.lora
        model = get_peft_model(base, LoraConfig(
            r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout, target_modules=list(lora.target_modules),
            bias='none', task_type='CAUSAL_LM', trainable_token_indices=markers))
    if config.gradient_checkpointing:
        # Same gradients; keeps one layer of dense-mask attention in memory instead of all.
        model.get_base_model().gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.to(settings.device)
    return Extractor(model, processor, fmt, settings)


class Trainer:
    """Runs (or resumes) one training run.

    Args:
        config: Run config.
        base: Base model to adapt instead of loading ``config.model``.
        processor: Processor to use instead of loading one.

    Raises:
        ValueError: If ``output`` holds a run without a resumable snapshot,
            the snapshot's config differs, or no training document exists.
    """

    def __init__(self, config: TrainConfig, base: torch.nn.Module | None = None, processor: Any = None) -> None:
        self.config = config
        self.output = Path(config.output)
        self.snapshot = latest_snapshot(self.output, config) if config.resume else None
        if (self.output / 'train.jsonl').exists() and self.snapshot is None:
            raise ValueError(f'{self.output} already holds a run without a resumable snapshot; '
                             'use a new output directory')
        seed_everything(config.seed)
        self.field_rng = random.Random(config.seed)
        self.extractor = build_model(config, self.snapshot or config.init_adapter, base, processor)
        self.model = self.extractor.model
        self.dtype = config.extractor().dtype
        selection = Selection(Path(config.selection.prepared), config.selection.seed,
                              config.selection.holdout_percent, config.selection.forms,
                              config.selection.balanced_forms)
        self.documents = selection.documents('train', config.train_documents)
        if not self.documents:
            raise ValueError('No training documents selected')
        self.monitor_documents: dict[str, list[Document]] = {
            split: (self.documents if split == 'train' else selection.documents('dev'))[:config.monitor.documents]
            for split in config.monitor.splits}
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda u: lr_multiplier(u, config.total_updates, config.warmup_updates, config.cosine))
        self.first_step, previous_seconds = 0, 0.0
        if self.snapshot is not None:
            restored = restore_state(self.snapshot, self.optimizer, self.scheduler, self.field_rng,
                                     self.extractor.device)
            self.first_step, previous_seconds = restored.step, restored.seconds
        self.start = time.monotonic() - previous_seconds

    # ----------------------------------------------------------------- steps

    def fields_for(self, step: int) -> tuple[Document, list[EncodedBranch]]:
        """Document of ``step`` and its sampled branches (targets within ``max_value_tokens``)."""
        document = self.documents[step % len(self.documents)]
        fmt, limit = self.extractor.format, self.config.format.max_value_tokens
        branches = [b for b in map(fmt.encode, document.fields) if len(b.target) <= limit]
        if not branches:
            raise ValueError(f'No field of {document.id} fits max_value_tokens={limit}')
        return document, self.field_rng.sample(branches, min(self.config.fields_per_document, len(branches)))

    def is_update(self, step: int) -> bool:
        """Whether the optimizer steps after ``step`` (0-based)."""
        return (step + 1) % self.config.gradient_accumulation == 0 or step + 1 == self.config.steps

    def due(self, step: int, every: int) -> bool:
        """Whether a periodic action (every ``every`` steps) falls on the update after ``step``.

        Actions run at update boundaries only; with accumulation the boundary
        that first reaches a multiple of ``every`` takes it.
        """
        return bool(every) and self.is_update(step) and (step + 1) % every < self.config.gradient_accumulation

    def train_step(self, step: int) -> dict[str, Any]:
        """Forward/backward one document; at an update boundary also step the optimizer.

        Returns:
            The ``train.jsonl`` record of the step.

        Raises:
            FloatingPointError: On a non-finite loss or gradient norm.
        """
        config, extractor = self.config, self.extractor
        cuda = extractor.device.type == 'cuda'
        if cuda:
            torch.cuda.reset_peak_memory_stats(extractor.device)
        document, branches = self.fields_for(step)
        self.model.train()
        prefix = extractor.encode_prefix(document.pages)
        with torch.no_grad():
            positions = prefix_positions(extractor.core, prefix)
        packed = pack(prefix, positions, branches, config.format.max_value_tokens, self.dtype)
        out = value_loss(extractor.core, extractor.lm_head, packed, config.loss_weighting, config.loss_chunk)
        if not torch.isfinite(out.loss):
            raise FloatingPointError(f'Non-finite loss at step {step + 1} ({document.id})')
        # Checkpointed layers are recomputed in backward and must use the forward's kernel.
        with sdpa_kernel(SDPBackend.MATH):
            (out.loss / config.gradient_accumulation).backward()
        record: dict[str, Any] = {
            'step': step + 1, 'document_id': document.id, 'form': document.form, 'fields': len(branches),
            'target_tokens': out.target_tokens, 'sequence_tokens': int(packed.inputs['input_ids'].shape[1]),
            'loss': float(out.loss.detach()), 'token_accuracy': out.token_accuracy,
            'close_accuracy': out.close_accuracy, 'lr': self.optimizer.param_groups[0]['lr']}
        if self.is_update(step):
            remainder = (step + 1) % config.gradient_accumulation
            if remainder:
                # A final partial accumulation: rescale to a mean over the steps it has.
                for parameter in self.parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(config.gradient_accumulation / remainder)
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, config.max_grad_norm)
            if not torch.isfinite(norm):
                raise FloatingPointError(f'Non-finite gradient norm at step {step + 1}')
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            record['grad_norm'] = float(norm)
        record['seconds'] = round(time.monotonic() - self.start, 2)
        if cuda:
            record['peak_gib'] = round(torch.cuda.max_memory_allocated(extractor.device) / 2**30, 2)
        return record

    # --------------------------------------------------------------- monitor

    def monitor(self, step: int) -> list[dict[str, Any]]:
        """Greedy-decode the monitor documents and measure what the model writes.

        Returns:
            One ``monitor.jsonl`` record per split: ``exact_match``,
            ``coordinate_rate`` (box output instead of text) and
            ``truncated_rate`` over the first ``monitor.fields`` fields.
        """
        settings = self.config.monitor
        records = []
        started = time.monotonic()
        for split, documents in self.monitor_documents.items():
            fields = matches = coordinates = truncated = 0
            for document in documents:
                chosen = document.fields[:settings.fields]
                results = extract(self.extractor, document.pages, [FieldRequest.from_field(f) for f in chosen],
                                  max_value_tokens=settings.max_value_tokens)
                for field, result in zip(chosen, results, strict=True):
                    fields += 1
                    matches += result.status == 'ok' and correct(result.value, field.value)
                    coordinates += is_coordinates(result.text)
                    truncated += result.status == 'truncated'
            records.append({'step': step, 'split': split, 'documents': len(documents), 'fields': fields,
                            'exact_match': matches / max(fields, 1), 'coordinate_rate': coordinates / max(fields, 1),
                            'truncated_rate': truncated / max(fields, 1),
                            'seconds': round(time.monotonic() - started, 2)})
        self.model.train()
        return records

    # ------------------------------------------------------------------- run

    def run(self, until: int | None = None) -> dict[str, Any] | None:
        """Train from the first unfinished step to ``config.steps``.

        Args:
            until: Stop after this many steps without the final save, as if
                interrupted (tests use it to check resuming).

        Returns:
            The training summary, or ``None`` when stopped by ``until``.
        """
        config = self.config
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / 'config.json', config.to_dict())
        log = _open_log(self.output / 'train.jsonl', self.first_step)
        monitor_log = _open_log(self.output / 'monitor.jsonl', self.first_step)

        def emit(records: list[dict[str, Any]], handle: TextIO, show: bool) -> None:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                if show:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
            handle.flush()

        try:
            if config.monitor.every and self.first_step == 0:
                emit(self.monitor(0), monitor_log, True)
            self.optimizer.zero_grad(set_to_none=True)
            for step in range(self.first_step, config.steps):
                if until is not None and step >= until:
                    return None
                record = self.train_step(step)
                emit([record], log, step == 0 or (step + 1) % config.log_every == 0)
                if self.due(step, config.monitor.every):
                    emit(self.monitor(step + 1), monitor_log, True)
                if self.is_update(step) and (self.due(step, config.save_every) or step + 1 == config.steps):
                    save_snapshot(self.model, self.optimizer, self.scheduler, self.field_rng, self.output, config,
                                  step + 1, time.monotonic() - self.start, self.extractor.device)
            save_adapter(self.model, self.output, config)
        finally:
            log.close()
            monitor_log.close()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        """Write and return ``training_summary.json`` from the logs."""
        records = read_jsonl(self.output / 'train.jsonl')
        monitor_path = self.output / 'monitor.jsonl'
        monitored = read_jsonl(monitor_path) if monitor_path.exists() else []

        def mean(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        peaks = [r['peak_gib'] for r in records if 'peak_gib' in r]
        window = min(50, len(records) // 2)
        summary = {
            'steps': self.config.steps, 'resumed_from_step': self.first_step,
            'documents_seen': len({r['document_id'] for r in records}), 'forms_seen': len({r['form'] for r in records}),
            'training_seconds': round(time.monotonic() - self.start, 1),
            'target_tokens': sum(r['target_tokens'] for r in records),
            'max_sequence_tokens': max((r['sequence_tokens'] for r in records), default=0),
            # Windows of up to 50 steps that never overlap (short runs use halves).
            'loss_window': window,
            'mean_loss_first': mean([r['loss'] for r in records[:window]]),
            'mean_loss_last': mean([r['loss'] for r in records[-window:]]) if window else None,
            'peak_allocated_gib': max(peaks) if peaks else None,
            'monitor_first': [r for r in monitored if r['step'] == monitored[0]['step']] if monitored else [],
            'monitor_last': [r for r in monitored if r['step'] == monitored[-1]['step']] if monitored else [],
        }
        write_json(self.output / 'training_summary.json', summary)
        return summary


def _open_log(path: Path, keep_through: int) -> TextIO:
    """Open a JSONL log for appending, dropping records after step ``keep_through``.

    On resume, steps after the snapshot are replayed, so their records (and a
    possibly torn last line) are removed first.
    """
    if path.exists():
        kept = []
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            if record['step'] <= keep_through:
                kept.append(line)
        path.write_text(''.join(line + '\n' for line in kept), encoding='utf-8')
    return path.open('a', encoding='utf-8')


def train(config: TrainConfig, base: torch.nn.Module | None = None, processor: Any = None,
          until: int | None = None) -> dict[str, Any] | None:
    """Build a :class:`Trainer` and run it (see :meth:`Trainer.run`)."""
    return Trainer(config, base, processor).run(until)
