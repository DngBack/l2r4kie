"""Restartable snapshots at optimizer-update boundaries.

Layout inside the run directory::

    latest_training.json            {"path": "snapshots/step-000200", "step": 200}
    snapshots/step-000100/          previous complete snapshot (kept)
    snapshots/step-000200/          adapter/, config.json, training_state.pt

Crash safety (unchanged from the old ``training.save_snapshot``):

* a snapshot is written to ``step-N.tmp`` and renamed only when complete;
* the pointer file is replaced atomically, only after the rename;
* the snapshot the pointer references is never deleted or overwritten, and
  its predecessor is kept, so a torn write never loses the last complete state.
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..utils.io import PathLike, read_json, write_json
from .config import TrainConfig

POINTER = 'latest_training.json'
STATE = 'training_state.pt'
KEEP = 2


@dataclass(frozen=True, slots=True)
class RestoredState:
    """Progress restored from a snapshot."""

    step: int
    seconds: float


def save_adapter(model: torch.nn.Module, directory: PathLike, config: TrainConfig) -> None:
    """Write the trainable weights (``adapter/``) and the run config (``config.json``)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory / 'adapter')
    write_json(directory / 'config.json', config.to_dict())


def latest_snapshot(output: PathLike, config: TrainConfig) -> Path | None:
    """Return the committed snapshot of ``output``, or ``None`` if there is none.

    Raises:
        ValueError: If the snapshot was trained with a different config
            (only :data:`~l2r4kie.train.config.OPERATIONAL_KEYS` may differ).
    """
    pointer = Path(output) / POINTER
    if not pointer.exists():
        return None
    snapshot = Path(output) / read_json(pointer)['path']
    saved = TrainConfig.from_dict(read_json(snapshot / 'config.json'))
    if saved.comparable() != config.comparable():
        changed = sorted(k for k in config.comparable() if saved.comparable().get(k) != config.comparable()[k])
        raise ValueError(f'Resume config differs from the snapshot in {changed}; use a new output directory')
    return snapshot


def save_snapshot(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any,
                  field_rng: random.Random, output: PathLike, config: TrainConfig, step: int, seconds: float,
                  device: torch.device) -> Path:
    """Commit a snapshot after ``step`` steps; returns its directory.

    Args:
        model: The PEFT model (its trainable weights are saved).
        optimizer: Optimizer whose state is saved.
        scheduler: LR scheduler whose state is saved.
        field_rng: Field-sampling generator.
        output: Run directory.
        config: Run config, stored for resume validation.
        step: Steps completed.
        seconds: Training time so far.
        device: Model device (for the CUDA RNG state).

    Raises:
        ValueError: If asked to overwrite the snapshot the pointer references.
    """
    output = Path(output)
    destination = output / 'snapshots' / f'step-{step:06d}'
    temporary = destination.with_name(destination.name + '.tmp')
    if temporary.exists():
        shutil.rmtree(temporary)
    save_adapter(model, temporary, config)
    state = {'step': step, 'seconds': seconds, 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'field_rng': field_rng.getstate(),
             'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state()}
    if device.type == 'cuda':
        state['cuda_rng'] = torch.cuda.get_rng_state(device)
    torch.save(state, temporary / STATE)
    pointer = output / POINTER
    if destination.exists():
        # A crash after the rename but before the pointer update leaves an
        # uncommitted directory at this step. Replaying may replace it, but
        # must never remove the snapshot the pointer references.
        current = read_json(pointer)['path'] if pointer.exists() else None
        if current == str(destination.relative_to(output)):
            raise ValueError(f'Refusing to overwrite the committed snapshot {destination}')
        shutil.rmtree(destination)
    temporary.rename(destination)
    write_json(pointer, {'path': str(destination.relative_to(output)), 'step': step})
    committed = sorted(p for p in destination.parent.glob('step-*') if p.is_dir() and not p.name.endswith('.tmp'))
    for old in committed[:-KEEP]:
        shutil.rmtree(old)
    return destination


def restore_state(snapshot: PathLike, optimizer: torch.optim.Optimizer, scheduler: Any,
                  field_rng: random.Random, device: torch.device) -> RestoredState:
    """Load optimizer, scheduler and every RNG state from a snapshot.

    The adapter weights are loaded separately, when the model is built.
    """
    state = torch.load(Path(snapshot) / STATE, map_location='cpu', weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    field_rng.setstate(state['field_rng'])
    random.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'])
    if 'cuda_rng' in state and device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda_rng'], device)
    return RestoredState(state['step'], state['seconds'])
