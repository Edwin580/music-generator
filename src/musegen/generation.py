"""Sampling from trained models.

The original notebook's temperature code was::

    pred = np.log(pred) / temperature
    pred = np.exp(pred) / np.sum(np.exp(pred))       # softmax over 128 notes...
    next_notes = np.random.random(pred.shape) < pred  # ...then 128 independent coin flips

A softmax forces the 128 probabilities to sum to 1, which throws away the
independent per-note probabilities a sigmoid output layer gives. The model's
confidence was lost and "temperature" did not mean what it should (``log(0)`` also
produced ``-inf`` warnings). Here:

* token models sample from a categorical distribution with temperature, top-k and
  nucleus (top-p) filtering, under a **grammar mask** so output always decodes
  cleanly, plus an optional **key constraint**;
* the piano-roll model uses the correct Bernoulli temperature ``sigmoid(logit / T)``
  with a polyphony limit and onset/sustain consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import GenerationConfig
from .data.midi_io import Score
from .data.pianoroll import notes_to_roll, roll_to_notes
from .data.tokenizer import BAR, EOS, GrammarState, REMITokenizer
from .theory import Key, Note, estimate_key


def filter_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a probability vector after masking, temperature, top-k and top-p filtering."""
    logits = logits.float().clone()
    if mask is not None:
        logits[~mask] = float("-inf")
    if temperature <= 0:  # greedy
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1.0
        return probs
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, int(torch.isfinite(logits).sum()))
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    if top_p is not None and 0 < top_p < 1:
        sorted_probs, order = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        drop = cumulative - sorted_probs > top_p  # keep the smallest set with mass >= top_p
        sorted_probs[drop] = 0.0
        probs = torch.zeros_like(probs).scatter(0, order, sorted_probs)
        probs = probs / probs.sum()
    return probs


def resolve_key(spec: str | Key | None, prompt_notes: list[Note] | None = None) -> Key | None:
    if spec is None or isinstance(spec, Key):
        return spec
    if spec.lower() in ("", "none", "off"):
        return None
    if spec.lower() == "auto":
        return estimate_key(prompt_notes) if prompt_notes else None
    return Key.parse(spec)


@dataclass
class GenerationResult:
    score: Score
    prompt_bars: int
    key: Key | None
    tokens: list[int] | None = None


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def generate_tokens(
    model: nn.Module,
    tokenizer: REMITokenizer,
    n_bars: int,
    prompt: list[int] | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    key: Key | None = None,
    monophonic: bool = False,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Continue ``prompt`` (REMI tokens, may be empty) by ``n_bars`` bars."""
    model.eval()
    device = _device_of(model)
    tokens = [t for t in (prompt or [tokenizer.bos_id]) if t != tokenizer.eos_id]
    if tokens[0] != tokenizer.bos_id:
        tokens = [tokenizer.bos_id, *tokens]
    grammar = GrammarState(tokenizer, key=key, monophonic=monophonic).feed(tokens)
    target_bars = grammar.bars + n_bars
    types = tokenizer.token_types()

    context = tokens[-getattr(model, "max_len", len(tokens)) :]
    logits, state = model(torch.tensor([context], device=device))
    # Generous safety limit: every step of every bar holding a note, plus bar tokens.
    budget = n_bars * (tokenizer.steps_per_bar * 4 + 1) + 8
    for _ in range(budget):
        mask = torch.from_numpy(grammar.allowed(allow_eos=False)).to(device)
        probs = filter_logits(logits[0, -1], temperature, top_k, top_p, mask)
        nxt = int(torch.multinomial(probs, 1, generator=generator).item())
        if types[nxt] == EOS or (types[nxt] == BAR and grammar.bars >= target_bars):
            break
        tokens.append(nxt)
        grammar.update(nxt)
        logits, state = model(torch.tensor([[nxt]], device=device), state)
    return tokens


@torch.no_grad()
def generate_pianoroll(
    model: nn.Module,
    tokenizer: REMITokenizer,
    n_bars: int,
    prompt_notes: list[Note] | None = None,
    temperature: float = 1.0,
    key: Key | None = None,
    max_polyphony: int = 1,
    generator: torch.Generator | None = None,
    prompt_steps: int | None = None,
) -> list[Note]:
    """Generate ``n_bars`` bars of frames after ``prompt_notes`` (a roll of ``prompt_steps``
    frames); returns only the new notes, timed from the end of the prompt."""
    model.eval()
    device = _device_of(model)
    n_p = tokenizer.n_pitches
    roll = notes_to_roll(prompt_notes or [], tokenizer.min_pitch, tokenizer.max_pitch,
                         prompt_steps)
    if len(roll) == 0:
        roll = np.zeros((1, 2 * n_p), dtype=np.float32)
    in_key = torch.ones(n_p, dtype=torch.bool, device=device)
    if key is not None:
        in_key = torch.tensor(
            [key.contains(p + tokenizer.min_pitch) for p in range(n_p)], device=device
        )

    logits, state = model(torch.from_numpy(roll)[None].to(device))
    prev = torch.from_numpy(roll[-1, :n_p] > 0.5).to(device)
    frames = []
    for _ in range(n_bars * tokenizer.steps_per_bar):
        probs = torch.sigmoid(logits[0, -1].float() / max(temperature, 1e-4))
        sus_p, on_p = probs[:n_p], probs[n_p:] * in_key
        onset = torch.bernoulli(on_p, generator=generator).bool()
        sustain = torch.bernoulli(sus_p, generator=generator).bool() & prev & ~onset
        onset = _keep_top(onset, on_p, max_polyphony)
        sustain = _keep_top(sustain, sus_p, max(0, max_polyphony - int(onset.sum())))
        active = onset | sustain
        frame = torch.cat([active, onset]).float()
        frames.append(frame.cpu().numpy())
        prev = active
        logits, state = model(frame[None, None], state)
    notes = roll_to_notes(np.stack(frames), tokenizer.min_pitch)
    return notes


def _keep_top(selected: torch.Tensor, scores: torch.Tensor, limit: int) -> torch.Tensor:
    if int(selected.sum()) <= limit:
        return selected
    out = torch.zeros_like(selected)
    if limit > 0:
        ranked = torch.where(selected, scores, torch.full_like(scores, -1.0))
        out[torch.topk(ranked, limit).indices] = True
    return out


def generate(
    model: nn.Module,
    tokenizer: REMITokenizer,
    gen: GenerationConfig,
    prompt: Score | None = None,
    monophonic: bool = True,
    seed: int | None = None,
    steps_per_beat: int = 4,
) -> GenerationResult:
    """High-level entry point: continue an optional prompt score and return a full Score."""
    device = _device_of(model)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    prompt_notes = list(prompt.notes) if prompt else []
    prompt_bars = prompt.n_bars if prompt else 0
    key = resolve_key(gen.key, prompt_notes)
    offset = prompt_bars * tokenizer.steps_per_bar

    tokens = None
    if getattr(model, "kind", "tokens") == "pianoroll":
        new = generate_pianoroll(
            model, tokenizer, gen.bars, prompt_notes, gen.temperature, key,
            gen.max_polyphony, generator, prompt_steps=offset or None,
        )
        notes = prompt_notes + [Note(n.start + offset, n.pitch, n.duration, n.velocity)
                                for n in new]
    else:
        prompt_tokens = (
            tokenizer.encode(prompt_notes, n_bars=prompt_bars, add_eos=False) if prompt else None
        )
        tokens = generate_tokens(
            model, tokenizer, gen.bars, prompt_tokens, gen.temperature, gen.top_k, gen.top_p,
            key, monophonic, generator,
        )
        notes = tokenizer.decode(tokens)

    score = Score(sorted(notes), steps_per_beat, tokenizer.steps_per_bar // steps_per_beat,
                  gen.tempo, name="generated", bars=prompt_bars + gen.bars)
    return GenerationResult(score, prompt_bars, key, tokens)
