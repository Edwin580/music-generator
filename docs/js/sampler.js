// Port of src/musegen/generation.py (token models): masked temperature + top-p sampling.
// Each step is recorded so the page can show what the model wanted and what was allowed.

import { BAR, EOS, GrammarState } from "./remi.js";

// Same order as filter_logits(): mask, temperature, (top-k), softmax, top-p.
export function filterProbs(logits, { temperature = 1, topK = null, topP = null, mask = null } = {}) {
  const n = logits.length;
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) z[i] = mask && !mask[i] ? -Infinity : logits[i];
  const probs = new Float64Array(n);
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < n; i++) if (z[i] > z[best]) best = i;
    probs[best] = 1;
    return probs;
  }
  for (let i = 0; i < n; i++) z[i] /= temperature;
  if (topK && topK > 0) {
    const finite = Array.from(z).filter(Number.isFinite).sort((a, b) => b - a);
    const kth = finite[Math.min(topK, finite.length) - 1];
    for (let i = 0; i < n; i++) if (z[i] < kth) z[i] = -Infinity;
  }
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (z[i] > max) max = z[i];
  let sum = 0;
  for (let i = 0; i < n; i++) { probs[i] = Number.isFinite(z[i]) ? Math.exp(z[i] - max) : 0; sum += probs[i]; }
  for (let i = 0; i < n; i++) probs[i] /= sum;
  if (topP && topP > 0 && topP < 1) {
    // keep the smallest set of most likely tokens whose mass reaches topP
    const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a] || a - b);
    let cum = 0, kept = 0;
    for (const i of order) {
      if (cum > topP) probs[i] = 0;
      else kept += probs[i];
      cum += probs[i] || 0;
    }
    for (let i = 0; i < n; i++) probs[i] /= kept;
  }
  return probs;
}

export function softmax(logits) {
  return filterProbs(logits, { temperature: 1 });
}

// Small seeded PRNG (mulberry32) so a seed always gives the same melody.
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function draw(probs, random) {
  let r = random(), last = 0;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] <= 0) continue;
    last = i;
    r -= probs[i];
    if (r < 0) return i;
  }
  return last;
}

// Continue promptTokens (BOS BAR ... ) by nBars bars. Mirrors generate_tokens().
// With trace=true, every sampling step stores the raw distribution and each filter's effect.
export function generateTokens(model, tok, promptTokens, nBars, {
  temperature = 1, topP = 0.95, topK = null, key = null, monophonic = true, seed = 0, trace = false,
} = {}) {
  const tokens = promptTokens.filter((t) => t !== tok.eosId);
  if (tokens[0] !== tok.bosId) tokens.unshift(tok.bosId);
  const grammar = new GrammarState(tok, { key, monophonic }).feed(tokens);
  const plain = trace ? new GrammarState(tok, { monophonic }).feed(tokens) : null;
  const targetBars = grammar.bars + nBars;
  if (grammar.bars && nBars > 0) {
    // start the first new bar so nothing is written into the prompt's last bar
    tokens.push(tok.barId);
    grammar.update(tok.barId);
    if (plain) plain.update(tok.barId);
  }
  const promptLength = tokens.length;
  const random = rng(seed);
  const state = model.newState();
  let logits;
  for (const t of tokens.slice(-model.maxLen)) logits = model.step(t, state);

  const steps = [];
  const budget = nBars * (tok.stepsPerBar * 4 + 1) + 8;
  for (let i = 0; i < budget; i++) {
    const mask = grammar.allowed(false);
    const probs = filterProbs(logits, { temperature, topK, topP, mask });
    const next = draw(probs, random);
    if (trace) {
      steps.push({
        index: tokens.length,
        model: softmax(logits),
        grammar: plain.allowed(true), // EOS is grammatical; the sampler just never stops early
        allowed: mask,
        tempered: filterProbs(logits, { temperature, mask }),
        final: probs,
        chosen: next,
      });
    }
    const type = tok.types[next];
    if (type === EOS || (type === BAR && grammar.bars >= targetBars)) break;
    tokens.push(next);
    grammar.update(next);
    if (plain) plain.update(next);
    logits = model.step(next, state);
  }
  return { tokens, promptLength, steps };
}
