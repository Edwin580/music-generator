// Runs the model off the main thread so dragging a value never freezes the page.
import { Transformer } from "./model.js";
import { REMITokenizer } from "./remi.js";
import { generateTokens } from "./sampler.js";
import { pieceMetrics, sliceBars } from "./metrics.js";
import { estimateKey, parseKey } from "./theory.js";

const base = new URL("../model", import.meta.url).href;
const ready = Transformer.load(base).then(async (model) => {
  const manifest = await (await fetch(`${base}/model.json`)).json();
  return { model, tok: new REMITokenizer(manifest.tokenizer) };
});

function resolveKey(mode, prompt) {
  if (mode === "auto") return prompt.length ? estimateKey(prompt) : null;
  if (mode === "none") return null;
  return parseKey(mode);
}

function continuation(model, tok, notes, p, seed, keyMode, trace) {
  const prompt = sliceBars(notes, 0, p.promptBars, tok.stepsPerBar);
  const key = resolveKey(keyMode, prompt);
  const promptTokens = tok.encode(prompt, p.promptBars, true, false);
  const out = generateTokens(model, tok, promptTokens, p.genBars, {
    temperature: p.temperature, topP: p.topP, key, seed, trace,
  });
  return { ...out, key: key ? key.name : null };
}

self.onmessage = async ({ data }) => {
  const { model, tok } = await ready;
  if (data.type === "generate") {
    const r = continuation(model, tok, data.notes, data.params, data.params.seed, data.params.keyMode, true);
    self.postMessage({ type: "generated", id: data.id, ...r });
  } else if (data.type === "experiment") {
    // Continue every held-out prompt with and without the key constraint (same seed for both).
    const p = data.params;
    const offset = p.promptBars * tok.stepsPerBar;
    data.pieces.forEach((piece, i) => {
      const row = {};
      for (const mode of ["auto", "none"]) {
        const r = continuation(model, tok, piece.notes, p, i + 1, mode, false);
        const gen = tok.decode(r.tokens).filter((n) => n.start >= offset).map((n) => ({ ...n, start: n.start - offset }));
        row[mode] = pieceMetrics(gen, p.genBars);
      }
      const realBars = Math.min(p.genBars, piece.bars - p.promptBars);
      row.real = pieceMetrics(sliceBars(piece.notes, p.promptBars, p.promptBars + realBars, tok.stepsPerBar), realBars);
      for (const k of Object.keys(row)) if (row[k]) row[k].key = row[k].key ? row[k].key.name : null;
      self.postMessage({ type: "progress", id: data.id, index: i, total: data.pieces.length, row });
    });
  }
};
