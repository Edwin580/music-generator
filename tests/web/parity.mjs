// Runs the browser modules in Node for tests/test_web_parity.py.
// Usage: node parity.mjs request.json  ->  prints a JSON response on stdout.
import { readFileSync } from "node:fs";
import { Transformer, decodeFloat16 } from "../../docs/js/model.js";
import { GrammarState, REMITokenizer } from "../../docs/js/remi.js";
import { filterProbs } from "../../docs/js/sampler.js";
import { keyClarity, repetitionRatio, scaleConsistency } from "../../docs/js/metrics.js";
import { estimateKey, keyCorrelations, parseKey } from "../../docs/js/theory.js";

const req = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = {};
const toNotes = (rows) => rows.map(([start, pitch, duration, velocity]) => ({ start, pitch, duration, velocity }));
const fromNotes = (notes) => notes.map((n) => [n.start, n.pitch, n.duration, n.velocity]);

if (req.model) {
  const dir = req.model.dir;
  const manifest = JSON.parse(readFileSync(`${dir}/model.json`, "utf8"));
  const bin = readFileSync(`${dir}/model.bin`);
  const weights = decodeFloat16(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
  const model = new Transformer(manifest, weights);
  const state = model.newState();
  out.logits = req.model.tokens.map((t) => Array.from(model.step(t, state)));
}

if (req.tokenizer) {
  const tok = new REMITokenizer(req.tokenizer.config);
  out.vocabSize = tok.vocabSize;
  out.encoded = tok.encode(toNotes(req.tokenizer.notes));
  out.decoded = fromNotes(tok.decode(req.tokenizer.tokens));
  const key = req.tokenizer.key ? parseKey(req.tokenizer.key) : null;
  const g = new GrammarState(tok, { key, monophonic: req.tokenizer.monophonic });
  out.masks = [];
  for (const t of req.tokenizer.walk) {
    out.masks.push(Array.from(g.allowed(true)));
    g.update(t);
  }
}

if (req.filter) {
  out.filtered = req.filter.cases.map((c) => Array.from(filterProbs(c.logits, {
    temperature: c.temperature, topK: c.top_k, topP: c.top_p, mask: c.mask,
  })));
}

if (req.theory) {
  out.correlations = req.theory.histograms.map((h) => keyCorrelations(h));
  out.metrics = req.theory.pieces.map((rows) => {
    const notes = toNotes(rows);
    const key = estimateKey(notes);
    return {
      key: key ? key.name : null,
      scale_consistency: scaleConsistency(notes),
      key_clarity: keyClarity(notes),
      repetition_ratio: repetitionRatio(notes),
    };
  });
}

process.stdout.write(JSON.stringify(out));
