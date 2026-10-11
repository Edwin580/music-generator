import { adjustable } from "./adjust.js";
import { drawKeyChart, drawRoll, drawVocab } from "./draw.js";
import { mean, pieceMetrics, sliceBars } from "./metrics.js";
import { DUR, PITCH, REMITokenizer } from "./remi.js";
import { Player } from "./synth.js";
import { PITCH_CLASS_NAMES, estimateKey, keyCorrelations, pitchClassHistogram, pitchName } from "./theory.js";

const $ = (sel) => document.querySelector(sel);
const pct = (x) => `${Math.round(x * 100)}%`;
const num = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "n/a");

const KEYS = ["auto", "none",
  ...PITCH_CLASS_NAMES.map((n) => `${n} major`), ...PITCH_CLASS_NAMES.map((n) => `${n} minor`)];

const params = { piece: 0, promptBars: 4, genBars: 12, keyMode: "auto", temperature: 0.9, topP: 0.95, seed: 1 };
let tok, manifest, pieces, result = null, stepIndex = 0, showTruth = true;
let renderKeyLabel = () => {}; // the "prompt's key" label changes with the piece and prompt
const player = new Player();

// ------------------------------------------------------------------ generation
const genWorker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let busy = false, pending = false, requestId = 0;

function requestGeneration() {
  if (busy) { pending = true; return; }
  busy = true;
  pending = false;
  $("#status").textContent = "writing…";
  genWorker.postMessage({ type: "generate", id: ++requestId, notes: pieces[params.piece].notes, params: { ...params } });
}

genWorker.onmessage = ({ data }) => {
  busy = false;
  if (pending) { requestGeneration(); return; }
  $("#status").textContent = "";
  result = data;
  // start the step view on the first pitch the model chose
  const first = result.steps.findIndex((s) => tok.types[s.chosen] === PITCH);
  stepIndex = Math.max(0, first);
  renderAll();
};
genWorker.onerror = (e) => { $("#status").textContent = `The model failed to load (${e.message || "worker error"}).`; };

// ------------------------------------------------------------------ views
const stepsPerBar = () => tok.stepsPerBar;

function parts() {
  const piece = pieces[params.piece];
  const offset = params.promptBars * stepsPerBar();
  const prompt = sliceBars(piece.notes, 0, params.promptBars, stepsPerBar());
  const generated = tok.decode(result.tokens).filter((n) => n.start >= offset);
  const realBars = Math.max(0, Math.min(params.genBars, piece.bars - params.promptBars));
  const truth = sliceBars(piece.notes, params.promptBars, params.promptBars + realBars, stepsPerBar());
  return { piece, offset, prompt, generated, truth, realBars };
}

let movePlayhead = () => {};

function renderRoll() {
  const { piece, prompt, generated, truth } = parts();
  movePlayhead = drawRoll($("#roll"), {
    prompt, generated, truth: showTruth ? truth : [], promptBars: params.promptBars,
    totalBars: params.promptBars + params.genBars, stepsPerBar: stepsPerBar(),
  });
  const last = params.promptBars + params.genBars;
  $("#roll-caption").textContent =
    `Bars 1 to ${params.promptBars} open ${piece.name}, a piece the model never trained on. ` +
    `Bars ${params.promptBars + 1} to ${last}, in red, are the model's, seed ${params.seed}` +
    (showTruth ? ". Outlines show what the piece really does next." : ".");
}

function renderStats() {
  const { generated, truth, offset, realBars } = parts();
  const shift = (ns) => ns.map((n) => ({ ...n, start: n.start - offset }));
  const g = pieceMetrics(shift(generated), params.genBars);
  const r = pieceMetrics(truth, realBars);
  const rows = [
    ["Key it sounds like", (m) => (m && m.key ? m.key.name : "n/a")],
    ["Notes in that key's scale", (m) => (m ? pct(m.scaleConsistency) : "n/a")],
    ["Key clarity (correlation)", (m) => (m ? num(m.keyClarity) : "n/a")],
    ["Notes per bar", (m) => (m ? num(m.notesPerBar, 1) : "n/a")],
    ["Mean leap (semitones)", (m) => (m ? num(m.meanInterval, 1) : "n/a")],
    ["Repeated 4-note figures", (m) => (m ? pct(m.repetition) : "n/a")],
  ];
  $("#stats tbody").replaceChildren(...rows.map(([label, f]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<th scope="row">${label}</th><td>${f(g)}</td><td>${f(r)}</td>`;
    return tr;
  }));
}

function renderTokens() {
  const box = $("#tokens");
  box.replaceChildren();
  const stepAt = new Map(result.steps.map((s, i) => [s.index, i]));
  let line = null;
  result.tokens.forEach((t, i) => {
    if (t === tok.barId || line === null) {
      line = document.createElement("div");
      box.appendChild(line);
    }
    const span = document.createElement(stepAt.has(i) ? "button" : "span");
    span.textContent = tok.describe(t);
    span.className = i < result.promptLength ? "tk prompt" : "tk gen";
    if (stepAt.has(i)) {
      span.type = "button";
      span.dataset.step = stepAt.get(i);
      if (stepAt.get(i) === stepIndex) span.classList.add("current");
    }
    line.append(span, " ");
  });
}

function describeChoice(t) {
  const name = tok.describe(t);
  return tok.types[t] === PITCH ? `${name} (${pitchName(tok.pitchOf(t))})` : name;
}

function renderStep() {
  const steps = result.steps;
  stepIndex = Math.min(stepIndex, steps.length - 1);
  const s = steps[stepIndex];
  $("#step-range").max = steps.length - 1;
  $("#step-range").value = stepIndex;
  $("#step-label").textContent = `Step ${stepIndex + 1} of ${steps.length}`;
  const before = result.tokens.slice(Math.max(0, s.index - 6), s.index).map((t) => tok.describe(t)).join(" ");
  $("#step-context").innerHTML = `<span class="muted">… ${before}</span> <span class="accent">${tok.describe(s.chosen)}</span>`;
  drawVocab($("#vocab"), s, tok);

  let grammarMass = 0, keyMass = 0, legal = 0, keyed = 0, kept = 0;
  for (let i = 0; i < tok.vocabSize; i++) {
    if (s.grammar[i]) { grammarMass += s.model[i]; legal++; }
    if (s.grammar[i] && !s.allowed[i] && i !== tok.eosId) { keyMass += s.model[i]; keyed++; }
    if (s.final[i] > 0) kept++;
  }
  const keyName = result.key;
  let text = `The grammar allows ${legal} of ${tok.vocabSize} tokens here. The model had put ${pct(grammarMass)} of its probability on them. `;
  if (keyed) text += `The key of ${keyName} removes ${keyed} of those, which held ${pct(keyMass)}. `;
  text += kept === 1
    ? `After temperature and top-p only one token is left, so the draw is forced: ${describeChoice(s.chosen)}.`
    : `Temperature and top-p leave ${kept} candidates, and the draw picks ${describeChoice(s.chosen)}, which had p = ${num(s.final[s.chosen])} after filtering.`;
  $("#step-text").textContent = text;

  const order = Array.from(s.model.keys()).sort((a, b) => s.model[b] - s.model[a]).slice(0, 8);
  if (!order.includes(s.chosen)) order.push(s.chosen);
  $("#candidates tbody").replaceChildren(...order.map((i) => {
    let fate;
    if (!s.grammar[i]) fate = "not allowed here";
    else if (i === tok.eosId && !s.allowed[i]) fate = "stopping early is off";
    else if (!s.allowed[i]) fate = `outside ${keyName}`;
    else if (s.final[i] === 0) fate = "cut by top-p";
    else fate = i === s.chosen ? "drawn" : "kept";
    const tr = document.createElement("tr");
    if (i === s.chosen) tr.className = "chosen";
    tr.innerHTML = `<td class="mono">${tok.describe(i)}</td>` +
      `<td><span class="meter" style="--w:${s.model[i]}"></span>${num(s.model[i])}</td>` +
      `<td><span class="meter" style="--w:${s.final[i]}"></span>${num(s.final[i])}</td><td>${fate}</td>`;
    return tr;
  }));

  const wasted = result.steps.map((st) => {
    let m = 0;
    for (let i = 0; i < tok.vocabSize; i++) if (!st.grammar[i]) m += st.model[i];
    return m;
  });
  $("#illegal").textContent = `Over this melody the model put ${num(mean(wasted) * 100, 1)}% of its probability on tokens the grammar rules out, on average, and ${num(Math.max(...wasted) * 100, 1)}% at its worst step.`;

  document.querySelectorAll("#tokens .current").forEach((n) => n.classList.remove("current"));
  const cur = document.querySelector(`#tokens [data-step="${stepIndex}"]`);
  if (cur) cur.classList.add("current");
}

function renderKey() {
  const { prompt } = parts();
  const corr = keyCorrelations(pitchClassHistogram(prompt));
  const best = estimateKey(prompt);
  drawKeyChart($("#keychart"), corr, best);
  if (!best) { $("#key-text").textContent = "The prompt has no notes, so there is no key to find."; return; }
  const ranked = [];
  ["major", "minor"].forEach((mode, m) => corr[m].forEach((r, t) => ranked.push({ r, name: `${PITCH_CLASS_NAMES[t]} ${mode}` })));
  ranked.sort((a, b) => b.r - a.r);
  $("#key-text").textContent = `For the first ${params.promptBars} bars of ${pieces[params.piece].name}, ` +
    `the best match is ${ranked[0].name} (r = ${num(ranked[0].r)}), then ${ranked[1].name} (r = ${num(ranked[1].r)}).`;
}

function renderAll() {
  if (!result) return;
  renderKeyLabel();
  renderRoll();
  renderStats();
  renderTokens();
  renderStep();
  renderKey();
  renderExperimentSettings();
}

// ------------------------------------------------------------------ experiment
const expWorker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let expRows = [], expId = 0, expStart = 0;

function renderExperimentSettings() {
  $("#exp-settings").textContent = `It uses the settings above: ${params.promptBars} bars of prompt, ` +
    `${params.genBars} bars of continuation, temperature ${params.temperature.toFixed(2)}, top-p ${params.topP.toFixed(2)}.`;
}

function renderExperiment(done) {
  const cols = [
    ["scaleConsistency", (x) => pct(x)], ["keyClarity", (x) => num(x)],
    ["meanInterval", (x) => num(x, 1)], ["repetition", (x) => pct(x)],
  ];
  const avg = (which, key) => mean(expRows.map((r) => r[which]).filter(Boolean).map((m) => m[key]));
  const rows = [["What the pieces really do", "real"], ["Model, constrained to the prompt's key", "auto"], ["Model, no constraint", "none"]];
  $("#experiment tbody").replaceChildren(...rows.map(([label, which]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<th scope="row">${label}</th>` + cols.map(([k, f]) => `<td>${f(avg(which, k))}</td>`).join("");
    return tr;
  }));
  $("#exp-progress").textContent = done
    ? `Done: ${expRows.length * 2} continuations in ${((performance.now() - expStart) / 1000).toFixed(1)} s.`
    : `Running: ${expRows.length} of ${pieces.length} prompts…`;
  if (done) {
    const out = (w) => pct(1 - avg(w, "scaleConsistency"));
    $("#exp-result").textContent = `Without the constraint, ${out("none")} of the model's notes fall outside the scale that fits its own melody best. ` +
      `With it, ${out("auto")} do. The real continuations repeat ${pct(avg("real", "repetition"))} of their 4-note figures; ` +
      `the model's repeat ${pct(avg("auto", "repetition"))}.`;
  }
}

$("#run-exp").addEventListener("click", () => {
  expRows = [];
  expStart = performance.now();
  $("#run-exp").disabled = true;
  $("#exp-result").textContent = "";
  $("#experiment").hidden = false;
  renderExperiment(false);
  expWorker.postMessage({ type: "experiment", id: ++expId, pieces, params: { ...params } });
});
expWorker.onmessage = ({ data }) => {
  if (data.id !== expId) return;
  expRows.push(data.row);
  const done = data.index + 1 === data.total;
  renderExperiment(done);
  if (done) $("#run-exp").disabled = false;
};

// ------------------------------------------------------------------ controls
function setUp() {
  const regenerate = () => { player.stop(); requestGeneration(); };
  adjustable($("#a-prompt"), { min: 1, max: 8, step: 1, pxPerStep: 18, label: "prompt bars",
    get: () => params.promptBars, set: (v) => { params.promptBars = v; regenerate(); },
    format: (v) => `${v} bar${v > 1 ? "s" : ""}` });
  adjustable($("#a-piece"), { min: 0, max: pieces.length - 1, step: 1, pxPerStep: 10, label: "held-out piece",
    get: () => params.piece, set: (v) => { params.piece = v; regenerate(); },
    format: (v) => `${v + 1} of ${pieces.length}` });
  adjustable($("#a-bars"), { min: 1, max: 12, step: 1, pxPerStep: 14, label: "bars to write",
    get: () => params.genBars, set: (v) => { params.genBars = v; regenerate(); },
    format: (v) => `${v} more bar${v > 1 ? "s" : ""}` });
  renderKeyLabel = adjustable($("#a-key"), { options: KEYS, pxPerStep: 14, label: "key constraint",
    get: () => params.keyMode, set: (v) => { params.keyMode = v; regenerate(); },
    format: (v) => {
      if (v === "auto") {
        const k = estimateKey(sliceBars(pieces[params.piece].notes, 0, params.promptBars, stepsPerBar()));
        return `the prompt's key${k ? ` (${k.name})` : ""}`;
      }
      return v === "none" ? "any key at all" : v;
    } });
  adjustable($("#a-temp"), { min: 0.1, max: 2, step: 0.05, pxPerStep: 6, label: "temperature",
    get: () => params.temperature, set: (v) => { params.temperature = v; regenerate(); },
    format: (v) => v.toFixed(2) });
  adjustable($("#a-topp"), { min: 0.5, max: 1, step: 0.01, pxPerStep: 5, label: "top-p",
    get: () => params.topP, set: (v) => { params.topP = v; regenerate(); },
    format: (v) => `${Math.round(v * 100)}%` });
  $("#new-take").addEventListener("click", () => { params.seed += 1; player.stop(); requestGeneration(); });
  $("#toggle-truth").addEventListener("click", () => {
    showTruth = !showTruth;
    $("#toggle-truth").textContent = showTruth ? "Hide the real continuation" : "Show the real continuation";
    renderRoll();
  });
  $("#play").addEventListener("click", () => {
    if (player.playing) { player.stop(); return; }
    const { piece, prompt, generated } = parts();
    const total = (params.promptBars + params.genBars) * stepsPerBar();
    player.play([...prompt, ...generated], piece.tempo, total);
    $("#play").textContent = "Stop";
    const sps = 60 / piece.tempo / 4;
    const tick = () => {
      const pos = player.position(sps);
      movePlayhead(pos !== null && pos >= 0 ? pos : null);
      if (player.playing) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
  player.onEnd = () => { $("#play").textContent = "Play"; movePlayhead(null); };

  $("#step-range").addEventListener("input", (e) => { stepIndex = +e.target.value; renderStep(); });
  $("#step-prev").addEventListener("click", () => { stepIndex = Math.max(0, stepIndex - 1); renderStep(); });
  $("#step-next").addEventListener("click", () => { stepIndex = Math.min(result.steps.length - 1, stepIndex + 1); renderStep(); });
  $("#step-pitch").addEventListener("click", () => {
    const next = result.steps.findIndex((s, i) => i > stepIndex && tok.types[s.chosen] === PITCH);
    stepIndex = next >= 0 ? next : result.steps.findIndex((s) => tok.types[s.chosen] === PITCH);
    renderStep();
  });
  $("#tokens").addEventListener("click", (e) => {
    const b = e.target.closest("[data-step]");
    if (b) { stepIndex = +b.dataset.step; renderStep(); }
  });

  let width = window.innerWidth;
  window.addEventListener("resize", () => {
    if (window.innerWidth === width) return;
    width = window.innerWidth;
    renderAll();
  });
}

async function main() {
  const base = new URL("../model", import.meta.url).href;
  [manifest, pieces] = await Promise.all([
    fetch(`${base}/model.json`).then((r) => r.json()),
    fetch(`${base}/prompts.json`).then((r) => r.json()).then((p) => p.pieces),
  ]);
  pieces = pieces.map((p) => ({ ...p, notes: p.notes.map(([start, pitch, duration, velocity]) => ({ start, pitch, duration, velocity })) }));
  tok = new REMITokenizer(manifest.tokenizer);
  const t = manifest.training;
  document.querySelectorAll("[data-fact]").forEach((node) => {
    node.textContent = {
      params: t.n_params.toLocaleString("en-US"),
      size: `${(t.n_params * 2 / 1e6).toFixed(1)} MB`,
      vocab: tok.vocabSize,
      pieces: pieces.length,
      layers: manifest.config.n_layers,
      dmodel: manifest.config.d_model,
      epochs: t.epoch,
      ppl: num(t.val_perplexity),
    }[node.dataset.fact];
  });
  setUp();
  requestGeneration();
}

main().catch((e) => { $("#status").textContent = `Could not load the model: ${e.message}`; });
