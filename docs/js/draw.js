// SVG figures. Colours come from CSS classes in style.css.
import { BAR, DUR, PITCH, POS, VEL } from "./remi.js";
import { PITCH_CLASS_NAMES, pitchName } from "./theory.js";

const NS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function reset(svg, width, height) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
}

const fmt = (x, d = 2) => x.toFixed(d);

// Piano roll: prompt notes in ink, generated notes in the accent colour, the piece's real
// continuation as outlines. Returns a function that moves the playhead (in steps).
export function drawRoll(svg, { prompt, generated, truth, promptBars, totalBars, stepsPerBar = 16 }) {
  const width = Math.max(280, Math.floor(svg.parentElement.clientWidth));
  const narrow = width < 520;
  const all = [...prompt, ...generated, ...truth];
  const lo = Math.min(...all.map((n) => n.pitch), 60) - 2;
  const hi = Math.max(...all.map((n) => n.pitch), 72) + 2;
  const rows = hi - lo + 1;
  const left = narrow ? 26 : 30, right = 4, top = 18, bottom = 18;
  const rowH = Math.max(3, Math.min(7, Math.floor((narrow ? 190 : 250) / rows)));
  const height = top + rows * rowH + bottom;
  reset(svg, width, height);
  const steps = totalBars * stepsPerBar;
  const x = (s) => left + (s / steps) * (width - left - right);
  const y = (p) => top + (hi - p) * rowH;

  for (let p = lo; p <= hi; p++) {
    if (p % 12 === 0) {
      el("line", { x1: left, x2: width - right, y1: y(p) + rowH, y2: y(p) + rowH, class: "grid" }, svg);
      el("text", { x: left - 4, y: y(p) + rowH, class: "axis", "text-anchor": "end", "dominant-baseline": "middle" }, svg)
        .textContent = pitchName(p);
    }
  }
  for (let b = 0; b <= totalBars; b++) {
    el("line", { x1: x(b * stepsPerBar), x2: x(b * stepsPerBar), y1: top, y2: top + rows * rowH,
      class: b === promptBars ? "split" : "grid" }, svg);
    if (b < totalBars && (!narrow || b % 2 === 0)) {
      el("text", { x: x(b * stepsPerBar) + 2, y: height - 4, class: "axis" }, svg).textContent = b + 1;
    }
  }
  if (promptBars > 0) {
    el("text", { x: x(promptBars * stepsPerBar) - 4, y: 11, class: "axis", "text-anchor": "end" }, svg)
      .textContent = "prompt";
  }
  el("text", { x: x(promptBars * stepsPerBar) + 4, y: 11, class: "axis accent" }, svg).textContent = "model";

  const offset = promptBars * stepsPerBar;
  const box = (n, cls, shift = 0) => el("rect", {
    x: fmt(x(n.start + shift)), y: y(n.pitch) + 0.5,
    width: fmt(Math.max(1.5, x(n.start + shift + n.duration) - x(n.start + shift) - 1)),
    height: rowH - 1, class: cls,
  }, svg);
  for (const n of truth) box(n, "note truth", offset);
  for (const n of prompt) box(n, "note prompt");
  for (const n of generated) box(n, "note gen");

  const head = el("line", { y1: top, y2: top + rows * rowH, class: "playhead", visibility: "hidden" }, svg);
  return (step) => {
    if (step === null || step > steps) { head.setAttribute("visibility", "hidden"); return; }
    head.setAttribute("visibility", "visible");
    head.setAttribute("x1", x(step));
    head.setAttribute("x2", x(step));
  };
}

const GROUPS = [[BAR, "BAR"], [POS, "POS"], [VEL, "VEL"], [PITCH, "PITCH"], [DUR, "DUR"]];

// Every token's probability under the model, grouped by token type. Tokens the grammar
// rules out are drawn faint; the drawn token is marked.
export function drawVocab(svg, step, tok) {
  const width = Math.max(280, Math.floor(svg.parentElement.clientWidth));
  const height = 96, top = 6, base = 72;
  reset(svg, width, height);
  const ids = [];
  for (const [type] of GROUPS) for (let i = 0; i < tok.vocabSize; i++) if (tok.types[i] === type) ids.push(i);
  const gap = 6, barSlot = 26; // the lone BAR token gets room for its label
  const unit = (width - gap * (GROUPS.length - 1) - barSlot) / (ids.length - 1);
  let max = 0;
  for (const i of ids) max = Math.max(max, step.model[i]);
  let xPos = 0, prevType = null, groupStart = 0;
  const label = (type, start, end) => {
    const name = GROUPS.find((g) => g[0] === type)[1];
    el("text", { x: (start + end) / 2, y: height - 4, class: "axis", "text-anchor": "middle" }, svg).textContent = name;
    el("line", { x1: start, x2: end, y1: base + 0.5, y2: base + 0.5, class: "grid" }, svg);
  };
  for (const i of ids) {
    const type = tok.types[i];
    if (prevType !== null && type !== prevType) {
      label(prevType, groupStart, xPos);
      xPos += gap;
      groupStart = xPos;
    }
    prevType = type;
    const slot = type === BAR ? barSlot : unit;
    const bx = xPos + (slot - unit) / 2;
    const h = (step.model[i] / max) * (base - top);
    const cls = i === step.chosen ? "bar chosen" : step.allowed[i] ? "bar ok" : step.grammar[i] ? "bar keyed" : "bar off";
    if (h > 0.2) el("rect", { x: fmt(bx), y: fmt(base - h), width: fmt(Math.max(0.8, unit - 0.4)), height: fmt(h), class: cls }, svg);
    if (i === step.chosen) {
      el("path", { d: `M${fmt(bx + unit / 2 - 4)},${base + 8} l4,-6 l4,6 z`, class: "marker" }, svg);
    }
    xPos += slot;
  }
  label(prevType, groupStart, xPos);
}

// Correlation of the prompt's pitch-class durations with all 24 Krumhansl-Kessler keys.
export function drawKeyChart(svg, corr, best) {
  const width = Math.max(280, Math.floor(svg.parentElement.clientWidth));
  const rowH = 92, left = 40, top = 6;
  const height = top + 2 * rowH + 14;
  reset(svg, width, height);
  const colW = (width - left) / 12;
  ["major", "minor"].forEach((mode, m) => {
    const zero = top + m * rowH + rowH / 2;
    el("text", { x: 0, y: zero, class: "axis", "dominant-baseline": "middle" }, svg).textContent = mode;
    el("line", { x1: left, x2: width, y1: zero, y2: zero, class: "grid" }, svg);
    for (let t = 0; t < 12; t++) {
      const r = corr[m][t];
      const h = Math.abs(r) * (rowH / 2 - 4);
      const isBest = best && best.tonic === t && best.mode === mode;
      el("rect", { x: fmt(left + t * colW + colW * 0.2), y: fmt(r >= 0 ? zero - h : zero),
        width: fmt(colW * 0.6), height: fmt(Math.max(0.5, h)), class: isBest ? "bar chosen" : "bar ok" }, svg);
    }
  });
  for (let t = 0; t < 12; t++) {
    el("text", { x: left + (t + 0.5) * colW, y: height - 2, class: "axis", "text-anchor": "middle" }, svg)
      .textContent = PITCH_CLASS_NAMES[t];
  }
}
