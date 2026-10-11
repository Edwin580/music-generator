// Port of the per-piece metrics in src/musegen/evaluation/metrics.py.

import { Key, estimateKey, keyCorrelations, pitchClassHistogram } from "./theory.js";

export function sliceBars(notes, startBar, endBar, stepsPerBar = 16) {
  const lo = startBar * stepsPerBar, hi = endBar * stepsPerBar;
  return notes
    .filter((n) => n.start >= lo && n.start < hi)
    .map((n) => ({ ...n, start: n.start - lo, duration: Math.min(n.duration, hi - n.start) }));
}

// Highest note per onset: the line a listener tracks as melody.
function melodicLine(notes) {
  const top = new Map();
  for (const n of notes) if (!top.has(n.start) || n.pitch > top.get(n.start).pitch) top.set(n.start, n);
  return [...top.keys()].sort((a, b) => a - b).map((s) => top.get(s));
}

export function scaleConsistency(notes) {
  if (!notes.length) return 0;
  let best = 0;
  for (const mode of ["major", "minor"]) {
    for (let t = 0; t < 12; t++) {
      const key = new Key(t, mode);
      best = Math.max(best, notes.filter((n) => key.contains(n.pitch)).length / notes.length);
    }
  }
  return best;
}

export function keyClarity(notes) {
  const corr = keyCorrelations(pitchClassHistogram(notes));
  return Math.max(...corr[0], ...corr[1]);
}

export function repetitionRatio(notes, n = 4) {
  const line = melodicLine(notes);
  if (line.length <= n) return 0;
  const events = [];
  for (let i = 0; i + 1 < line.length; i++) events.push(`${line[i + 1].pitch - line[i].pitch}:${line[i].duration}`);
  const grams = [];
  for (let i = 0; i + n <= events.length; i++) grams.push(events.slice(i, i + n).join(" "));
  const counts = new Map();
  for (const g of grams) counts.set(g, (counts.get(g) || 0) + 1);
  let repeated = 0;
  for (const c of counts.values()) if (c > 1) repeated += c;
  return repeated / grams.length;
}

export function pieceMetrics(notes, nBars) {
  if (!notes.length) return null;
  const line = melodicLine(notes);
  const intervals = [];
  for (let i = 0; i + 1 < line.length; i++) intervals.push(Math.abs(line[i + 1].pitch - line[i].pitch));
  if (!intervals.length) intervals.push(0);
  const pitches = notes.map((n) => n.pitch);
  return {
    nNotes: notes.length,
    notesPerBar: notes.length / Math.max(1, nBars),
    pitchRange: Math.max(...pitches) - Math.min(...pitches),
    scaleConsistency: scaleConsistency(notes),
    keyClarity: keyClarity(notes),
    key: estimateKey(notes),
    meanInterval: intervals.reduce((a, b) => a + b, 0) / intervals.length,
    largeLeapRate: intervals.filter((i) => i > 7).length / intervals.length,
    repetition: repetitionRatio(notes),
  };
}

export function mean(values) {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : NaN;
}
