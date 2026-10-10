// Port of src/musegen/theory.py: keys, scales and Krumhansl-Schmuckler key finding.
// A note is {start, pitch, duration, velocity}, with start and duration in 16th-note steps.

export const PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// Krumhansl & Kessler (1982) probe-tone ratings, tonic first.
export const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
export const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

const MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11];
// Natural minor plus the harmonic-minor leading tone, as in theory.py.
const MINOR_SCALE_EXTENDED = [0, 2, 3, 5, 7, 8, 10, 11];

export class Key {
  constructor(tonic, mode, confidence = 1) {
    if (mode !== "major" && mode !== "minor") throw new Error(`bad mode ${mode}`);
    this.tonic = tonic;
    this.mode = mode;
    this.confidence = confidence;
    const scale = mode === "major" ? MAJOR_SCALE : MINOR_SCALE_EXTENDED;
    this.pcs = new Set(scale.map((s) => (tonic + s) % 12));
  }
  get name() {
    return `${PITCH_CLASS_NAMES[this.tonic]} ${this.mode}`;
  }
  contains(pitch) {
    return this.pcs.has(((pitch % 12) + 12) % 12);
  }
}

export function pitchClassHistogram(notes, weighted = true) {
  const hist = new Array(12).fill(0);
  for (const n of notes) hist[n.pitch % 12] += weighted ? n.duration : 1;
  return hist;
}

function pearson(a, b) {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return num / Math.sqrt(da * db);
}

function roll(profile, shift) {
  return profile.map((_, i) => profile[(i - shift + 12) % 12]);
}

// Correlation with all 24 keys: [major[12], minor[12]], indexed by tonic.
export function keyCorrelations(hist) {
  const out = [new Array(12).fill(0), new Array(12).fill(0)];
  const total = hist.reduce((a, b) => a + b, 0);
  if (total === 0 || hist.every((v) => Math.abs(v - hist[0]) < 1e-12)) return out;
  [MAJOR_PROFILE, MINOR_PROFILE].forEach((profile, m) => {
    for (let t = 0; t < 12; t++) out[m][t] = pearson(hist, roll(profile, t));
  });
  return out;
}

export function estimateKey(notes) {
  const corr = keyCorrelations(pitchClassHistogram(notes));
  let best = null;
  for (let m = 0; m < 2; m++) {
    for (let t = 0; t < 12; t++) {
      if (corr[m][t] !== 0 && (best === null || corr[m][t] > best.r)) best = { m, t, r: corr[m][t] };
    }
  }
  return best ? new Key(best.t, best.m === 0 ? "major" : "minor", best.r) : null;
}

export function parseKey(text) {
  const m = /^\s*([A-Ga-g])([#b]?)\s*(major|minor)?\s*$/.exec(text);
  if (!m) throw new Error(`cannot parse key ${text}`);
  const flats = { DB: 1, EB: 3, GB: 6, AB: 8, BB: 10, CB: 11, FB: 4 };
  const name = (m[1] + m[2]).toUpperCase();
  const tonic = name in flats ? flats[name] : PITCH_CLASS_NAMES.indexOf(name);
  return new Key(tonic, m[3] === "minor" ? "minor" : "major");
}

export function pitchName(pitch) {
  return `${PITCH_CLASS_NAMES[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
}
